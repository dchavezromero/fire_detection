# UBC Building Classification — Training & Inference

Cascade Mask R-CNN training and inference for the UBC (Urban Building Classification) dataset, built on [MMDetection 3.3.0](https://mmdetection.readthedocs.io/en/v3.3.0/). Part of a larger fire detection pipeline — when YOLO + FFireNet confirm a fire alert, the system queries the inference service with the drone's GPS coordinates to identify buildings in the surrounding area and their type, informing fire severity assessment and suppression strategy.

## Overview

This branch contains the full replication of Huang et al. (2022) CVPRW on the public UBC v1 dataset across all three classification tasks (roof_coarse, roof_fine, use_coarse), plus a FastAPI inference service (`ubc_server/`) and a standalone Windows client (`ubc_client.py`) that demonstrates end-to-end classification from a GPS coordinate.

### Results Summary

All three UBC v1 tasks trained for 100 epochs each on single-GPU RTX 5080, Cascade Mask R-CNN R-50-FPN, batch size 6, LR 0.0075 (linear-scaled from paper's 8-GPU recipe), multi-scale training between 400–900 px. Final validation metrics:

| Task | Classes | Bbox mAP | Bbox AP@50 | Segm mAP | Segm AP@50 | Paper Cascade bbox mAP |
|------|---------|----------|------------|----------|------------|------------------------|
| **roof_coarse** | 5 | **0.150** | **0.245** | **0.140** | **0.246** | — (not reported) |
| use_coarse | 5 | 0.109 | 0.175 | 0.101 | 0.173 | 0.136 |
| roof_fine | 11 | 0.095 | 0.152 | 0.088 | 0.151 | 0.153 |

Replication ratio ranges from 62% (roof_fine) to 80% (use_coarse). The gap scales inversely with task granularity and is attributable to single-GPU effective batch size (6 vs paper's 16) and val-vs-test split differences (paper evaluates on the withheld test split). Per-class rankings match the paper qualitatively across all three tasks.

**use_coarse** is the primary model for the fire pipeline integration — building function (residential/commercial/industrial/public) is the most fire-response-relevant signal. **roof_coarse** is available as a secondary view.

## Requirements

- Python 3.10
- NVIDIA GPU (training: any CUDA-capable; inference tested on Ampere sm_86+)
- Ubuntu 22.04+ (dual-boot native preferred; WSL2 works but slower)
- NVIDIA driver 525+
- Miniconda

GPU-specific environment differences are documented separately below — **Blackwell (RTX 50-series)** requires source-building MMCV, while **Ampere (RTX 30/40-series)** uses prebuilt wheels.

## Dataset

[UBC v1](https://github.com/CityDevelopmentLab/UBC_dataset) (Huang et al., 2022) — 800 satellite image tiles of Beijing + Munich at 600×600 px, 0.5–0.8 m/px GSD. Authors release train (560) + val (160) splits; test split (80) is withheld. Three parallel COCO-format annotation sets.

Place the dataset under `training_dataset/`:

```
training_dataset/UBC_v1.0/
├── annotations/
│   ├── roof_coarse_{train,val}.json   5 classes (flat, gable, hipped, arched, other)
│   ├── roof_fine_{train,val}.json    11 classes (fine-grained taxonomy)
│   └── use_coarse_{train,val}.json    5 classes (residential, commercial, industrial, public, other)
├── train/   560 × .tif tiles + .xml sidecars
└── val/     160 × .tif tiles + .xml sidecars
```

Training assumes the dataset lives at `~/UBC_v1.0/`. Move or symlink as appropriate:

```bash
ln -s /path/to/training_dataset/UBC_v1.0 ~/UBC_v1.0
```

Inference does not need the dataset at runtime.

## Project Structure

```
├── training/
│   └── README.md                       # Documents training process; configs live in ubc_server/
├── ubc_server/
│   ├── configs/                        # Shared: training + runtime
│   │   ├── _base_/                     # MMDetection base configs
│   │   └── ubc/                        # UBC task configs (4 files)
│   ├── checkpoints/                    # .gitignored — ~1 GB of trained weights
│   ├── mmdet_plugins/
│   │   └── ubc.py                      # UBC dataset class (shared: training + runtime)
│   ├── ubc_inference.py                # Core inference library
│   └── server.py                       # FastAPI wrapper
├── training_dataset/                   # UBC v1 data (structure documented above)
├── ubc_client.py                       # Windows client
└── README.md
```

---

# Training

Training reproduces Huang et al.'s Cascade Mask R-CNN setup on the public UBC v1 train/val split. Each task takes **~2.5–3.5 hours** on a single RTX 5080 at batch size 6; total across all three tasks is **~8–9 hours**.

## GPU environment setup

The MMDetection install path depends on which NVIDIA GPU you're using. Pick the matching section.

<details>
<summary><b>Option A — Blackwell (RTX 50-series, sm_120) — primary training host</b></summary>

Used for our training runs on an RTX 5080, 16 GB VRAM. Blackwell's sm_120 compute capability requires CUDA 12.8+ and a source-built MMCV — prebuilt wheels don't include sm_120 kernels.

### 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl ninja-build \
    libgl1 libglib2.0-0 software-properties-common ca-certificates
```

### 2. CUDA Toolkit 12.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify: `nvcc --version` → `release 12.8`.

### 3. Conda env + PyTorch 2.11 + cu128

```bash
conda create -n ubc_mmdet python=3.10 -y
conda activate ubc_mmdet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Critical checkpoint — `sm_120` must appear in arch list:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# Expect: 2.11.0+cu128 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']

python -c "import torch; a=torch.randn(1000,1000,device='cuda'); b=torch.randn(1000,1000,device='cuda'); print((a@b).shape)"
# Expect: torch.Size([1000, 1000]) — no 'no kernel image' error
```

### 4. MMEngine + source-built MMCV

```bash
pip install -U "setuptools>=64,<80"
pip install -U openmim
mim install mmengine

# Source-build MMCV with Blackwell arch
cd ~
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v2.1.0

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;12.0"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
pip install . -v --no-build-isolation
```

Build takes 5–15 minutes. Watch nvcc output for `compute_120,code=sm_120` confirmation.

### 5. MMDetection + remaining deps

```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v3.3.0
pip install -v -e . --no-build-isolation

pip install ftfy regex tensorboard
pip install "numpy<2.0"
```

### 6. Smoke test

```bash
python -c "
from mmcv.ops import box_iou_rotated
import torch
b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5]]).cuda()
b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0, 0.3]]).cuda()
print(box_iou_rotated(b1, b2))
# Expect: tensor([[0.3708]], device='cuda:0')
"
python -c "import mmdet; print(mmdet.__version__)"
# Expect: 3.3.0
```

</details>

<details>
<summary><b>Option B — Ampere (RTX 30/40-series, sm_86/sm_89) — alternate training or inference host</b></summary>

Used for our inference deployment on an RTX 3070 Ti Laptop (Ampere, sm_86, 8 GB VRAM). Ampere has prebuilt MMCV wheels available, which dramatically simplifies the install. This same setup also works for training on Ampere hardware (expect longer wall-clock than Blackwell — probably 5–8 hours per task at batch 4 on an 8 GB card).

### 1. Conda env + anti-pollution setting

```bash
conda create -n ubc_mmdet python=3.10 -y
conda activate ubc_mmdet

# Disable user-site to prevent ~/.local/lib/ pollution from stale pip installs
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh
conda deactivate && conda activate ubc_mmdet
```

### 2. PyTorch 2.1.0 + cu121

Pin torch to 2.1.0 because MMCV 2.1.0 has prebuilt wheels for exactly this version. Newer torch forces a source build.

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install --force-reinstall nvidia-nvjitlink-cu12==12.1.105

# Persist nvidia lib path
cat > $CONDA_PREFIX/etc/conda/activate.d/nvidia_libs.sh << 'EOF'
export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -type d -name "lib" | tr '\n' ':')$LD_LIBRARY_PATH
EOF
conda deactivate && conda activate ubc_mmdet
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# Expect: 2.1.0+cu121 [..., 'sm_86', ...]
python -c "import torch; a=torch.randn(1000,1000,device='cuda'); print((a@a).shape)"
```

### 3. MMEngine + prebuilt MMCV

```bash
pip install -U "setuptools>=64,<80"
pip install -U openmim
mim install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

MMCV downloads a ~80 MB wheel — no compilation.

### 4. MMDetection + remaining deps

```bash
cd ~
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v3.3.0
pip install -v -e . --no-build-isolation

pip install ftfy regex tensorboard
pip install "numpy<2.0"
```

### 5. Smoke test

Same as Blackwell section — `box_iou_rotated` should produce `tensor([[0.3708]], device='cuda:0')`.

</details>

## Install UBC dataset class + configs

MMDetection needs to know about the UBC dataset classes. Drop the plugin file into MMDetection's datasets directory and register it in `__init__.py`, then copy configs into place.

```bash
# UBC dataset class
cp ubc_server/mmdet_plugins/ubc.py ~/mmdetection/mmdet/datasets/

# Register in __init__.py
python3 << EOF
import os
path = os.path.expanduser('~/mmdetection/mmdet/datasets/__init__.py')
with open(path) as f:
    content = f.read()
if 'UBCRoofFineDataset' not in content:
    import_line = 'from .ubc import UBCRoofFineDataset, UBCRoofCoarseDataset, UBCUseCoarseDataset\n'
    lines = content.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith('from .'):
            insert_at = i + 1
    lines.insert(insert_at, import_line)
    content = ''.join(lines)
    content = content.replace(
        "__all__ = [",
        "__all__ = [\n    'UBCRoofFineDataset', 'UBCRoofCoarseDataset', 'UBCUseCoarseDataset',", 1)
    with open(path, 'w') as f:
        f.write(content)
    print("Registered.")
else:
    print("Already registered.")
EOF

# Verify
python -c "from mmdet.datasets import UBCRoofFineDataset; print(UBCRoofFineDataset.METAINFO['classes'])"
# Expect: 11-class tuple

# Configs
mkdir -p ~/mmdetection/configs/ubc
cp ubc_server/configs/ubc/*.py ~/mmdetection/configs/ubc/
```

## Training configuration

All task configs inherit from `cascade-mask-rcnn_r50_fpn_ubc_base.py`:

```python
MAX_EPOCHS = 100
VAL_INTERVAL = 5
LR_STEP_EPOCHS = [int(MAX_EPOCHS * 0.6), int(MAX_EPOCHS * 0.9)]   # 60, 90

data_root = '/home/dennis/UBC_v1.0/'   # edit to match your host

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomResize', scale=[(400, 400), (900, 900)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(batch_size=6, ...)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco.pth'
```

Task-specific configs override `num_classes` + dataset type + annotation filenames. No other changes needed.

## Training runs

### 1. Pretrained weight

Download Cascade Mask R-CNN's COCO pretrained checkpoint (used as training starting point):

```bash
cd ~/mmdetection
mkdir -p checkpoints
wget -O checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco.pth \
  https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth
```

### 2. 1-epoch sanity run (strongly recommended before full training)

Before committing to a ~3 hour run, verify the pipeline end-to-end:

```bash
sed -i 's/^MAX_EPOCHS = .*/MAX_EPOCHS = 1/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
sed -i 's/^VAL_INTERVAL = .*/VAL_INTERVAL = 1/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py

cd ~/mmdetection
rm -rf work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine
python tools/train.py configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_fine.py
```

Success criteria: 5–10 min runtime, completes without error, produces an 11-row per-class AP table, saves a best checkpoint. Overall mAP will be near zero — that's expected at 1 epoch, structural validation is the goal.

### 3. Full 100-epoch runs

Restore the full schedule:

```bash
sed -i 's/^MAX_EPOCHS = .*/MAX_EPOCHS = 100/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
sed -i 's/^VAL_INTERVAL = .*/VAL_INTERVAL = 5/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py

grep -E "^(MAX_EPOCHS|VAL_INTERVAL)" ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
# Expect: MAX_EPOCHS = 100, VAL_INTERVAL = 5
```

Launch each task (they can be run back-to-back overnight):

```bash
cd ~/mmdetection

# Task 1 — roof_coarse (best result)
rm -rf work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse
python tools/train.py configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse.py 2>&1 \
    | tee work_dirs/roof_coarse_run.log

# Task 2 — use_coarse (primary demo task)
rm -rf work_dirs/cascade-mask-rcnn_r50_fpn_ubc_use_coarse
python tools/train.py configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_use_coarse.py 2>&1 \
    | tee work_dirs/use_coarse_run.log

# Task 3 — roof_fine (paper replication baseline)
rm -rf work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine
python tools/train.py configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_fine.py 2>&1 \
    | tee work_dirs/roof_fine_run.log
```

### 4. TensorBoard monitoring (optional)

In a second terminal during training:

```bash
conda activate ubc_mmdet
tensorboard --logdir ~/mmdetection/work_dirs/
# Open http://localhost:6006 in a browser
```

Useful for spotting loss divergence or stuck classes without watching the console log.

## Per-task training budgets

Measured on RTX 5080 (Blackwell, 16 GB VRAM):

| Task | Time | Best epoch | Final Segm mAP |
|------|------|------------|----------------|
| roof_coarse | ~3 h | 75 | 0.140 |
| roof_fine | ~3 h | 75 | 0.088 |
| use_coarse | ~2.5 h | 50 | 0.101 |

Total: ~8–9 hours across all three tasks. On Ampere (RTX 30-series), expect roughly 1.5–2× these times depending on VRAM available; may need to drop batch size to 4 on 8 GB cards.

## Training observations

- **Three-task gap pattern.** roof_fine → 62% of paper AP; use_coarse → 80%; roof_coarse → estimated 65-70% (paper doesn't report this task). The gap scales with how much the task benefits from large effective batch size.
- **use_coarse converges fastest.** Best checkpoint at epoch 50 vs epoch 75 for roof tasks. Consistent with the paper's observation that function classification saturates earlier — the latent visual signal for "residential vs commercial" is less granular than roof geometry.
- **Per-class rankings match the paper.** Dominant classes (flat, gable, hipped_v2 for roof; residential for use) learn well; rare classes (shed_roof, pinnacle_roof, industrial) consistently underperform in both our runs and the paper's.

## After training

Copy best checkpoints to `ubc_server/checkpoints/` with short names:

```bash
cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse/best_coco_segm_mAP_epoch_*.pth \
   ubc_server/checkpoints/roof_coarse.pth
cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine/best_coco_segm_mAP_epoch_*.pth \
   ubc_server/checkpoints/roof_fine.pth
cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_use_coarse/best_coco_segm_mAP_epoch_*.pth \
   ubc_server/checkpoints/use_coarse.pth
```

If inference runs on a different host, transfer the three `.pth` files via thumb drive or scp.

## Training troubleshooting

### `Cannot use unregistered type UBCRoofFineDataset`

Plugin registration didn't take effect. Verify with `python -c "from mmdet.datasets import UBCRoofFineDataset"`. Re-run the registration snippet above if this fails.

### `ERROR: Project ... uses a build backend that is missing the 'build_editable' hook`

Setuptools too old (<64). `pip install -U "setuptools>=64,<80"`.

### `ModuleNotFoundError: No module named 'pkg_resources'` during MMCV build

Pip's build isolation used setuptools 82+. Add `--no-build-isolation`.

### `RuntimeError: CUDA error: no kernel image is available for execution on the device`

PyTorch or MMCV was built without support for your GPU's compute capability. For Blackwell, confirm `'sm_120'` in `torch.cuda.get_arch_list()` and rebuild MMCV with `TORCH_CUDA_ARCH_LIST="12.0"` explicit.

### OpenCV TIFF warnings: `Unknown field with tag 33550/33922/34735/34737`

Harmless. UBC tiles are GeoTIFFs carrying geospatial metadata OpenCV's reader doesn't recognize. Silence with `export OPENCV_LOG_LEVEL=ERROR`.

### Loss NaN or exploding early in training

Most commonly LR too high or bad data. With batch 6 and LR 0.0075, this shouldn't happen on UBC. If it does, drop LR to 0.0025 as first try.

---

# Inference

Inference runs as a FastAPI service on any GPU host. Ampere or newer recommended (Blackwell also works but requires the source-built MMCV from the training-host recipe).

## Server setup

Follow the **Ampere (Option B)** environment setup above, then add server-specific dependencies:

```bash
pip install fastapi "uvicorn[standard]" python-multipart opencv-python requests
```

Populate `ubc_server/checkpoints/` with the three trained `.pth` files (as in the "After training" step).

## Running the server

```bash
cd ubc_server
python server.py --warmup use_coarse
```

Binds to `0.0.0.0:8000`. `--warmup` eagerly loads one or more task models (~3 s per model) to eliminate first-request latency. All three models fit in 8 GB VRAM.

Other launch options:

```bash
python server.py --warmup use_coarse roof_coarse   # preload two models
python server.py --port 9000                       # custom port
python server.py                                   # lazy load (loads on first request per task)
```

## Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness + cached task list |
| `/tasks` | GET | List tasks + class names + load status |
| `/classify/gps` | POST | `{lat, lon, task}` → annotated image + detections |
| `/classify/image` | POST | Upload image → annotated image + detections (bypasses Esri) |

Sample:

```bash
curl -X POST http://192.168.1.11:8000/classify/gps \
    -H "Content-Type: application/json" \
    -d '{"lat": 48.1374, "lon": 11.5755, "task": "use_coarse"}'
```

Response shape:

```json
{
  "detections": [
    {"class_id": 0, "class_name": "residential", "score": 0.94, "bbox_xyxy": [...]},
    ...
  ],
  "annotated_image_b64": "iVBORw0KGgo...",
  "raw_image_b64": "iVBORw0KGgo...",
  "meta": {"inference_sec": 0.098, "total_sec": 1.593, ...}
}
```

## Client

Run on Windows inside the fire_detection venv (with `pip install requests` added):

```bash
python ubc_client.py --lat 48.1374 --lon 11.5755 --task use_coarse
```

Pings the server, sends the coordinates, opens an OpenCV window with the annotated tile, prints per-class detection summary to stdout.

Other client modes:

```bash
# Switch tasks
python ubc_client.py --lat 48.1374 --lon 11.5755 --task roof_coarse

# Save without display (for batch report figures)
python ubc_client.py --lat 42.2626 --lon -71.8023 --task use_coarse \
    --save figures/worcester.png --no-display

# Different server host
python ubc_client.py --lat 48.1374 --lon 11.5755 --server http://192.168.1.11:8000
```

## Performance

Measured on RTX 3070 Ti Laptop GPU (Ampere, sm_86, 8 GB VRAM):

| Stage | Latency |
|-------|---------|
| First per-task request (model load) | ~3–4 s |
| Cached-model inference | ~100 ms |
| Esri imagery fetch (600×600 PNG) | ~1–1.5 s |
| Total end-to-end `/classify/gps` | ~1.5–2 s |

On Blackwell (5080), inference latency drops to ~40-60 ms but Esri fetch time dominates, so total roundtrip is similar.

## Good demo coordinates

| Location | Lat, Lon | Shows |
|----------|----------|-------|
| Munich Altstadt | `48.1374, 11.5755` | Dense historic core, mixed gable/hipped, public dominant |
| Munich suburb | `48.1100, 11.5900` | Apartment blocks, flat roofs, residential dominant |
| Beijing Chaoyang | `39.9200, 116.4500` | Modern residential high-rise |
| Beijing Hutongs | `39.9250, 116.3800` | Traditional courtyard architecture |
| Worcester, MA | `42.2626, -71.8023` | Out-of-training-distribution test |

Running all three tasks at one location produces a useful comparison figure.

## Inference troubleshooting

### `[client] ERROR: could not reach server`

- Server running? `curl http://localhost:8000/health` on server host should return JSON
- Firewall: `sudo ufw allow 8000/tcp` on the server host
- LAN IP changed? DHCP can reassign — check current IP with `ip addr show`; consider router DHCP reservation

### `ImportError: libnvJitLink.so.12`

CUDA runtime library missing. `pip install --force-reinstall nvidia-nvjitlink-cu12==12.1.105` and verify `LD_LIBRARY_PATH` includes the nvidia lib dirs.

### `ModuleNotFoundError: No module named 'typing_extensions'` (and related)

User-site pollution from stale `~/.local/lib/python3.10/`. Ensure `PYTHONNOUSERSITE=1` is set (see Ampere setup step 1), then reinstall affected packages.

### Server returns 502 "Esri image fetch failed"

Esri's free tier occasionally rate-limits. Retry after 30 seconds. For higher throughput, use an Esri developer account with an API key (modify `fetch_esri_image()` in `ubc_inference.py`).

---

# Integration with the Fire Detection Pipeline

When the pipeline on the `main` branch fires a fire alert, it queries the inference server with the drone's GPS coordinates to retrieve building context for the affected area. The architecture is deliberately split across two machines:

- **Windows desktop** runs YOLO + FFireNet in real-time
- **Ubuntu laptop** runs Cascade Mask R-CNN as a reachable service

Rationale:

1. **Compute offload.** Cascade Mask R-CNN's ~100 ms inference is the heaviest stage. Keeping it off the fire-detection machine preserves GPU memory and scheduling for the real-time path.
2. **Hardware availability.** Training on Blackwell (RTX 5080) via dual-boot Ubuntu; inference on Ampere (RTX 3070 Ti Laptop) where MMDetection's standard prebuilt wheels work without source-building.
3. **Deployment parity.** In a real drone system, the inference service would run in the cloud rather than on the drone or ground station. The LAN-based demo mirrors that request shape exactly.
4. **Matches proposal.** Phase 2 proposal Figure 11 describes this as a service call triggered when fire is localized.

To integrate into the real-time fire pipeline, the conviction tracker calls `classify_gps()` from `ubc_client.py` when `alert_triggered` transitions to True, passing the drone's current (lat, lon). The returned annotated image and detection list can be overlaid into the pipeline's display or logged for post-incident review.

# Class Mapping

**roof_coarse** (5 classes):
| ID | Name |
|----|------|
| 0 | flat |
| 1 | gable |
| 2 | hipped |
| 3 | arched |
| 4 | other |

**roof_fine** (11 classes):
| ID | Name |
|----|------|
| 0 | flat |
| 1 | flat_roof_complex |
| 2 | shed_roof |
| 3 | gable_roof |
| 4 | gambrel_roof |
| 5 | hipped_roof_v1 |
| 6 | hipped_roof_v2 |
| 7 | mansard_roof |
| 8 | pinnacle_roof |
| 9 | arched |
| 10 | other |

**use_coarse** (5 classes):
| ID | Name |
|----|------|
| 0 | residential |
| 1 | commercial |
| 2 | industrial |
| 3 | public |
| 4 | other |

Mappings declared in [`ubc_server/mmdet_plugins/ubc.py`](ubc_server/mmdet_plugins/ubc.py) via `METAINFO` and must match the training annotation files.

# References

- Cai, Z., & Vasconcelos, N. (2019). Cascade R-CNN: High quality object detection and instance segmentation. *IEEE TPAMI, 43*(5), 1483–1498.
- Chen, K., Wang, J., Pang, J., et al. (2019). MMDetection: Open MMLab detection toolbox and benchmark. *arXiv:1906.07155*.
- Huang, X., Ren, L., Liu, C., et al. (2022). Urban Building Classification (UBC): A dataset for individual building detection and classification. *CVPRW 2022*, 1412–1420.

# License

Research and educational purposes.