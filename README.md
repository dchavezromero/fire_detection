# UBC Building Classification — Training & Inference Module

Cascade Mask R-CNN training pipeline and importable inference module for the UBC (Urban Building Classification) dataset, built on [MMDetection 3.3.0](https://mmdetection.readthedocs.io/en/v3.3.0/). Part of a larger fire detection pipeline — when YOLO + FFireNet confirm a fire alert, the main-branch pipeline imports this module to identify buildings around the drone's GPS coordinates and their type, informing fire severity assessment and suppression strategy.

## Overview

This branch contains the full replication of Huang et al. (2022) CVPRW on the public UBC v1 dataset across all three classification tasks (roof_coarse, roof_fine, use_coarse), an importable Python module (`ubc_classifier/`) that provides Esri imagery fetch + inference + visualization functions, and a standalone demo script (`examples/ubc_inference_from_gps.py`) that runs the full GPS-to-classification flow from the command line.

The classifier runs **in-process** as a module on the same host as the fire detection pipeline. There is no separate server.

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
- NVIDIA GPU (training: any CUDA-capable; inference: any working CUDA setup)
- Ubuntu 22.04+ (dual-boot native preferred; WSL2 works but slower)
- NVIDIA driver 525+
- Miniconda
- ≥16 GB system RAM recommended for the install (use `--no-cache-dir` on lower-RAM systems)

GPU-specific environment differences are documented separately below — **Blackwell (RTX 50-series)** requires source-building MMCV; **Ampere or older (RTX 30/40-series, GTX 16-series, etc.)** uses prebuilt wheels.

**If you only want to run inference**, you can skip the entire training pipeline — download our pretrained checkpoints (see [Pretrained weights](#pretrained-weights)) and jump to the [Inference](#inference) section.

## Dataset

[UBC v1](https://github.com/CityDevelopmentLab/UBC_dataset) (Huang et al., 2022) — 800 satellite image tiles of Beijing + Munich at 600×600 px, 0.5–0.8 m/px GSD. Authors release train (560) + val (160) splits; test split (80) is withheld. Three parallel COCO-format annotation sets.

The dataset ships with this branch under `training_datasets/`:

```
training_datasets/UBC_v1.0/
├── annotations/
│   ├── roof_coarse_{train,val}.json   5 classes (flat, gable, hipped, arched, other)
│   ├── roof_fine_{train,val}.json    11 classes (fine-grained taxonomy)
│   └── use_coarse_{train,val}.json    5 classes (residential, commercial, industrial, public, other)
├── train/   560 × .tif tiles + .xml sidecars
└── val/     160 × .tif tiles + .xml sidecars
```

The training configs reference `~/UBC_v1.0/` as the dataset root. Point that path at the repo's dataset via a symlink (run from the repo root):

```bash
ln -s "$(pwd)/training_datasets/UBC_v1.0" ~/UBC_v1.0

# Verify — should list six JSON files
ls ~/UBC_v1.0/annotations/
```

Alternative: edit `data_root` in `ubc_classifier/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py` to point at your actual dataset path.

Inference does not need the dataset at runtime.

## Pretrained weights

The three trained checkpoints from our RTX 5080 training runs are available as a pre-packaged download, for users who want to run inference without reproducing training.

**Download:** [Mega folder](https://mega.nz/folder/sBZDTYAD#8vHE33KrOt_M20QyJQnl3Q)

Contents:

| File | Size | Task | Best epoch | Final Segm mAP |
|------|------|------|------------|-----------------|
| `roof_coarse.pth` | ~310 MB | 5-class roof geometry | 75 | 0.140 |
| `roof_fine.pth` | ~310 MB | 11-class fine-grained roof | 75 | 0.088 |
| `use_coarse.pth` | ~310 MB | 5-class building function | 50 | 0.101 |

After downloading, place all three files in `ubc_classifier/checkpoints/`:

```bash
mkdir -p ubc_classifier/checkpoints
mv ~/Downloads/roof_coarse.pth ubc_classifier/checkpoints/
mv ~/Downloads/roof_fine.pth   ubc_classifier/checkpoints/
mv ~/Downloads/use_coarse.pth  ubc_classifier/checkpoints/

# Verify
ls -la ubc_classifier/checkpoints/
# Expect 3 files, ~310 MB each
```

**Note on the `main` branch convention:** the integrated fire-detection pipeline on the `main` branch expects UBC checkpoints under `models/ubc_*_600imgsz_100epochs/<task>.pth` instead, matching the YOLO/FFireNet directory pattern. If you'll be using both branches, populate both locations or symlink between them.

With the checkpoints in place, skip directly to [Inference](#inference) — you don't need any of the training setup.

## Project Structure

```
├── training/
│   └── README.md                       # Documents training process; configs live in ubc_classifier/
├── ubc_classifier/                     # Importable Python module
│   ├── configs/                        # Shared: training + runtime
│   │   ├── _base_/                     # Minimal MMDetection base configs (only what UBC inherits)
│   │   └── ubc/                        # UBC task configs (4 files)
│   ├── checkpoints/                    # .gitignored — download from Mega link (see Pretrained weights)
│   ├── mmdet_plugins/
│   │   └── ubc.py                      # UBC dataset class (shared: training + runtime)
│   └── ubc_inference.py                # Public module API
├── training_datasets/                  # UBC v1 data
├── examples/
│   └── ubc_inference_from_gps.py       # Standalone CLI demo (runs the full flow)
└── README.md
```

The `ubc_classifier/` directory is the deliverable — drop it into any Python project (with a working MMDetection install) and you have building classification on tap.

---

# Training

Skip this section if you're using our [pretrained weights](#pretrained-weights) — training is only needed to reproduce our numbers from scratch or train on a different dataset.

Training reproduces Huang et al.'s Cascade Mask R-CNN setup on the public UBC v1 train/val split. Each task takes **~2.5–3.5 hours** on a single RTX 5080 at batch size 6 (16 GB VRAM); total across all three tasks is **~8–9 hours**. Smaller GPUs need smaller batches — see the [VRAM & batch size](#vram--batch-size-guidance) section below before launching.

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
# Re-pin setuptools — earlier installs may have downgraded it, which breaks PEP 660 editable install
pip install -U "setuptools>=64,<80"

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
<summary><b>Option B — Ampere or older (RTX 30/40-series, GTX 16-series — anything sm_75 to sm_89)</b></summary>

Used for our inference deployment on an RTX 3070 Ti Laptop (Ampere, sm_86, 8 GB VRAM) and successfully reproduced by users on GTX 1660 (Turing, sm_75). Any GPU with compute capability sm_75 to sm_89 works — these cards have prebuilt MMCV wheels available, which dramatically simplifies the install.

This setup also works for training at smaller batch size; see [VRAM & batch size](#vram--batch-size-guidance) below.

> **Low system RAM (≤16 GB)?** Append `--no-cache-dir` to every `pip install` command in this section. pip's wheel cache balloons memory during install of large packages (PyTorch, MMCV) and can OOM on 16 GB systems with no swap. The flag forces re-download on retry but keeps install memory bounded.

### 1. Conda env + anti-pollution setting

```bash
conda create -n ubc_mmdet python=3.10 -y
conda activate ubc_mmdet

# Disable user-site to prevent ~/.local/lib/ pollution from stale pip installs
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export PYTHONNOUSERSITE=1' > $CONDA_PREFIX/etc/conda/activate.d/no_user_site.sh
conda deactivate && conda activate ubc_mmdet
```

### 2. PyTorch 2.1.x + cu121

Pin torch to the 2.1.x line because MMCV 2.1.0's prebuilt wheels target torch 2.1's compiled API. Patch versions 2.1.0, 2.1.1, and 2.1.2 all work — pick whichever is currently easiest to fetch.

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
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
# Expect: 2.1.2+cu121 [..., 'sm_75', 'sm_80', 'sm_86', 'sm_89', ...]
```

### 3. MMEngine + prebuilt MMCV

```bash
pip install -U "setuptools>=64,<80"
pip install -U openmim
mim install mmengine

# Pin MMCV via direct pip install (more deterministic than `mim install mmcv` —
# `mim` can fall back to source builds on edge cases and take a long time before failing)
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

MMCV downloads a ~80 MB wheel — no compilation.

### 4. MMDetection + remaining deps

```bash
# Re-pin setuptools — earlier installs may have downgraded it, which breaks PEP 660 editable install
pip install -U "setuptools>=64,<80"

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

MMDetection needs to know about the UBC dataset classes. Drop the plugin file into MMDetection's datasets directory and register it in `__init__.py`, then copy configs into place. Run these commands from the repo root.

```bash
# UBC dataset class
cp ubc_classifier/mmdet_plugins/ubc.py ~/mmdetection/mmdet/datasets/

# Register in __init__.py (portable — uses the running user's home)
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
cp ubc_classifier/configs/ubc/*.py ~/mmdetection/configs/ubc/
```

## Training configuration

All task configs inherit from `cascade-mask-rcnn_r50_fpn_ubc_base.py`:

```python
MAX_EPOCHS = 100
VAL_INTERVAL = 5
LR_STEP_EPOCHS = [int(MAX_EPOCHS * 0.6), int(MAX_EPOCHS * 0.9)]   # 60, 90

data_root = '~/UBC_v1.0/'   # symlink to training_datasets/UBC_v1.0 (see Dataset section)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomResize', scale=[(400, 400), (900, 900)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(batch_size=6, ...)   # adjust to fit your GPU

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco.pth'
```

Task-specific configs override `num_classes` + dataset type + annotation filenames. No other changes needed.

## VRAM & batch size guidance

Cascade Mask R-CNN at 600×600 is memory-heavy. Use this table to size the batch before launching:

| GPU VRAM | Recommended batch | Scaled LR | Expected time/task |
|----------|-------------------|-----------|---------------------|
| 16 GB (RTX 5080, 4070 Ti Super) | 6 | 0.0075 | ~2.5–3.5 h |
| 12 GB (RTX 5070, 4070, 3080 Ti) | 4 | 0.005 | ~3.5–5 h |
| 8 GB (RTX 3070 Ti Laptop, 4060) | 2 | 0.0025 | ~6–9 h |
| 6 GB (GTX 1660 / 1660 Ti) | 1 | 0.00125 | ~10–14 h |

**LR scaling rule:** linear with batch size, using MMDetection's reference of `0.02 at batch 16` as the anchor. For batch B on 1 GPU: `lr = 0.02 * B / 16`.

**If you hit CUDA OOM during training**, drop the batch size in `ubc_classifier/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py`:

```python
train_dataloader = dict(batch_size=2, ...)   # was 6
```

And proportionally drop the LR:

```python
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, ...))   # was 0.0075
```

Then re-copy the config to `~/mmdetection/configs/ubc/` and restart.

**Don't fight VRAM headroom** — Cascade's memory usage varies by tile density, so leave ~1 GB buffer between your peak and the card's limit. A run that OOMs halfway through is a multi-hour loss; a conservative batch is ~20% longer but always finishes.

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

Before committing to a multi-hour run, verify the pipeline end-to-end:

```bash
sed -i 's/^MAX_EPOCHS = .*/MAX_EPOCHS = 1/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
sed -i 's/^VAL_INTERVAL = .*/VAL_INTERVAL = 1/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py

cd ~/mmdetection
rm -rf work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine
python tools/train.py configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_fine.py
```

Success criteria: completes without error, produces an 11-row per-class AP table, saves a best checkpoint, doesn't OOM. Overall mAP will be near zero — that's expected at 1 epoch, structural validation is the goal.

### 3. Full 100-epoch runs

Restore the full schedule:

```bash
sed -i 's/^MAX_EPOCHS = .*/MAX_EPOCHS = 100/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
sed -i 's/^VAL_INTERVAL = .*/VAL_INTERVAL = 5/' ~/mmdetection/configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_base.py
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

## Per-task training budgets

Measured on RTX 5080 (Blackwell, 16 GB VRAM, batch 6):

| Task | Time | Best epoch | Final Segm mAP |
|------|------|------------|----------------|
| roof_coarse | ~3 h | 75 | 0.140 |
| roof_fine | ~3 h | 75 | 0.088 |
| use_coarse | ~2.5 h | 50 | 0.101 |

Total: ~8–9 hours across all three tasks at batch 6. Scale up proportionally for smaller batch sizes — batch 2 on 8 GB VRAM roughly triples the wall-clock because the per-epoch iteration count triples.

## Training observations

- **Three-task gap pattern.** roof_fine → 62% of paper AP; use_coarse → 80%; roof_coarse → estimated 65-70% (paper doesn't report this task). The gap scales with how much the task benefits from large effective batch size.
- **use_coarse converges fastest.** Best checkpoint at epoch 50 vs epoch 75 for roof tasks. Consistent with the paper's observation that function classification saturates earlier — the latent visual signal for "residential vs commercial" is less granular than roof geometry.
- **Per-class rankings match the paper.** Dominant classes (flat, gable, hipped_v2 for roof; residential for use) learn well; rare classes (shed_roof, pinnacle_roof, industrial) consistently underperform in both our runs and the paper's.

## After training

Copy best checkpoints to `ubc_classifier/checkpoints/` with short names. Run from the repo root:

```bash
# Create the checkpoints directory (gitignored — doesn't exist on fresh clone)
mkdir -p ubc_classifier/checkpoints

cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse/best_coco_segm_mAP_epoch_*.pth \
   ubc_classifier/checkpoints/roof_coarse.pth
cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine/best_coco_segm_mAP_epoch_*.pth \
   ubc_classifier/checkpoints/roof_fine.pth
cp ~/mmdetection/work_dirs/cascade-mask-rcnn_r50_fpn_ubc_use_coarse/best_coco_segm_mAP_epoch_*.pth \
   ubc_classifier/checkpoints/use_coarse.pth
```

## Training troubleshooting

### `FileNotFoundError: [Errno 2] No such file or directory: '/home/<user>/UBC_v1.0/annotations/...'`

Dataset not found at the path the config expects. Either create the symlink (see [Dataset](#dataset)) or edit `data_root` in the base config.

### `torch.cuda.OutOfMemoryError: CUDA out of memory`

Your GPU can't fit the configured batch size. See [VRAM & batch size](#vram--batch-size-guidance) for recommended values per card. Drop batch + LR together.

### `Cannot use unregistered type UBCRoofFineDataset`

Plugin registration didn't take effect. Verify with `python -c "from mmdet.datasets import UBCRoofFineDataset"`. Re-run the registration snippet above if this fails.

### `ERROR: Project ... uses a build backend that is missing the 'build_editable' hook`

Setuptools too old (<64). Re-pin before retrying the editable install:

```bash
pip install -U "setuptools>=64,<80"
pip install -v -e . --no-build-isolation
```

### `mim install mmcv` hangs or starts compiling from source

`mim`'s wheel resolution can fall through to source build on edge cases. Bypass with direct pip install:

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
```

### `pip install` is OOM-killed during install

Common on systems with ≤16 GB RAM and no swap. PyTorch and MMCV wheels balloon pip's cache during install. Add `--no-cache-dir` to the failing install:

```bash
pip install <package> --no-cache-dir
```

### `_pickle.UnpicklingError: Weights only load failed`

PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which rejects MMEngine's checkpoint format. Patch MMEngine's loader once per env:

```bash
python3 << 'EOF'
import mmengine.runner.checkpoint as ckpt_mod
import inspect
src = inspect.getsourcefile(ckpt_mod)
with open(src, 'r') as f:
    content = f.read()
old = "checkpoint = torch.load(filename, map_location=map_location)"
new = "checkpoint = torch.load(filename, map_location=map_location, weights_only=False)"
if old in content and new not in content:
    content = content.replace(old, new)
    with open(src, 'w') as f:
        f.write(content)
    print("Patched")
elif new in content:
    print("Already patched")
EOF
```

Safe because the checkpoints are ones you (or the repo maintainers) trained.

### OpenCV TIFF warnings: `Unknown field with tag 33550/33922/34735/34737`

Harmless. UBC tiles are GeoTIFFs carrying geospatial metadata OpenCV's reader doesn't recognize. Silence with `export OPENCV_LOG_LEVEL=ERROR`.

### Loss NaN or exploding early in training

Most commonly LR too high relative to batch size. Confirm you've scaled LR with batch — see [VRAM & batch size](#vram--batch-size-guidance). If LR is already scaled appropriately, try halving it as a first diagnostic.

---

# Inference

The `ubc_classifier/` directory is an importable Python module. Three public functions in `ubc_inference.py` form the API: fetch satellite imagery from Esri given a GPS coordinate, run a trained model on it, and visualize the results. The fire detection pipeline on the `main` branch wraps these with a non-blocking thread.

## Module API

```python
import ubc_classifier.ubc_inference as ubc

# 1. Fetch satellite imagery for a GPS coordinate
img_bgr = ubc.fetch_esri_image(
    lat=48.1374, lon=11.5755,    # Munich Altstadt
    buffer_meters=150.0,          # 300m × 300m ground area
    img_size=600,                 # 600×600 px (matches UBC training GSD)
)

# 2. Run a trained Cascade Mask R-CNN model
result = ubc.run_ubc_inference(
    img_bgr,
    task="use_coarse",            # or "roof_coarse" / "roof_fine"
    score_threshold=0.3,
)
# result.pred_instances has bboxes, scores, labels, masks

# 3. Render predictions over the input image
annotated = ubc.visualize_predictions(img_bgr, result, task="use_coarse")
# annotated is a BGR ndarray ready for cv2.imshow / cv2.imwrite
```

The `get_model(task)` function lazily loads + caches the requested checkpoint on first call (~3 s per task). Subsequent calls are free. Call it explicitly at startup if you want to absorb load latency before the first user-visible query:

```python
ubc.get_model("use_coarse")    # warmup
```

All three task models fit comfortably in 8 GB VRAM if loaded simultaneously. On 6 GB cards you'll need the [low-VRAM lazy-load pattern](#low-vram-cards-6-gb).

## Standalone CLI demo

`examples/ubc_inference_from_gps.py` runs the full pipeline as a one-shot command — useful for testing the install, generating report figures, or sanity-checking new locations:

```bash
# Munich city center, function classification
python examples/ubc_inference_from_gps.py \
    --lat 48.1374 --lon 11.5755 \
    --task use_coarse

# Roof type at the same coordinate
python examples/ubc_inference_from_gps.py \
    --lat 48.1374 --lon 11.5755 \
    --task roof_coarse

# Save without display, custom output dir (good for batch figures)
python examples/ubc_inference_from_gps.py \
    --lat 42.2626 --lon -71.8023 \
    --task use_coarse \
    --out-dir figures/worcester
```

Output: two PNG files per query — `<lat>_<lon>_<task>_raw.png` (the satellite image) and `<lat>_<lon>_<task>_annotated.png` (with predictions overlaid).

## Performance

Measured on RTX 5080 (Blackwell, 16 GB VRAM):

| Stage | Latency |
|-------|---------|
| First per-task `get_model()` call | ~3 s (loads checkpoint) |
| Cached-model inference | ~50 ms |
| Esri imagery fetch (600×600 PNG) | ~1–1.5 s |
| Total end-to-end (cached) | ~1.5–2 s (network-dominated) |

Esri's free tier is the bottleneck; switching to a paid Esri API key would mostly remove that latency.

## Good demo coordinates

| Location | Lat, Lon | Shows |
|----------|----------|-------|
| Munich Altstadt | `48.1374, 11.5755` | Dense historic core, mixed gable/hipped, public dominant |
| Munich suburb | `48.1100, 11.5900` | Apartment blocks, flat roofs, residential dominant |
| Beijing Chaoyang | `39.9200, 116.4500` | Modern residential high-rise |
| Beijing Hutongs | `39.9250, 116.3800` | Traditional courtyard architecture |
| Wang Fuk Court (Tai Po, Hong Kong) | `22.4478, 114.1755` | Tower-on-podium typology — illustrates OOD failure modes (towers misread as `public`, podiums as `commercial`) |
| Worcester, MA | `42.2626, -71.8023` | North American suburban OOD test |

Running all three tasks at one location produces a useful comparison figure.

## Low-VRAM cards (≤6 GB)

The integrated pipeline on the `main` branch loads three models simultaneously (FFireNet + YOLO + one UBC task), totaling ~4–5 GB peak VRAM. On a 6 GB card (e.g., GTX 1660), this leaves no headroom and OOMs during inference bursts.

Mitigation: switch the UBC handler to lazy-load + purge mode. In `ubc_handler.py` on the `main` branch, replace `warmup()` with a no-op:

```python
def warmup(self):
    """Lazy-load mode for low-VRAM cards: model loads on first 'U' press."""
    print(f"[UBC] Ready (lazy-load mode). Press 'U' to classify buildings at "
          f"({self.cfg.ubc_demo_lat}, {self.cfg.ubc_demo_lon})")
```

And modify `_run_query()` to load the model at the top, then evict after inference:

```python
def _run_query(self):
    """Thread target for low-VRAM mode: load → fetch → infer → visualize → purge."""
    try:
        t0 = time.perf_counter()

        # Lazy load — only push the model into VRAM now, not at startup
        print(f"[UBC] Loading '{self.cfg.ubc_task}' into VRAM...")
        ubc.get_model(self.cfg.ubc_task)

        img = ubc.fetch_esri_image(
            lat=self.cfg.ubc_demo_lat, lon=self.cfg.ubc_demo_lon,
            buffer_meters=self.cfg.ubc_buffer_meters,
            img_size=self.cfg.ubc_img_size,
        )
        if img is None:
            raise RuntimeError("Esri fetch failed (network error or rate limit)")

        result = ubc.run_ubc_inference(
            img, self.cfg.ubc_task,
            score_threshold=self.cfg.ubc_score_threshold,
        )
        annotated = ubc.visualize_predictions(img, result, self.cfg.ubc_task)

        # Evict the UBC model so the fire-detection loop can reclaim its VRAM
        ubc._MODEL_CACHE.clear()    # drop Python references in ubc_inference's cache
        import torch
        torch.cuda.empty_cache()

        n = len(result.pred_instances)
        ms = int((time.perf_counter() - t0) * 1000)
        print(f"[UBC] {n} buildings detected in {ms} ms (model purged from VRAM)")

        with self._lock:
            self._latest_result = annotated
            self._needs_repaint = True
    except Exception as e:
        msg = f"UBC query failed: {e}"
        print(f"[UBC] ERROR: {msg}")
        with self._lock:
            self._last_error = msg
            self._needs_repaint = True
    finally:
        self._pending = False
        import torch
        torch.cuda.empty_cache()
```

The `ubc._MODEL_CACHE.clear()` call is what actually frees the model — `torch.cuda.empty_cache()` alone only releases unused allocator blocks, not live model tensors.

Trade-off: every `U` press now incurs the ~3 s model load cost. Acceptable for demos with infrequent queries; less so for sustained workloads.

## Inference troubleshooting

### `ImportError: cannot import name 'ubc_inference'`

Make sure your importing code adds `ubc_classifier/` to `sys.path` before `import ubc_inference`, or import via the full package path `from ubc_classifier import ubc_inference`. The fire detection pipeline's `ubc_handler.py` shows the sys.path approach.

### `Esri returned HTTP 502` or empty response

Esri's free tier rate-limits intermittently. Retry after 30 seconds. For higher throughput, register an Esri developer account and supply an API key in `fetch_esri_image()`.

### `_pickle.UnpicklingError: Weights only load failed`

See the same entry in the [Training troubleshooting](#training-troubleshooting) section above — same patch applies.

### `torch.cuda.OutOfMemoryError` on inference (not training)

The integrated pipeline on `main` runs three models concurrently. On 6 GB cards this leaves no headroom. See [Low-VRAM cards](#low-vram-cards-6-gb) for the lazy-load + purge pattern.

---

# Integration with the Fire Detection Pipeline

When the pipeline on the `main` branch fires a fire alert, it imports `ubc_classifier.ubc_inference` and runs building classification on satellite imagery near the drone's GPS coordinates. Both the fire pipeline (YOLO + FFireNet) and the classifier run in a single Python process on the same host, sharing the GPU.

In our demo flow, the pipeline binds the `'U'` key to a manual classifier trigger so the operator chooses when to query — useful for live demos, where unsolicited classifications would be visually noisy. The handler runs the query in a background thread (Esri fetch is ~1–1.5 s of network I/O), and a result window pops up when the response arrives.

For a production drone deployment, the same module functions would be called automatically when `alert_triggered` transitions to True in the conviction tracker, with the drone's actual GPS feed providing `(lat, lon)` instead of the demo coordinate.

## Single-machine rationale

Earlier iterations of this branch shipped with a FastAPI server (`server.py`) and an HTTP client, supporting a two-machine architecture. We removed that in favor of in-process module use because:

1. **Same machine, same GPU.** The fire pipeline and classifier run on the same host in the demo. HTTP between two processes on `localhost` adds serialization overhead (base64 PNG encoding, JSON transport, FastAPI request handling) for no architectural gain.
2. **One environment to maintain.** With `ubc_mmdet` already supporting both MMDetection and Ultralytics, we don't need a separate server env or a separate client env.
3. **Cleaner failure modes.** A missing checkpoint surfaces as an `ImportError` at startup, not a 500 response 30 seconds into a demo.

The "production drone deployment" in the proposal continues to suggest the classifier as a cloud service, which is sensible at that scale — but for a single-host demo, modules win on simplicity.

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

Mappings declared in [`ubc_classifier/mmdet_plugins/ubc.py`](ubc_classifier/mmdet_plugins/ubc.py) via `METAINFO` and must match the training annotation files.

# References

- Cai, Z., & Vasconcelos, N. (2019). Cascade R-CNN: High quality object detection and instance segmentation. *IEEE TPAMI, 43*(5), 1483–1498.
- Chen, K., Wang, J., Pang, J., et al. (2019). MMDetection: Open MMLab detection toolbox and benchmark. *arXiv:1906.07155*.
- Huang, X., Ren, L., Liu, C., et al. (2022). Urban Building Classification (UBC): A dataset for individual building detection and classification. *CVPRW 2022*, 1412–1420.

# License

Research and educational purposes.
