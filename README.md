# Fire Detection Pipeline

Three-stage fire detection system for UAV footage combining **FFireNet** (binary gate), **YOLO26** (object detector), and **UBC Cascade Mask R-CNN** (building classifier). FFireNet + YOLO identify fire in real time; pressing `U` queries the UBC module for buildings at the drone's GPS location, giving responders immediate context about what's in the affected area.

All three models run in a single Python process on one GPU-equipped Ubuntu host.

## How It Works

### Fire detection — two-stage gate

```
Every frame → FFireNet (fire/no-fire sigmoid)
                  │
                  ├─ sigmoid < gate_thresh → Open gate, start countdown
                  │                              │
                  │                              ▼
                  │                          YOLO runs every frame
                  │                              │
                  │                    ┌─────────┴──────────┐
                  │                    │                     │
                  │              Fire/smoke found       Nothing found
                  │              Reset countdown        Countdown ticks down
                  │                    │                     │
                  │                    ▼                     ▼
                  │              Conviction rises      Conviction decays
                  │              (fire weighted         (aggressive: 0.30)
                  │               heavier than smoke)        │
                  │                                    Countdown hits 0?
                  │                                    Close gate
                  │
                  └─ sigmoid >= gate_thresh → Gate stays closed
                                              Conviction decays passively
```

FFireNet runs on every frame as a cheap trip wire. The moment it detects possible fire, it opens a gate that hands control to YOLO26 for a configurable number of frames. YOLO's detections drive a conviction tracker that builds confidence over sustained fire and penalizes flickering, producing the final fire alert.

### Building classification — on-demand UBC query

Pressing `U` during playback triggers a separate pipeline: fetch a satellite image from Esri at the demo GPS, run a trained Cascade Mask R-CNN (one of three UBC tasks), and display the annotated result in its own window. This runs on a background thread so the fire detection loop stays responsive. Latency is ~1.5–2 s end-to-end, network-dominated.

The classifier module lives at `ubc_classifier/` in this branch. For training reproduction and the module's public API, see the [`ubc_model`](../../tree/ubc_model) branch.

### Gate mechanism

FFireNet is not used for scoring — it's purely a trigger. Its sigmoid output is compared against a single threshold (`gate_thresh`). Below it, the gate opens and YOLO activates for `gate_frames` frames. If YOLO keeps finding fire, the countdown resets each time, keeping the gate open indefinitely. Once YOLO stops finding fire, the countdown ticks down and the gate closes.

### Class-aware scoring

YOLO detections are split by class and scored separately. Fire detections are the primary signal, contributing peak confidence (×0.5) and a count density factor (×0.3). Smoke detections add a weaker boost (×0.2 of peak smoke confidence). This means smoke alone builds conviction slower than fire, reducing false alerts from non-fire smoke while still allowing smoke to corroborate fire detections.

### Conviction tracking

While the gate is open, YOLO's detections drive the conviction tracker:

- **YOLO finding fire raises conviction.** The rise accelerates with streak length — sustained detections build conviction faster than isolated ones.
- **YOLO finding nothing decays conviction.** Decay (0.30) is aggressive to ensure conviction drops quickly when fire disappears.
- **Flickering is penalized.** Each detection↔no-detection transition applies a flat penalty, so rapid flickering actively erodes conviction.
- **Hysteresis prevents alert flickering.** The alert triggers at `alert_conviction` (default `0.60`) but doesn't clear until conviction drops to `clear_conviction` (default `0.20`).

When the gate is closed, conviction simply decays — no YOLO runs, no compute wasted.

### Auto image size

The FFireNet preprocessing image size is automatically parsed from the model folder name (e.g. `mobilenet_v2_640imgsz_100epochs_0.01lr` → `640x640`). No manual configuration needed when switching between models trained at different resolutions.

## Setup

This pipeline needs a working MMDetection install alongside PyTorch + Ultralytics, because the UBC classifier is built on MMDetection. A single conda environment — `ubc_mmdet` — supports all three stages.

### 1. Clone the repo

```bash
git clone <repo-url> fire_detection
cd fire_detection
```

Everything needed structurally is in the clone: `ubc_classifier/` (the UBC inference module), empty `models/ubc_*_600imgsz_100epochs/` folders for checkpoints, sample videos, configs.

### 2. Build the `ubc_mmdet` conda environment

The environment build is the slowest part of setup (MMCV source build can take 15 minutes on Blackwell). Follow the appropriate section in the [ubc_model branch README](../../blob/ubc_model/README.md#gpu-environment-setup) — pick **Blackwell (RTX 50-series)** or **Ampere (RTX 30/40-series)** based on your GPU. That installs PyTorch + CUDA + MMEngine + MMCV + MMDetection.

### 3. Add fire-pipeline dependencies

With `ubc_mmdet` active, add Ultralytics and confirm the numpy pin:

```bash
conda activate ubc_mmdet
pip install ultralytics
pip install "numpy<2.0"   # Ultralytics deps may pull numpy 2.x; MMDetection needs <2.0

# Sanity check — warnings about opencv/openxlab numpy versions are cosmetic, safe to ignore
pip check
```

Verify both stacks work together:

```bash
python -c "
from ultralytics import YOLO
from mmcv.ops import box_iou_rotated
import torch
print('CUDA:', torch.cuda.get_device_name(0))
print('sm_120 support:', 'sm_120' in torch.cuda.get_arch_list())   # RTX 50-series only
b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5]]).cuda()
b2 = torch.tensor([[0.0, 2.0, 2.0, 5.0, 0.3]]).cuda()
print('MMCV op:', box_iou_rotated(b1, b2))
print('YOLO:', YOLO('yolo26m.pt').predict('https://ultralytics.com/images/bus.jpg', verbose=False)[0].boxes.shape)
"
```

### 4. Patch MMEngine's checkpoint loader

PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which rejects MMEngine's checkpoint format. One-time patch per env:

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

Safe because the checkpoints are ones the team trained.

### 5. Download the model weights

Three groups of weights go under `models/`. The folders already exist in the clone — just drop the files in.

**FFireNet + YOLO26 weights** — train on their respective branches, or copy from a team member:

| Folder | What goes there |
|--------|-----------------|
| `models/mobilenet_v2_640imgsz_100epochs_0.01lr/` | `ffirenet.pth` |
| `models/26m_640imgsz_200epochs/weights/` | `best.pt` |

**UBC weights** — download the pre-packaged checkpoints from [Mega](https://mega.nz/folder/sBZDTYAD#8vHE33KrOt_M20QyJQnl3Q):

```bash
# After downloading roof_coarse.pth, roof_fine.pth, use_coarse.pth
mv ~/Downloads/roof_coarse.pth models/ubc_roof_coarse_600imgsz_100epochs/
mv ~/Downloads/roof_fine.pth   models/ubc_roof_fine_600imgsz_100epochs/
mv ~/Downloads/use_coarse.pth  models/ubc_use_coarse_600imgsz_100epochs/

# Each folder should now have roof_coarse.pth / roof_fine.pth / use_coarse.pth (~310 MB each)
ls -lh models/ubc_*_600imgsz_100epochs/*.pth
```

### 6. Run

```bash
python main.py
```

If everything is in place, you'll see the FFireNet and YOLO loaders fire, then the UBC model warmup, then the video window opens and playback starts.

## Project Structure

```
fire_detection/
├── main.py                  # Entry point — FFireNet, YOLO, conviction tracker, video loop
├── config.py                # PipelineConfig dataclass (thresholds + UBC demo coordinates)
├── fusion.py                # ConvictionTracker (state machine with hysteresis)
├── display.py               # OpenCV overlay drawing
├── ubc_handler.py           # 'U' keypress workflow — background thread + OpenCV window
├── ubc_classifier/          # Importable UBC module (tracked here; see ubc_model branch for training)
│   ├── configs/_base_/      # Minimal MMDetection base configs
│   ├── configs/ubc/         # UBC task configs (4 files)
│   ├── mmdet_plugins/ubc.py # Dataset class registration
│   └── ubc_inference.py     # Public module API: fetch / infer / visualize
├── models/
│   ├── mobilenet_v2_*/                       # FFireNet weights (imgsz parsed from folder name)
│   │   └── ffirenet.pth
│   ├── 26m_640imgsz_200epochs/               # YOLO26 weights
│   │   └── weights/best.pt
│   └── ubc_{task}_600imgsz_100epochs/        # UBC Cascade Mask R-CNN weights
│       └── {task}.pth
├── sample_videos/*.mp4
└── README.md
```

### Module breakdown

| File | Responsibility |
|------|---------------|
| `config.py` | All paths, thresholds, and UBC demo coordinates in a single dataclass |
| `fusion.py` | `ConvictionTracker` — streak-aware state machine with hysteresis, driven by YOLO output |
| `display.py` | OpenCV overlay rendering (bounding boxes, alert banner, gate status, score panel) |
| `main.py` | FFireNet model definition, gate logic, YOLO orchestration, video loop, keybindings |
| `ubc_handler.py` | `UBCQueryHandler` — triggers UBC lookups on `U`, runs on background thread |
| `ubc_classifier/` | Cascade Mask R-CNN classifier module; see `ubc_model` branch for training details |

## Usage

Edit paths and thresholds in `config.py` (or override them at the bottom of `main.py`), then run:

```bash
python main.py
```

### Keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `U` or `u` | Trigger UBC building classification at `ubc_demo_lat`, `ubc_demo_lon` |

A summary prints after processing showing total frames, YOLO utilization, and alert count.

### UBC on-demand lookup

Pressing `U` during playback opens a second window titled "UBC Building Classification" with an annotated satellite tile at whatever coordinate is set in the config. The query runs in a background thread — the fire detection video keeps playing smoothly during the ~1.5 s Esri fetch + inference. Subsequent `U` presses replace the image with a new classification.

The first `U` press has no warmup cost because `UBCQueryHandler.warmup()` eagerly loads the UBC model at pipeline startup.

### Saving the output video

```python
cfg = PipelineConfig(
    save_output=True,
    output_path="output_pipeline.mp4",
    ...
)
```

## Configuration

All thresholds and UBC demo settings live in `config.py` inside the `PipelineConfig` dataclass.

### FFireNet gate

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_thresh` | `0.8` | FFireNet sigmoid below this trips the gate and activates YOLO |
| `gate_frames` | `60` | Number of frames YOLO stays active after the gate trips |

Lower `gate_thresh` makes FFireNet more conservative (needs higher fire confidence to trigger YOLO). Higher values make it more sensitive. `gate_frames` controls how long YOLO keeps running after each trigger — if YOLO finds fire during that window, the countdown resets so the gate stays open.

### YOLO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_conf` | `0.3` | YOLO confidence threshold for detections |
| `yolo_device` | `0` | GPU device index, or `"cpu"` |

### Conviction tracker

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conviction_rise` | `0.15` | Conviction added per YOLO fire frame (scales with streak length and score) |
| `conviction_decay` | `0.30` | Conviction removed per no-fire frame |
| `flicker_penalty` | `0.10` | Conviction subtracted on each fire↔no-fire transition |
| `alert_conviction` | `0.60` | Conviction level that triggers a fire alert |
| `clear_conviction` | `0.20` | Conviction level below which an active alert is cleared |

The gap between `alert_conviction` and `clear_conviction` is the **hysteresis band**. A wider gap means alerts are stickier (harder to trigger, harder to clear). A narrower gap makes the system more responsive but more prone to alert flickering. The decay rate (0.30) is intentionally higher than the rise rate (0.15) so conviction drops quickly when fire disappears, while the longer gate window (60 frames) ensures brief occlusions don't prematurely close the gate.

### UBC classifier

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ubc_demo_lat` | `48.1374` | Latitude queried when `U` is pressed |
| `ubc_demo_lon` | `11.5755` | Longitude queried when `U` is pressed |
| `ubc_task` | `"use_coarse"` | One of `roof_coarse`, `roof_fine`, `use_coarse` |
| `ubc_buffer_meters` | `150.0` | Half-side of ground area in meters (150 → 300×300 m tile) |
| `ubc_img_size` | `600` | Output image side in pixels (600 matches UBC training GSD) |
| `ubc_score_threshold` | `0.3` | Filter out UBC predictions below this confidence |

**Good demo coordinates:**

| Location | Lat, Lon | Shows |
|----------|----------|-------|
| Munich Altstadt | `48.1374, 11.5755` | Dense historic core, mixed gable/hipped, public dominant |
| Beijing Chaoyang | `39.9200, 116.4500` | Modern residential high-rise |
| Beijing Hutongs | `39.9250, 116.3800` | Traditional courtyard architecture |
| Hong Kong — Sha Tin | `22.4465, 114.1763` | Ultra-dense residential high-rise |
| Worcester, MA | `42.2626, -71.8023` | Out-of-training-distribution test |

## Tuning guide

Run a video you know contains fire and watch the score panel in the bottom-left:

1. If the gate never opens (YOLO never runs), **raise `gate_thresh`** — FFireNet needs a more lenient trigger
2. If the gate opens too often on non-fire scenes, **lower `gate_thresh`**
3. If YOLO finds fire but the gate closes too soon during pauses, **increase `gate_frames`**
4. If real fires take too long to trigger an alert, **lower `alert_conviction`** or **increase `conviction_rise`**
5. If you get false alerts from brief YOLO detections, **increase `flicker_penalty`** or **raise `alert_conviction`**
6. If alerts linger too long after fire disappears, **raise `clear_conviction`** or **increase `conviction_decay`**
7. Check the YOLO utilization % at the end — this tells you how much compute the gating is saving

## On-screen display

- **Red banner** at the top when a fire alert is active
- **Class-colored YOLO bounding boxes** — red for fire, teal for smoke — with class name and confidence score labels
- **Score panel** (bottom-left) shows FFireNet sigmoid, gate state with remaining countdown, YOLO score, per-frame fire and smoke detection counts, conviction level with streak count, alert state, inference time, and YOLO call count
- **UBC window** (appears on first `U` press) shows the annotated satellite tile with building masks, class labels, and confidence scores

## Models

**FFireNet** — MobileNetV2 backbone fine-tuned for binary fire/no-fire classification. Runs on every frame as a lightweight gate. Sigmoid output where values near 0 indicate fire and values near 1 indicate no fire. Image size is automatically inferred from the model folder name (`<N>imgsz` pattern). See the `ffirenet` branch for training.

**YOLO26 Medium** — Ultralytics' latest object detection model with NMS-free end-to-end inference, trained on fire datasets. Only runs when FFireNet trips the gate. Provides bounding box localization and per-detection confidence. See the `yolo` branch for training.

**UBC Cascade Mask R-CNN** — Three MMDetection models (one per task) trained on the UBC v1 dataset for 100 epochs. Triggered manually via `U` keypress. See the `ubc_model` branch for training setup, results, and module API.

## Troubleshooting

### `FileNotFoundError: Checkpoint not found: ...ubc_*_600imgsz_100epochs/*.pth`

UBC weights are missing. Download from [Mega](https://mega.nz/folder/sBZDTYAD#8vHE33KrOt_M20QyJQnl3Q) and drop into the folders as described in [Setup — step 5](#5-download-the-model-weights).

### `FileNotFoundError: ...ffirenet.pth` or `...best.pt`

FFireNet or YOLO weights are missing. Train them on the `ffirenet` / `yolo` branches or copy from a team member.

### `_pickle.UnpicklingError: Weights only load failed`

PyTorch 2.6+ safety default. Apply the MMEngine patch in [Setup — step 4](#4-patch-mmengines-checkpoint-loader).

### `KeyError: ... is already registered in dataset at mmdet.datasets.ubc`

The UBC dataset classes are being registered twice (once from the plugin, once from a leftover training install at `~/mmdetection/mmdet/datasets/ubc.py`). The plugin decorators in `ubc_classifier/mmdet_plugins/ubc.py` use `@DATASETS.register_module(force=True)` to avoid this — if you've manually modified them, restore the `force=True` argument.

### `torch.cuda.OutOfMemoryError` on the UBC lookup

All three models (FFireNet + YOLO + one UBC task) fit comfortably in ~5–6 GB VRAM. OOM here usually means another CUDA process is holding memory — run `nvidia-smi` to check. If running on an 8 GB card, only the currently-active UBC task model stays resident.

### UBC window shows "Esri fetch failed"

Esri's free tile service rate-limits intermittently. Retry after 30 seconds. For sustained use, register a paid Esri API key.

### `RuntimeError: CUDA error: no kernel image is available for execution on the device`

PyTorch doesn't have kernels for your GPU's compute capability. For RTX 50-series (Blackwell, sm_120) this means you installed a PyTorch wheel built against an older CUDA. See the [ubc_model branch README](../../blob/ubc_model/README.md#gpu-environment-setup) — use the Blackwell install path with `cu128` wheels.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Integrated real-time pipeline — FFireNet + YOLO + UBC in one process |
| `ffirenet` | FFireNet training code and documentation |
| `yolo` | YOLO26 training code and documentation |
| `ubc_model` | UBC Cascade Mask R-CNN training + module, dataset, training reproduction docs |

## License

Research and educational purposes.