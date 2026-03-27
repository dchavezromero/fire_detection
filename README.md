# FFireNet + YOLO Fire Detection Pipeline

Two-stage fire detection system that combines **FFireNet** (lightweight binary classifier) with **YOLOv8** (object detector) for reliable fire detection in drone footage.

FFireNet runs on every frame as a fast gate. YOLO runs conditionally for localization and cross-validation. Temporal smoothing over a rolling window eliminates flickering and single-frame false positives.

## How It Works

```
Frame → FFireNet (fire/no-fire score)
              │
              ├─ High confidence fire ──→ YOLO (normal conf) ──→ Fuse scores ──→ Rolling window ──→ Alert?
              ├─ Uncertain ─────────────→ YOLO (normal conf) ──→ Fuse scores ──→ Rolling window ──→ Alert?
              └─ High confidence no-fire → Skip YOLO (spot-check every 30 frames)
```

### Cross-Correlation

When both models run on a frame, their outputs are fused with a weighted average (default 60% YOLO / 40% FFireNet). A disagreement penalty pulls the fused score toward uncertainty when the models strongly disagree, preventing one model from overriding the other on ambiguous frames.

### Temporal Smoothing

Per-frame fused scores feed into a rolling window (default 8 frames). A fire alert triggers only when the average score exceeds the threshold **and** a minimum number of individual frames agree. This suppresses sporadic false positives from either model.

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- ultralytics (YOLOv8)
- torchvision
- NumPy

```bash
pip install torch torchvision opencv-python ultralytics numpy
```

## Project Structure

```
├── fire_pipeline.py                  # Main pipeline script
├── ffirenet_test.py                  # Standalone FFireNet inference
├── models/
│   ├── mobilenet_v2_.../ffirenet.pth # FFireNet weights
│   └── 26m_.../weights/best.pt       # YOLOv8 weights
└── sample_videos/
    └── *.mp4                         # Test videos
```

## Usage

Edit the paths and thresholds in `PipelineConfig` at the bottom of the script, then run:

```bash
python fire_pipeline.py
```

Press `q` to quit. A summary prints after processing showing total frames, YOLO utilization, and alert count.

To save the output video:

```python
cfg = PipelineConfig(
    save_output=True,
    output_path="output_pipeline.mp4",
    ...
)
```

## Configuration

All thresholds live in the `PipelineConfig` dataclass. Here's what each one does:

### FFireNet Gating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ffirenet_fire_thresh` | `0.3` | FFireNet sigmoid below this = confident fire → always run YOLO |
| `ffirenet_nofire_thresh` | `0.7` | FFireNet sigmoid above this = confident no fire → skip YOLO |

The gap between these two values defines the **uncertain zone** where YOLO always runs. Widening the zone (e.g. `0.2` / `0.8`) runs YOLO more often for higher accuracy at the cost of speed. Narrowing it saves compute but trusts FFireNet more.

### YOLO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_conf` | `0.3` | Standard YOLO confidence threshold |
| `yolo_recheck_conf` | `0.15` | Lower threshold used during spot-checks and disagreement rechecks |
| `yolo_device` | `0` | GPU device index, or `"cpu"` |

### Score Fusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_ffirenet` | `0.4` | Weight given to FFireNet in the fused score |
| `w_yolo` | `0.6` | Weight given to YOLO in the fused score |

If one model consistently outperforms the other on your footage, shift the weight toward the better one. The two weights don't need to sum to 1 but it's cleaner if they do.

### Temporal Smoothing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | `8` | Number of frames in the rolling window |
| `alert_threshold` | `0.55` | Smoothed score above this triggers an alert |
| `min_frames_for_alert` | `4` | Minimum frames in the window that must individually exceed the threshold |

Increasing `window_size` and `min_frames_for_alert` makes the system more conservative (fewer false alarms, slower reaction). Decreasing them makes it more responsive but noisier.

## Tuning Guide

**Start here** — run a video you know contains fire and watch the score panel in the bottom-left:

1. If fire frames are getting skipped by FFireNet (YOLO never runs), **lower `ffirenet_nofire_thresh`** (e.g. `0.6`)
2. If YOLO runs on almost every frame, **raise `ffirenet_nofire_thresh`** or lower `ffirenet_fire_thresh` to widen the skip zone
3. If you get brief false alerts, **increase `min_frames_for_alert`** or `window_size`
4. If real fires take too long to trigger, **decrease `min_frames_for_alert`** or lower `alert_threshold`
5. Check the YOLO utilization % at the end — if it's >80%, the gating isn't saving much compute

## On-Screen Display

- **Red banner** at the top when a fire alert is active
- **YOLO bounding boxes** drawn in red with confidence scores
- **Score panel** (bottom-left) shows real-time values for FFireNet score, YOLO score, fused score, smoothed score, inference time, and YOLO call count

## Models

**FFireNet** — MobileNetV2 backbone fine-tuned for binary fire/no-fire classification. Lightweight enough to run on every frame. Sigmoid output where values near 0 indicate fire and values near 1 indicate no fire.

**YOLOv8** — Object detection model trained on fire datasets. Provides bounding box localization and per-detection confidence. Heavier than FFireNet but gives spatial information about where fire is in the frame.

## License

This project is for research and educational purposes.