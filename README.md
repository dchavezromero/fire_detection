# FFireNet + YOLO26 Fire Detection Pipeline

Two-stage fire detection system that combines **FFireNet** (lightweight binary classifier) with **YOLO26 Medium** (object detector) for reliable fire detection in drone footage.

FFireNet runs on every frame as a cheap trip wire. The moment it detects possible fire, it opens a gate that hands control to YOLO26 for a configurable number of frames. YOLO's detections drive a conviction tracker that builds confidence over sustained fire and penalizes flickering, producing the final fire alert.

## How It Works

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
                  │              Fire found            No fire found
                  │              Reset countdown       Countdown ticks down
                  │                    │                     │
                  │                    ▼                     ▼
                  │              Conviction rises     Conviction decays
                  │                                         │
                  │                                   Countdown hits 0?
                  │                                   Close gate
                  │
                  └─ sigmoid >= gate_thresh → Gate stays closed
                                              Conviction decays passively
```

### Gate Mechanism

FFireNet is not used for scoring — it's purely a trigger. Its sigmoid output is compared against a single threshold (`gate_thresh`). Below it, the gate opens and YOLO activates for `gate_frames` frames. If YOLO keeps finding fire, the countdown resets each time, keeping the gate open indefinitely. Once YOLO stops finding fire, the countdown ticks down and the gate closes.

### Conviction Tracking

While the gate is open, YOLO's detections drive the conviction tracker:

- **YOLO finding fire raises conviction.** The rise accelerates with streak length — sustained detections build conviction faster than isolated ones.
- **YOLO finding nothing decays conviction.**
- **Flickering is penalized.** Each detection↔no-detection transition applies a flat penalty, so rapid flickering actively erodes conviction.
- **Hysteresis prevents alert flickering.** The alert triggers at `alert_conviction` (default `0.60`) but doesn't clear until conviction drops to `clear_conviction` (default `0.20`).

When the gate is closed, conviction simply decays — no YOLO runs, no compute wasted.

### Auto Image Size

The FFireNet preprocessing image size is automatically parsed from the model folder name (e.g. `mobilenet_v2_640imgsz_100epochs_0.01lr` → `640x640`). No manual configuration needed when switching between models trained at different resolutions.

## Requirements

- Python 3.10+
- PyTorch
- OpenCV
- ultralytics (YOLO26)
- torchvision
- NumPy

```bash
pip install torch torchvision opencv-python ultralytics numpy
```

## Project Structure

```
fire_detection/
├── main.py                  # Entry point — FFireNet model, gate logic, pipeline, video loop
├── config.py                # PipelineConfig dataclass (all tunable thresholds)
├── fusion.py                # ConvictionTracker
├── display.py               # On-screen overlay drawing
├── README.md
├── models/
│   ├── mobilenet_v2_.../    # FFireNet weights (imgsz parsed from folder name)
│   │   └── ffirenet.pth
│   ├── 26m_.../             # YOLO26m weights
│   │   └── weights/best.pt
│   └── ...                  # Other trained model variants
└── sample_videos/
    └── *.mp4
```

### Module Breakdown

| File | Responsibility |
|------|---------------|
| `config.py` | All paths and tunable thresholds in a single dataclass |
| `fusion.py` | `ConvictionTracker` — streak-aware state machine with hysteresis, driven by YOLO output |
| `display.py` | OpenCV overlay rendering (bounding boxes, alert banner, gate status, score panel) |
| `main.py` | FFireNet model definition, image size parsing, gate logic, YOLO orchestration, video loop |

## Usage

Edit the paths and thresholds in `config.py` (or override them at the bottom of `main.py`), then run:

```bash
python main.py
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

All thresholds live in `config.py` inside the `PipelineConfig` dataclass.

### FFireNet Gate

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gate_thresh` | `0.5` | FFireNet sigmoid below this trips the gate and activates YOLO |
| `gate_frames` | `30` | Number of frames YOLO stays active after the gate trips |

Lower `gate_thresh` makes FFireNet more conservative (needs higher fire confidence to trigger YOLO). Higher values make it more sensitive. `gate_frames` controls how long YOLO keeps running after each trigger — if YOLO finds fire during that window, the countdown resets so the gate stays open.

### YOLO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_conf` | `0.3` | YOLO confidence threshold for detections |
| `yolo_device` | `0` | GPU device index, or `"cpu"` |

### Conviction Tracker

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conviction_rise` | `0.15` | Conviction added per YOLO fire frame (scales with streak length and score) |
| `conviction_decay` | `0.08` | Conviction removed per no-fire frame |
| `flicker_penalty` | `0.10` | Conviction subtracted on each fire↔no-fire transition |
| `alert_conviction` | `0.60` | Conviction level that triggers a fire alert |
| `clear_conviction` | `0.20` | Conviction level below which an active alert is cleared |

The gap between `alert_conviction` and `clear_conviction` is the **hysteresis band**. A wider gap means alerts are stickier (harder to trigger, harder to clear). A narrower gap makes the system more responsive but more prone to alert flickering.

## Tuning Guide

Run a video you know contains fire and watch the score panel in the bottom-left:

1. If the gate never opens (YOLO never runs), **raise `gate_thresh`** — FFireNet needs a more lenient trigger
2. If the gate opens too often on non-fire scenes, **lower `gate_thresh`**
3. If YOLO finds fire but the gate closes too soon during pauses, **increase `gate_frames`**
4. If real fires take too long to trigger an alert, **lower `alert_conviction`** or **increase `conviction_rise`**
5. If you get false alerts from brief YOLO detections, **increase `flicker_penalty`** or **raise `alert_conviction`**
6. If alerts linger too long after fire disappears, **raise `clear_conviction`** or **increase `conviction_decay`**
7. Check the YOLO utilization % at the end — this tells you how much compute the gating is saving

## On-Screen Display

- **Red banner** at the top when a fire alert is active
- **YOLO bounding boxes** drawn in red with confidence scores
- **Score panel** (bottom-left) shows FFireNet sigmoid, gate state with remaining countdown, YOLO score, conviction level with streak count, alert state, inference time, and YOLO call count

## Models

**FFireNet** — MobileNetV2 backbone fine-tuned for binary fire/no-fire classification. Runs on every frame as a lightweight gate. Sigmoid output where values near 0 indicate fire and values near 1 indicate no fire. Image size is automatically inferred from the model folder name (`<N>imgsz` pattern).

**YOLO26 Medium** — Ultralytics' latest object detection model with NMS-free end-to-end inference, trained on fire datasets. Only runs when FFireNet trips the gate. Provides bounding box localization and per-detection confidence.

## License

This project is for research and educational purposes.