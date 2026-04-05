# YOLO Fire & Smoke Detection — Training & Inference

Training and inference scripts for YOLO-based fire and smoke detection on UAV imagery, built on the [Ultralytics](https://docs.ultralytics.com/) framework. Part of a larger two-stage gated fire detection pipeline that pairs YOLO with FFireNet for compute-efficient real-time detection.

## Overview

This branch contains everything needed to train YOLO models on the Fire and Smoke 5 dataset and run inference on video. Trained weights are automatically organized into a `models/` directory with metadata-based naming for easy version tracking.

### Results Summary

Seven configurations were trained spanning YOLOv8 and YOLO26, three model scales, two resolutions, and varying epoch counts. The top performers on the Fire and Smoke 5 validation set:

| Run | Img Size | Epochs | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|-----|----------|--------|--------|-----------|-----------|--------|----|
| **YOLO26m** | **640** | **200** | **0.881** | **0.674** | **0.894** | **0.824** | **0.857** |
| YOLO26l | 640 | 200 | 0.884 | 0.673 | 0.895 | 0.828 | 0.860 |
| YOLOv8m | 640 | 200 | 0.882 | 0.675 | 0.893 | 0.820 | 0.855 |
| YOLO26m | 1280 | 200 | 0.890 | 0.640 | 0.884 | 0.830 | 0.856 |
| *Ramos et al. baseline* | *640* | *—* | *0.856* | *0.661* | *—* | *—* | *—* |

**YOLO26m at 640px** is the selected model for the pipeline. It matches YOLOv8m in accuracy while being 20% smaller (20.4M vs 25.8M params), requiring 14% fewer FLOPs (67.9B vs 78.7B), and providing NMS-free end-to-end inference for simpler edge deployment.

## Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU training)
- Ultralytics

```bash
pip install torch torchvision ultralytics
```

A CUDA-capable GPU is strongly recommended. All experiments were run on a single NVIDIA RTX 5080.

## Dataset

The [Fire and Smoke 5](https://universe.roboflow.com/) dataset from Roboflow (~20,000 annotated images, 2 classes: fire and smoke with bounding box labels). Download it in YOLO format and place it under `training_datasets/`:

```
training_datasets/
└── Fire And Smoke 5.v1i.yolo26/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

The `data.yaml` file defines class names and dataset paths. If using a different YOLO version (e.g., YOLOv8), export the dataset in the corresponding format — Roboflow provides version-specific exports.

## Project Structure

```
├── YOLO_training.py          # Training script with auto-organized output
├── YOLO_detection.py         # Inference script for video
├── training_datasets/        # Dataset(s) in YOLO format
│   └── Fire And Smoke 5.v1i.yolo26/
│       └── data.yaml
├── models/                   # Auto-created: trained model outputs
│   ├── 26m_640imgsz_200epochs/
│   │   └── weights/
│   │       ├── best.pt
│   │       └── last.pt
│   ├── v8m_640imgsz_200epochs/
│   └── ...
└── sample_videos/            # Test videos for inference
    └── *.mp4
```

## Training

Edit the model and hyperparameters in `YOLO_training.py`, then run:

```bash
python YOLO_training.py
```

### What the script does

1. Loads a pretrained YOLO checkpoint (e.g., `yolo26m.pt` — downloaded automatically on first run)
2. Trains on the Fire and Smoke 5 dataset with the specified configuration
3. Runs a validation pass after training
4. Automatically moves results from `runs/detect/trainN/` into `models/` with a descriptive folder name parsed from `args.yaml` (e.g., `26m_640imgsz_200epochs`)
5. Merges the validation results into the same folder

### Configuration

Modify the training call in `YOLO_training.py` to change hyperparameters:

```python
model = YOLO("yolo26m.pt")  # Model variant: yolo26n, yolo26m, yolo26l, yolov8m, etc.

model.train(
    data="training_datasets/Fire And Smoke 5.v1i.yolo26/data.yaml",
    epochs=200,        # Training epochs
    patience=50,       # Early stopping patience
    imgsz=640,         # Input resolution (640 or 1280)
    batch=20,          # Batch size (adjust for GPU memory)
    device=0,          # GPU index, or "cpu"
)
```

Key observations from our experiments:

- **Resolution:** 640px outperformed 1280px on mAP@50:95 for this dataset (0.674 vs 0.640) — the source images are natively lower resolution, so upscaling added interpolation artifacts without improving detection
- **Scale:** YOLO26l (large) provided no measurable gain over YOLO26m (0.673 vs 0.674 mAP@50:95), confirming medium as the efficiency sweet spot
- **Epochs:** 200 epochs with patience 50 yielded the best results; the nano model at 50 epochs was significantly weaker (0.478 mAP@50:95)

### Output Structure

After training completes, the script creates a folder like:

```
models/26m_640imgsz_200epochs/
├── weights/
│   ├── best.pt            # Best validation checkpoint
│   └── last.pt            # Final epoch checkpoint
├── args.yaml              # Full training configuration
├── results.csv            # Per-epoch metrics
├── results.png            # Training curves
├── confusion_matrix.png
├── BoxPR_curve.png        # Precision-Recall curve
├── val/                   # Validation results (auto-merged)
└── ...
```

## Inference

Edit the model path and video source in `YOLO_detection.py`, then run:

```bash
python YOLO_detection.py
```

```python
model = YOLO("models/26m_640imgsz_200epochs/weights/best.pt")

results = model.predict(
    source="sample_videos/fire7.mp4",  # Video file, image, directory, or camera index
    conf=0.3,                           # Confidence threshold
    show=True,                          # Display live window
    save=False,                         # Save annotated output
    device=0,                           # GPU index
    stream=True,                        # Stream mode for video (memory efficient)
)
```

Press `q` in the display window to quit.

The `source` parameter accepts video files, image files, directories of images, RTSP streams, or a webcam index (e.g., `source=0`).

## Integration with the Fire Detection Pipeline

These trained models plug directly into the two-stage gated pipeline on the `main` branch. The pipeline uses FFireNet as a lightweight binary gate (~3ms per frame) that activates YOLO only when fire is likely, reducing average compute load for real-time UAV deployment. YOLO's detections feed a conviction tracker with streak-aware gains, flicker penalties, and hysteresis to produce stable fire alerts.

To use a trained model in the pipeline, update the path in `config.py`:

```python
cfg = PipelineConfig(
    yolo_model_path="models/26m_640imgsz_200epochs/weights/best.pt",
    ...
)
```

The model folder naming convention (`<N>imgsz` pattern) is also used by the pipeline's FFireNet loader to auto-parse input resolution.

## Class Mapping

The dataset defines two classes:

| Class ID | Name |
|----------|------|
| 0 | smoke |
| 1 | fire |

This mapping is configured in the pipeline via `PipelineConfig.class_names`.

## License

Research and educational purposes.