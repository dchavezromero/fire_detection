# FFireNet: Fire Classification with MobileNetV2 Transfer Learning

PyTorch reimplementation of **FFireNet** (Khan & Khan, 2022) — a lightweight binary fire/no-fire classifier built on MobileNetV2. Part of a larger two-stage gated fire detection pipeline where FFireNet serves as a cheap every-frame gate that triggers YOLO26 only when fire is likely.

Based on: Khan & Khan (2022), *"FFireNet: Deep Learning Based Forest Fire Classification and Detection in Smart Cities"* (Symmetry, 14, 2155)

## Overview

FFireNet freezes a pretrained MobileNetV2 backbone and trains a small classification head for binary fire detection. The model outputs a sigmoid score where values near 0 indicate fire and values near 1 indicate no fire. With only 3.5M total parameters (660K trainable), it runs at ~345 FPS on GPU, making it suitable as an every-frame gate in compute-constrained UAV pipelines.

### Why PyTorch?

The original FFireNet was implemented in TensorFlow/Keras. We reimplemented in PyTorch because TensorFlow dropped native Windows GPU support after version 2.10, and the rest of our pipeline (YOLO26 via Ultralytics, UBC Cascade Mask R-CNN via MMDetection) is PyTorch-native — a single framework simplifies dependency management and GPU memory sharing.

### Results Summary

Five configurations were trained varying input resolution (224, 460, 640) and epoch count (50, 100, 200):

| Run | Img Size | Epochs | Accuracy | Precision | Recall | F1 | FDR | AUC | Latency |
|-----|----------|--------|----------|-----------|--------|----|-----|-----|---------|
| **MobileNetV2** | **640** | **100** | **97.38%** | **99.25%** | **95.78%** | **97.48%** | **0.75%** | **0.9989** | **2.89ms** |
| MobileNetV2 | 224 | 50 (run 2) | 96.84% | 95.88% | 97.89% | 96.88% | 4.12% | 0.9960 | 2.34ms |
| MobileNetV2 | 460 | 200 | 96.69% | 99.54% | 94.18% | 96.78% | 0.46% | 0.9989 | 2.64ms |
| MobileNetV2 | 224 | 50 | 95.47% | 99.22% | 92.29% | 95.63% | 0.78% | 0.9955 | 2.19ms |
| MobileNetV2 | 640 | 50 | 91.48% | 98.65% | 85.30% | 91.49% | 1.35% | 0.9905 | 3.05ms |
| *Khan & Khan baseline* | *224* | *50* | *98.42%* | *97.42%* | *99.47%* | *—* | *—* | *—* | *—* |

The best configuration (640px, 100 epochs) achieves **97.38% accuracy** with higher precision than the original paper (99.25% vs 97.42%), at the cost of slightly lower recall (95.78% vs 99.47%). This trade-off is favorable for the gate role: high precision means fewer false gate triggers that waste YOLO compute.

## Requirements

- Python 3.10
- NVIDIA GPU (any CUDA-capable; we trained on RTX 5080)
- PyTorch with CUDA support — version depends on GPU generation (see below)
- torchvision, NumPy <2.0, Pillow, tqdm
- matplotlib, seaborn, scikit-learn (for evaluation)

### Already have the `ubc_mmdet` env from the ubc_model branch?

Just add the FFireNet-specific deps on top — the env's PyTorch install already supports MobileNetV2:

```bash
conda activate ubc_mmdet
pip install matplotlib seaborn scikit-learn tqdm
pip install "numpy<2.0"   # confirm the pin is still in place
```

You can skip the rest of this section.

### Fresh install — GPU setup

The default `pip install torch` pulls a **CPU-only** build. Match the install to your GPU generation:

<details>
<summary><b>Blackwell (RTX 50-series, sm_120)</b></summary>

Blackwell needs CUDA 12.8+ wheels — older `cu124` wheels lack `sm_120` kernels and fail at runtime with "no kernel image is available."

```bash
conda create -n ffirenet python=3.10 -y
conda activate ffirenet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib seaborn scikit-learn tqdm pillow "numpy<2.0"
```

Verify `sm_120` is in the supported architecture list:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# Expect: 2.11.0+cu128 [..., 'sm_120']
```

</details>

<details>
<summary><b>Ampere or older (RTX 30/40-series, GTX 16-series — anything sm_75 to sm_89)</b></summary>

```bash
conda create -n ffirenet python=3.10 -y
conda activate ffirenet

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib seaborn scikit-learn tqdm pillow "numpy<2.0"
```

Low system RAM (≤16 GB)? Append `--no-cache-dir` to each pip command — pip's wheel cache balloons during install of large packages and can OOM on systems with no swap.

</details>

All experiments in the results table were run on a single NVIDIA RTX 5080 (Blackwell, 16 GB VRAM, batch 64).

## Project Structure

```
├── ffirenet_train.py         # Training script (data prep, model, training loop, evaluation)
├── ffirenet_metrics.py       # Evaluation metrics & plot generation (also runs standalone)
├── ffirenet_test.py          # Video inference with live overlay
├── training_datasets/
│   └── custom_ffirenet_data/
│       ├── fire/             # Fire images
│       ├── nofire/           # No-fire images
│       └── splits/           # Auto-generated train/val/test splits
├── models/                   # Auto-created: trained model outputs
│   ├── mobilenet_v2_640imgsz_100epochs_0.01lr/
│   │   ├── ffirenet.pth
│   │   ├── metrics.txt
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   ├── f1_confidence_curve.png
│   │   ├── precision_confidence_curve.png
│   │   └── recall_confidence_curve.png
│   └── ...
└── sample_videos/            # Test videos for inference
    └── *.mp4
```

## Dataset

The training dataset combines two sources for domain-relevant binary classification:

- **Fire images:** Extracted from the Fire and Smoke 5 dataset (Roboflow), the same dataset used for YOLO training
- **No-fire images:** Combined from the original FFireNet forest dataset and the Urban Aerial Computer Vision dataset, which provides overhead urban scenes representative of UAV viewpoints

Place fire and no-fire images in separate directories:

```
training_datasets/custom_ffirenet_data/
├── fire/       # All fire images
└── nofire/     # All no-fire images (forest + urban aerial)
```

The training script automatically creates an 80/10/10 train/val/test split on first run and reuses it on subsequent runs. Corrupt images are detected and removed before splitting.

## Training

Edit the configuration constants at the top of `ffirenet_train.py`, then run:

```bash
python ffirenet_train.py
```

### Configuration

Key parameters at the top of `ffirenet_train.py`:

```python
DATASET_DIR = "training_datasets/custom_ffirenet_data"
IMG_SIZE = 640          # Input resolution (224, 460, or 640)
EPOCHS = 100            # Training epochs
BATCH_SIZE = 64         # Batch size
LEARNING_RATE = 0.01    # SGD learning rate
TRAIN_SPLIT = 0.8       # 80/10/10 split
```

### What the script does

1. Validates all images and removes corrupt files
2. Creates train/val/test splits (80/10/10) if they don't already exist
3. Builds DataLoaders with augmentation (rotation, affine, horizontal flip) for training and clean transforms for evaluation
4. Trains the FFireNet model (frozen MobileNetV2 backbone + trainable classification head)
5. Saves the model checkpoint to `models/mobilenet_v2_<imgsz>imgsz_<epochs>epochs_<lr>lr/`
6. Runs full evaluation and generates all plots and metrics

### Architecture

```
MobileNetV2 (frozen, ImageNet pretrained)
    → AdaptiveAvgPool2d(1)
    → Flatten
    → Linear(1280, 512) + ReLU
    → Dropout(0.2)
    → Linear(512, 1)
    → BCEWithLogitsLoss (sigmoid applied in loss)
```

Only the classification head is trained (~660K parameters). The MobileNetV2 backbone (2.8M parameters) remains frozen with ImageNet weights.

### Data Augmentation

Following the original paper (Table 4):

| Augmentation | Value |
|-------------|-------|
| Random rotation | ±50° |
| Translation | ±20% |
| Shear | ±20% |
| Zoom | ±20% |
| Horizontal flip | 50% |

Augmentation is applied only to training data. Validation and test sets use resize + normalize only.

### Batch size guidance

FFireNet is small enough that batch size mostly affects training speed rather than fitting in VRAM. The defaults in `ffirenet_train.py` work on most cards:

| GPU VRAM | Recommended batch (640px) | Notes |
|----------|---------------------------|-------|
| 16 GB (RTX 5080, 4070 Ti Super) | 64 | Used for our results |
| 12 GB (RTX 5070, 4070, 3080 Ti) | 64 | Same — plenty of headroom |
| 8 GB (RTX 3070 Ti Laptop, 4060) | 32 | |
| 6 GB (GTX 1660 / 1660 Ti) | 16 | Drop further to 8 if OOM |

At 224 px the model fits much larger batches; bump to 128+ on cards with ≥12 GB VRAM if training time matters.

### Output

Results are saved to a named folder under `models/`:

```
models/mobilenet_v2_640imgsz_100epochs_0.01lr/
├── ffirenet.pth                    # Model checkpoint (weights + class mapping + training history)
├── metrics.txt                     # All evaluation metrics as text
├── training_history.png            # Loss and accuracy curves
├── confusion_matrix.png            # Raw count confusion matrix
├── confusion_matrix_normalized.png # Normalized confusion matrix
├── roc_curve.png                   # ROC curve with zoomed inset
├── pr_curve.png                    # Per-class Precision-Recall curve
├── f1_confidence_curve.png         # F1 vs confidence threshold
├── precision_confidence_curve.png  # Precision vs confidence threshold
└── recall_confidence_curve.png     # Recall vs confidence threshold
```

The folder naming convention (`<N>imgsz` pattern) is used by the main pipeline to auto-parse the preprocessing image size — no manual configuration needed when switching between models trained at different resolutions.

## Evaluation (Standalone)

To re-evaluate a saved model without retraining:

```bash
python ffirenet_metrics.py \
    --model models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth \
    --test-dir training_datasets/custom_ffirenet_data/splits/test \
    --output eval_results/
```

This generates all metrics and plots without requiring access to the training data or history.

### Metrics Reported

Following the paper's evaluation tables:

| Category | Metrics |
|----------|---------|
| Confusion matrix (Table 6) | TP, TN, FP, FN, accuracy, error rate |
| True/false rates (Table 7) | TNR, TPR, FPR, FNR, AUC-ROC |
| Precision & recall (Table 8) | Precision, recall, FDR, F1, per-class AP |
| Latency | ms/frame, FPS |

## Video Inference

To run FFireNet on a video file with a live overlay:

```bash
python ffirenet_test.py
```

Edit the paths at the top of the script:

```python
MODEL_PATH = "models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth"
VIDEO_PATH = "sample_videos/fire7.mp4"
```

The overlay shows the fire/no-fire prediction, confidence percentage, FPS, and frame count. Press `q` to quit.

**Note:** The standalone test script hardcodes 224×224 preprocessing. When using models trained at other resolutions (e.g., 640×640), update the `Resize` in the `preprocess` transform to match, or use the main pipeline (`main.py` on the main branch) which auto-parses the image size from the model folder name.

## Integration with the Fire Detection Pipeline

Trained FFireNet models plug directly into the gated pipeline on the `main` branch, where they serve as the every-frame gate. Copy the trained folder into the `main` branch's `models/` directory and update the path in `config.py`:

```python
cfg = PipelineConfig(
    ffirenet_model_path="models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth",
    ...
)
```

In the pipeline, FFireNet's sigmoid output is compared against `gate_thresh` (default 0.8). When it falls below that threshold (indicating possible fire), YOLO26 activates for spatial localization. FFireNet is used purely as a trigger — all scoring and confirmation is handled by YOLO and the conviction tracker.

## Troubleshooting

### `RuntimeError: CUDA error: no kernel image is available for execution on the device`

Your PyTorch install doesn't support your GPU's compute capability. Most commonly hit on RTX 50-series (Blackwell, sm_120) when installing the default cu121 or cu124 wheels — neither has `sm_120` kernels. Reinstall with the `cu128` index URL shown in the [Blackwell setup section](#fresh-install--gpu-setup).

### `pip install` is OOM-killed

Common on systems with ≤16 GB RAM and no swap. PyTorch wheels balloon pip's cache during install. Add `--no-cache-dir`:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
```

### Numpy version conflict warnings

If `pip check` reports `opencv-python requires numpy>=2`, that's cosmetic — OpenCV's declared requirement is loose, and 1.26.x works fine in practice. The `numpy<2.0` pin is needed downstream for MMDetection on the ubc_model branch and avoids surprises if you reuse this env across branches.

### `_pickle.UnpicklingError: Weights only load failed` when loading the checkpoint

PyTorch 2.6+ defaults `torch.load` to `weights_only=True`, which rejects checkpoints that contain non-tensor metadata (training history, class mappings). The repo's load functions explicitly pass `weights_only=False`. If you've copied loading code into a fresh script and hit this, add the flag:

```python
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
```

Safe because the checkpoints are ones you trained.

## License

Research and educational purposes.
