"""
FFireNet: Forest Fire Classification using MobileNetV2 Transfer Learning (PyTorch)
====================================================================================
Based on: Khan & Khan (2022) - "FFireNet: Deep Learning Based Forest Fire
Classification and Detection in Smart Cities" (Symmetry, 14, 2155)

Replication for CS 534 Project - WPI
Authors: Ankith Adkoli, Dennis Chavez Romero, Alexander Kobsa

Paper methodology:
  - Backbone: MobileNetV2 (ImageNet pretrained, frozen)
  - Head: GlobalAveragePooling2D -> Dense(512, ReLU) -> Dropout(0.2) -> Dense(1, sigmoid)
  - Optimizer: SGD (lr=0.01)
  - Loss: Binary cross-entropy
  - Epochs: 50, Batch size: 64
  - Data augmentation: rotation, scaling, shear, translation
  - Input size: 224x224
"""

import os
import math
import shutil
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from ffirenet_metrics import run_evaluation


# ============================================================
# Configuration
# ============================================================

# --- Paths ---
DATASET_DIR = "training_datasets/custom_ffirenet_data"
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NOFIRE_DIR = os.path.join(DATASET_DIR, "nofire")
OUTPUT_DIR = "./models"

# --- Paper hyperparameters (Table 5) ---
IMG_SIZE = 224
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SEED = 42
NUM_WORKERS = 4  # dataloader workers, set to 0 if you get multiprocessing errors

# --- Data split (proposal: 80/10/10) ---
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# --- Augmentation (Table 4) ---
ROTATION_RANGE = 50
SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2


# ============================================================
# Dataset Preparation
# ============================================================
def validate_and_clean_images(directory):
    """Remove corrupt/unreadable images."""
    removed = 0
    for root, _, files in os.walk(directory):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
            except Exception:
                print(f"  Removing corrupt file: {fpath}")
                os.remove(fpath)
                removed += 1
    return removed


def create_splits(fire_dir, nofire_dir, output_dir):
    """
    Build train/val/test folders from separate fire and nofire directories.

    Output:
        output_dir/splits/
            train/fire/, train/nofire/
            val/fire/,   val/nofire/
            test/fire/,  test/nofire/
    """
    split_dir = os.path.join(output_dir, "splits")

    # skip if already created
    train_check = os.path.join(split_dir, "train")
    if os.path.exists(train_check) and len(os.listdir(train_check)) >= 2:
        print(f"[INFO] Splits already exist at {split_dir}, skipping.")
        return split_dir

    print("[INFO] Creating train/val/test splits...")
    np.random.seed(SEED)

    for split in ["train", "val", "test"]:
        for cls in ["fire", "nofire"]:
            os.makedirs(os.path.join(split_dir, split, cls), exist_ok=True)

    sources = {"fire": fire_dir, "nofire": nofire_dir}

    for cls, src_dir in sources.items():
        images = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ])
        np.random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        assignments = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_imgs in assignments.items():
            for img_name in split_imgs:
                src = os.path.join(src_dir, img_name)
                dst = os.path.join(split_dir, split_name, cls, img_name)
                shutil.copy2(src, dst)

        print(f"  {cls}: {len(assignments['train'])} train / "
              f"{len(assignments['val'])} val / {len(assignments['test'])} test")

    return split_dir


# ============================================================
# Data Loading
# ============================================================
def create_dataloaders(split_dir):
    """
    Create train/val/test DataLoaders.

    Training data gets augmentation per paper Table 4.
    ImageNet normalization is applied (required for pretrained MobileNetV2).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(ROTATION_RANGE),
        transforms.RandomAffine(
            degrees=0,
            translate=(SHIFT_RANGE, SHIFT_RANGE),
            shear=(-SHEAR_RANGE * 180, SHEAR_RANGE * 180),  # degrees
            scale=(1 - ZOOM_RANGE, 1 + ZOOM_RANGE),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(split_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(split_dir, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(split_dir, "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"\n[INFO] Class indices: {train_dataset.class_to_idx}")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")

    return train_loader, val_loader, test_loader, test_dataset


# ============================================================
# Model (Section 3.3)
# ============================================================
class FFireNet(nn.Module):
    """
    FFireNet architecture:
      1. MobileNetV2 convolutional base (frozen, ImageNet weights)
      2. Global Average Pooling (adaptive)
      3. Dense(512, relu)
      4. Dropout(0.2)
      5. Dense(1) — raw logit, sigmoid applied via BCEWithLogitsLoss
    """
    def __init__(self):
        super().__init__()

        # load pretrained MobileNetV2 and freeze it
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.features.requires_grad_(False)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# Training
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

        vram = torch.cuda.memory_allocated(0)/1024**3 if device.type == "cuda" else 0
        pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                         acc=f"{correct/total:.4f}",
                         vram=f"{vram:.1f}GB")

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Run validation. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train(model, train_loader, val_loader, device):
    """Full training loop for the configured number of epochs."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=LEARNING_RATE)

    print(f"\n{'='*60}")
    print(f"  Training FFireNet")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    print(f"  Optimizer: SGD  |  Loss: BCEWithLogitsLoss")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
    }

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return history


# ============================================================
# Main
# ============================================================
def main():
    # --- Setup ---
    folder_name = f"mobilenet_v2_{IMG_SIZE}imgsz_{EPOCHS}epochs_{LEARNING_RATE}lr"

    run_dir = os.path.join(OUTPUT_DIR, folder_name)

    # Handle duplicate names
    counter = 2
    base_run_dir = run_dir
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{counter}"
        counter += 1

    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to: {run_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"[INFO] VRAM reserved:  {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
    else:
        print("[WARNING] No GPU detected, training will be slow.")

    # Seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # --- Clean images ---
    print("\n[INFO] Validating images...")
    for d in [FIRE_DIR, NOFIRE_DIR]:
        removed = validate_and_clean_images(d)
        if removed:
            print(f"  Removed {removed} corrupt files from {d}")

    # --- Split dataset ---
    split_dir = create_splits(FIRE_DIR, NOFIRE_DIR, DATASET_DIR)

    # --- Data loaders ---
    train_loader, val_loader, test_loader, test_dataset = create_dataloaders(split_dir)

    # --- Build & train ---
    model = FFireNet().to(device)
    print(f"\n[INFO] Model parameters: "
          f"{sum(p.numel() for p in model.parameters()):,} total, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    history = train(model, train_loader, val_loader, device)

    # --- Save model ---
    model_path = os.path.join(run_dir, "ffirenet.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": test_dataset.class_to_idx,
        "history": history,
    }, model_path)
    print(f"\n[INFO] Model saved to: {model_path}")

    # --- Evaluate & generate all plots ---
    run_evaluation(model, history, test_loader, test_dataset.class_to_idx, device, run_dir)

    print(f"\n[INFO] All results saved to: {run_dir}")


if __name__ == "__main__":
    main()