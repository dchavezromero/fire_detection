"""
FFireNet Metrics & Evaluation (PyTorch)
========================================
All evaluation metrics (Tables 6-8 from the paper) and plots.
Called by ffirenet_train.py after training completes.

Can also be run standalone to re-evaluate a saved model:
    python ffirenet_metrics.py --model ./ffirenet_results/run_xxx/ffirenet.pth \
                               --test-dir ./ffirenet_results/splits/test \
                               --output ./ffirenet_results/run_xxx
"""

import os
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


# ============================================================
# Model Definition (needed for standalone loading)
# ============================================================
class FFireNet(nn.Module):
    def __init__(self):
        super().__init__()
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
# Core Evaluation
# ============================================================
def compute_metrics(model, test_loader, class_to_idx, device):
    """
    Run inference on the test set and compute all paper metrics.

    Returns a dict with all values + raw arrays for plotting.
    """
    class_names = list(class_to_idx.keys())
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Inference latency ---
    print("\n[INFO] Measuring inference latency...")
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(100):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    # --- Confusion matrix (paper Table 6) ---
    # ImageFolder: alphabetical -> fire=0, nofire=1
    # Paper convention: fire is the positive class
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]  # fire predicted as fire
    FN = cm[0][1]  # fire predicted as nofire
    FP = cm[1][0]  # nofire predicted as fire
    TN = cm[1][1]  # nofire predicted as nofire

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total
    error_rate = (FP + FN) / total

    # True/False rates (paper Table 7)
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = 1 - TNR
    FNR = 1 - TPR

    # Precision, Recall, FDR, F1 (paper Table 8)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TPR
    FDR = 1 - precision
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # ROC / AUC (paper Section 5.2.3)
    # fire=0 is positive, so fire_prob = 1 - y_prob
    fire_prob = 1 - y_prob
    fire_true = 1 - y_true
    fpr_curve, tpr_curve, _ = roc_curve(fire_true, fire_prob)
    roc_auc = auc(fpr_curve, tpr_curve)

    # PR curve (paper Section 5.2.4)
    pr_precision, pr_recall, _ = precision_recall_curve(fire_true, fire_prob)
    avg_precision = average_precision_score(fire_true, fire_prob)

    # sklearn report
    report = classification_report(y_true, y_pred, target_names=class_names)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy, "error_rate": error_rate,
        "TNR": TNR, "TPR": TPR, "FPR": FPR, "FNR": FNR,
        "precision": precision, "recall": recall,
        "FDR": FDR, "f1": f1,
        "roc_auc": roc_auc, "avg_precision": avg_precision,
        "fpr_curve": fpr_curve, "tpr_curve": tpr_curve,
        "pr_precision": pr_precision, "pr_recall": pr_recall,
        "avg_latency": avg_latency, "std_latency": std_latency,
        "cm": cm, "class_names": class_names, "report": report,
        "test_samples": len(y_true),
        "class_to_idx": class_to_idx,
    }


def print_metrics(m):
    """Print all metrics matching the paper's table format."""
    print(f"\n{'='*60}")
    print(f"  FFireNet Test Results ({m['test_samples']} images)")
    print(f"{'='*60}")

    print(f"\n  Confusion Matrix Values (Paper Table 6):")
    print(f"    TP (fire correct):     {m['TP']}")
    print(f"    TN (nofire correct):   {m['TN']}")
    print(f"    FP (nofire as fire):   {m['FP']}")
    print(f"    FN (fire as nofire):   {m['FN']}")
    print(f"    Prediction Accuracy:   {m['accuracy']*100:.2f}%")
    print(f"    Error Rate:            {m['error_rate']*100:.2f}%")

    print(f"\n  True/False Rates (Paper Table 7):")
    print(f"    TNR:  {m['TNR']*100:.2f}%")
    print(f"    TPR:  {m['TPR']*100:.2f}%")
    print(f"    FPR:  {m['FPR']*100:.2f}%")
    print(f"    FNR:  {m['FNR']*100:.2f}%")
    print(f"    AUC:  {m['roc_auc']:.4f}")

    print(f"\n  Precision & Recall (Paper Table 8):")
    print(f"    Precision:  {m['precision']*100:.2f}%")
    print(f"    Recall:     {m['recall']*100:.2f}%")
    print(f"    FDR:        {m['FDR']*100:.2f}%")
    print(f"    F1 Score:   {m['f1']*100:.2f}%")
    print(f"    AP:         {m['avg_precision']*100:.2f}%")

    print(f"\n  Inference Latency:")
    print(f"    {m['avg_latency']:.2f} +/- {m['std_latency']:.2f} ms/frame")
    print(f"    ({1000/m['avg_latency']:.1f} FPS)")

    print(f"\n  Classification Report:")
    print(m["report"])
    print(f"{'='*60}")


def save_metrics(m, output_dir):
    """Write all metrics to a text file."""
    path = os.path.join(output_dir, "metrics.txt")
    with open(path, "w") as f:
        f.write("FFireNet Evaluation Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test samples: {m['test_samples']}\n")
        f.write(f"Class mapping: {m['class_to_idx']}\n\n")

        f.write(f"TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}\n")
        f.write(f"Accuracy:   {m['accuracy']*100:.2f}%\n")
        f.write(f"Error Rate: {m['error_rate']*100:.2f}%\n\n")

        f.write(f"TNR: {m['TNR']*100:.2f}%\n")
        f.write(f"TPR: {m['TPR']*100:.2f}%\n")
        f.write(f"FPR: {m['FPR']*100:.2f}%\n")
        f.write(f"FNR: {m['FNR']*100:.2f}%\n")
        f.write(f"AUC: {m['roc_auc']:.4f}\n\n")

        f.write(f"Precision: {m['precision']*100:.2f}%\n")
        f.write(f"Recall:    {m['recall']*100:.2f}%\n")
        f.write(f"FDR:       {m['FDR']*100:.2f}%\n")
        f.write(f"F1 Score:  {m['f1']*100:.2f}%\n")
        f.write(f"AP:        {m['avg_precision']*100:.2f}%\n\n")

        f.write(f"Latency: {m['avg_latency']:.2f} +/- {m['std_latency']:.2f} ms/frame\n")
        f.write(f"FPS:     {1000/m['avg_latency']:.1f}\n\n")

        f.write("Classification Report:\n")
        f.write(m["report"])

    print(f"[INFO] Metrics saved to {path}")


# ============================================================
# Plots
# ============================================================
def plot_confusion_matrix(cm, class_names, output_dir):
    """Confusion matrix heatmap (paper Figure 6)."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix - FFireNet", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """ROC curve with zoomed inset (paper Figure 7)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("Receiver Operating Characteristic Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[1].set_xlim([0, 0.2])
    axes[1].set_ylim([0.9, 1.0])
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve (Zoomed)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_pr_curve(pr_recall, pr_precision, avg_precision, output_dir):
    """Precision-Recall curve with zoomed inset (paper Figure 8)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(pr_recall, pr_precision, color="blue", lw=2,
                 label=f"PR curve (AP = {avg_precision:.4f})")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pr_recall, pr_precision, color="blue", lw=2,
                 label=f"PR curve (AP = {avg_precision:.4f})")
    axes[1].set_xlim([0.9, 1.0])
    axes[1].set_ylim([0.9, 1.0])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curve (Zoomed)")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_training_history(history, output_dir):
    """Training/validation loss and accuracy (paper Figure 5)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Training Loss", color="red")
    axes[0].plot(history["val_loss"], label="Validation Loss", color="blue")
    axes[0].set_title("Performance of the Proposed Method")
    axes[0].set_xlabel("Num of Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Training Accuracy", color="red")
    axes[1].plot(history["val_acc"], label="Validation Accuracy", color="blue")
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Num of Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


# ============================================================
# Entry Point
# ============================================================
def run_evaluation(model, history, test_loader, class_to_idx, device, output_dir):
    """
    Run full evaluation pipeline. Called from ffirenet_train.py after training.
    """
    print(f"\n[INFO] Running evaluation...")

    m = compute_metrics(model, test_loader, class_to_idx, device)
    print_metrics(m)
    save_metrics(m, output_dir)

    plot_training_history(history, output_dir)
    plot_confusion_matrix(m["cm"], m["class_names"], output_dir)
    plot_roc_curve(m["fpr_curve"], m["tpr_curve"], m["roc_auc"], output_dir)
    plot_pr_curve(m["pr_recall"], m["pr_precision"], m["avg_precision"], output_dir)

    return m


# ============================================================
# Standalone usage: re-evaluate a saved model
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Re-evaluate a saved FFireNet model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved .pth model")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to test/ directory with fire/ and nofire/ subfolders")
    parser.add_argument("--output", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"[INFO] Loading model from {args.model}")
    model = FFireNet().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    class_to_idx = checkpoint["class_to_idx"]

    # Build test loader
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    test_dataset = datasets.ImageFolder(args.test_dir, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Evaluate (no history available in standalone mode)
    m = compute_metrics(model, test_loader, class_to_idx, device)
    print_metrics(m)
    save_metrics(m, args.output)
    plot_confusion_matrix(m["cm"], m["class_names"], args.output)
    plot_roc_curve(m["fpr_curve"], m["tpr_curve"], m["roc_auc"], args.output)
    plot_pr_curve(m["pr_recall"], m["pr_precision"], m["avg_precision"], args.output)

    print(f"\n[INFO] Results saved to {args.output}")