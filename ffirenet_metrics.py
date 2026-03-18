"""
FFireNet Metrics & Evaluation (PyTorch)
========================================
All evaluation metrics (Tables 6-8 from the paper) and plots.
Includes YOLO-style confidence-threshold curves for visual consistency.

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
# Confidence-Threshold Sweep (YOLO-style)
# ============================================================
def compute_confidence_curves(y_true, y_prob, class_names):
    """
    Sweep confidence thresholds from 0 to 1 and compute per-class
    and all-class Precision, Recall, and F1 at each threshold.

    Returns dict with arrays for plotting.
    """
    thresholds = np.linspace(0, 1, 1000)

    # For binary: fire=0, nofire=1
    # fire_prob = probability the image is fire
    fire_prob = 1 - y_prob
    fire_true = (y_true == 0).astype(int)
    nofire_true = (y_true == 1).astype(int)

    results = {
        "thresholds": thresholds,
        "fire": {"precision": [], "recall": [], "f1": []},
        "nofire": {"precision": [], "recall": [], "f1": []},
        "all": {"precision": [], "recall": [], "f1": []},
    }

    for t in thresholds:
        # Fire class: predict fire if fire_prob >= t
        fire_pred = (fire_prob >= t).astype(int)
        tp_f = np.sum((fire_pred == 1) & (fire_true == 1))
        fp_f = np.sum((fire_pred == 1) & (fire_true == 0))
        fn_f = np.sum((fire_pred == 0) & (fire_true == 1))

        p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        f1_f = 2 * p_f * r_f / (p_f + r_f) if (p_f + r_f) > 0 else 0

        # Nofire class: predict nofire if y_prob >= t
        nofire_pred = (y_prob >= t).astype(int)
        tp_n = np.sum((nofire_pred == 1) & (nofire_true == 1))
        fp_n = np.sum((nofire_pred == 1) & (nofire_true == 0))
        fn_n = np.sum((nofire_pred == 0) & (nofire_true == 1))

        p_n = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0
        r_n = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0
        f1_n = 2 * p_n * r_n / (p_n + r_n) if (p_n + r_n) > 0 else 0

        results["fire"]["precision"].append(p_f)
        results["fire"]["recall"].append(r_f)
        results["fire"]["f1"].append(f1_f)
        results["nofire"]["precision"].append(p_n)
        results["nofire"]["recall"].append(r_n)
        results["nofire"]["f1"].append(f1_n)

        # All classes (macro average)
        results["all"]["precision"].append((p_f + p_n) / 2)
        results["all"]["recall"].append((r_f + r_n) / 2)
        results["all"]["f1"].append((f1_f + f1_n) / 2)

    # Convert to arrays
    for cls in ["fire", "nofire", "all"]:
        for metric in ["precision", "recall", "f1"]:
            results[cls][metric] = np.array(results[cls][metric])

    return results


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
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]

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

    # ROC / AUC
    fire_prob = 1 - y_prob
    fire_true = 1 - y_true
    fpr_curve, tpr_curve, _ = roc_curve(fire_true, fire_prob)
    roc_auc = auc(fpr_curve, tpr_curve)

    # PR curve
    pr_precision, pr_recall, _ = precision_recall_curve(fire_true, fire_prob)
    avg_precision = average_precision_score(fire_true, fire_prob)

    # Per-class PR curves
    # nofire as positive
    nofire_true = y_true  # nofire=1
    pr_prec_nofire, pr_rec_nofire, _ = precision_recall_curve(nofire_true, y_prob)
    ap_nofire = average_precision_score(nofire_true, y_prob)

    # Confidence-threshold curves
    conf_curves = compute_confidence_curves(y_true, y_prob, class_names)

    # sklearn report
    report = classification_report(y_true, y_pred, target_names=class_names)

    return {
        # counts
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        # rates
        "accuracy": accuracy, "error_rate": error_rate,
        "TNR": TNR, "TPR": TPR, "FPR": FPR, "FNR": FNR,
        # precision / recall
        "precision": precision, "recall": recall,
        "FDR": FDR, "f1": f1,
        # ROC
        "roc_auc": roc_auc, "avg_precision": avg_precision,
        "fpr_curve": fpr_curve, "tpr_curve": tpr_curve,
        # PR curves (fire as positive)
        "pr_precision": pr_precision, "pr_recall": pr_recall,
        # PR curves (nofire as positive)
        "pr_prec_nofire": pr_prec_nofire, "pr_rec_nofire": pr_rec_nofire,
        "ap_fire": avg_precision, "ap_nofire": ap_nofire,
        # confidence curves
        "conf_curves": conf_curves,
        # latency
        "avg_latency": avg_latency, "std_latency": std_latency,
        # extras
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
    print(f"    AP (fire):  {m['ap_fire']*100:.2f}%")
    print(f"    AP (nofire):{m['ap_nofire']*100:.2f}%")

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

        f.write(f"Precision:  {m['precision']*100:.2f}%\n")
        f.write(f"Recall:     {m['recall']*100:.2f}%\n")
        f.write(f"FDR:        {m['FDR']*100:.2f}%\n")
        f.write(f"F1 Score:   {m['f1']*100:.2f}%\n")
        f.write(f"AP (fire):  {m['ap_fire']*100:.2f}%\n")
        f.write(f"AP (nofire):{m['ap_nofire']*100:.2f}%\n\n")

        f.write(f"Latency: {m['avg_latency']:.2f} +/- {m['std_latency']:.2f} ms/frame\n")
        f.write(f"FPS:     {1000/m['avg_latency']:.1f}\n\n")

        f.write("Classification Report:\n")
        f.write(m["report"])

    print(f"[INFO] Metrics saved to {path}")


# ============================================================
# Plots — Paper Style
# ============================================================
def plot_confusion_matrix(cm, class_names, output_dir):
    """Confusion matrix heatmap with raw counts (paper Figure 6)."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_confusion_matrix_normalized(cm, class_names, output_dir):
    """Normalized confusion matrix (YOLO-style)."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 16},
    )
    plt.title("Confusion Matrix Normalized", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix_normalized.png")
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


def plot_pr_curve(m, output_dir):
    """Per-class Precision-Recall curve (YOLO-style with per-class AP in legend)."""
    plt.figure(figsize=(10, 8))

    plt.plot(m["pr_recall"], m["pr_precision"], color="skyblue", lw=1.5,
             label=f"fire {m['ap_fire']:.3f}")
    plt.plot(m["pr_rec_nofire"], m["pr_prec_nofire"], color="orange", lw=1.5,
             label=f"nofire {m['ap_nofire']:.3f}")

    # All-class average
    mean_ap = (m["ap_fire"] + m["ap_nofire"]) / 2
    # Plot average as thick line (use fire curve as proxy shape)
    plt.plot(m["pr_recall"], m["pr_precision"], color="navy", lw=3, alpha=0.5,
             label=f"all classes {mean_ap:.3f} mAP@0.5")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
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
# Plots — YOLO-Style Confidence Curves
# ============================================================
def _plot_confidence_curve(conf_curves, metric_name, ylabel, output_dir):
    """
    Generic confidence-threshold curve plotter (YOLO-style).
    Shows per-class lines + bold all-class line with best value annotated.
    """
    thresholds = conf_curves["thresholds"]

    plt.figure(figsize=(10, 8))

    # Per-class
    plt.plot(thresholds, conf_curves["fire"][metric_name],
             color="skyblue", lw=1.5, label="fire")
    plt.plot(thresholds, conf_curves["nofire"][metric_name],
             color="orange", lw=1.5, label="nofire")

    # All-class (macro avg)
    all_vals = conf_curves["all"][metric_name]
    best_idx = np.argmax(all_vals)
    best_val = all_vals[best_idx]
    best_thresh = thresholds[best_idx]

    plt.plot(thresholds, all_vals, color="navy", lw=3,
             label=f"all classes {best_val:.2f} at {best_thresh:.3f}")

    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{ylabel}-Confidence Curve", fontsize=14)
    plt.legend(loc="best", fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{metric_name}_confidence_curve.png"
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Saved: {path}")


def plot_f1_confidence(conf_curves, output_dir):
    _plot_confidence_curve(conf_curves, "f1", "F1", output_dir)


def plot_precision_confidence(conf_curves, output_dir):
    _plot_confidence_curve(conf_curves, "precision", "Precision", output_dir)


def plot_recall_confidence(conf_curves, output_dir):
    _plot_confidence_curve(conf_curves, "recall", "Recall", output_dir)


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

    # Paper-style plots
    plot_training_history(history, output_dir)
    plot_confusion_matrix(m["cm"], m["class_names"], output_dir)
    plot_confusion_matrix_normalized(m["cm"], m["class_names"], output_dir)
    plot_roc_curve(m["fpr_curve"], m["tpr_curve"], m["roc_auc"], output_dir)
    plot_pr_curve(m, output_dir)

    # YOLO-style confidence curves
    plot_f1_confidence(m["conf_curves"], output_dir)
    plot_precision_confidence(m["conf_curves"], output_dir)
    plot_recall_confidence(m["conf_curves"], output_dir)

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
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
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
    plot_confusion_matrix_normalized(m["cm"], m["class_names"], args.output)
    plot_roc_curve(m["fpr_curve"], m["tpr_curve"], m["roc_auc"], args.output)
    plot_pr_curve(m, args.output)
    plot_f1_confidence(m["conf_curves"], args.output)
    plot_precision_confidence(m["conf_curves"], args.output)
    plot_recall_confidence(m["conf_curves"], args.output)

    print(f"\n[INFO] Results saved to {args.output}")