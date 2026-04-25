"""
ubc_inference.py — laptop-server variant of ubc_inference_from_gps.py

Adapted for the laptop deployment where:
  * MMDetection is pip-installed (no editable ~/mmdetection checkout)
  * UBC dataset class lives in ~/ubc_server/mmdet_plugins/ubc.py
  * Configs live in ~/ubc_server/configs/
  * Checkpoints live in ~/ubc_server/checkpoints/ with short names
    (roof_coarse.pth, roof_fine.pth, use_coarse.pth)

Three public functions mirror the original script for server reuse:
    fetch_esri_image(lat, lon, ...) -> BGR image
    run_ubc_inference(image_bgr, task, ...) -> DetDataSample
    visualize_predictions(image_bgr, result, task, ...) -> annotated BGR image

plus a model cache helper:
    get_model(task) -> cached MMDetection model, lazily loaded once per task

Import this module once from the server entrypoint; the side-effect of import
registers the UBC dataset classes in MMDetection's registry.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the server works from any location
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
MODELS_ROOT = PACKAGE_ROOT.parent / "models"
PLUGIN_ROOT = PACKAGE_ROOT / "mmdet_plugins"


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------
TASK_REGISTRY = {
    "roof_coarse": {
        "config":     CONFIG_ROOT / "ubc" / "cascade-mask-rcnn_r50_fpn_ubc_roof_coarse.py",
        "checkpoint": MODELS_ROOT / "ubc_roof_coarse_600imgsz_100epochs" / "roof_coarse.pth",
        "classes":    ("flat", "gable", "hipped", "arched", "other"),
        "colors":     [(239, 68, 68), (59, 130, 246), (168, 85, 247),
                       (251, 191, 36), (156, 163, 175)],
    },
    "roof_fine": {
        "config":     CONFIG_ROOT / "ubc" / "cascade-mask-rcnn_r50_fpn_ubc_roof_fine.py",
        "checkpoint": MODELS_ROOT / "ubc_roof_fine_600imgsz_100epochs" / "roof_fine.pth",
        "classes":    ("flat", "flat_roof_complex", "shed_roof", "gable_roof",
                       "gambrel_roof", "hipped_roof_v1", "hipped_roof_v2",
                       "mansard_roof", "pinnacle_roof", "arched", "other"),
        "colors":     [(239, 68, 68), (248, 113, 113), (217, 70, 239),
                       (59, 130, 246), (37, 99, 235), (168, 85, 247),
                       (147, 51, 234), (251, 191, 36), (245, 158, 11),
                       (34, 197, 94), (156, 163, 175)],
    },
    "use_coarse": {
        "config":     CONFIG_ROOT / "ubc" / "cascade-mask-rcnn_r50_fpn_ubc_use_coarse.py",
        "checkpoint": MODELS_ROOT / "ubc_use_coarse_600imgsz_100epochs" / "use_coarse.pth",
        "classes":    ("residential", "commercial", "industrial", "public", "other"),
        "colors":     [(34, 197, 94), (59, 130, 246), (168, 85, 247),
                       (251, 191, 36), (156, 163, 175)],
    },
}


# ---------------------------------------------------------------------------
# Register the UBC dataset class with MMDetection at import time
# ---------------------------------------------------------------------------
def _register_ubc_plugin():
    """Load mmdet_plugins/ubc.py and register its dataset classes with mmdet."""
    plugin_path = PLUGIN_ROOT / "ubc.py"
    if not plugin_path.exists():
        raise FileNotFoundError(
            f"UBC plugin not found at {plugin_path}. "
            f"Ensure the file exists before starting the server."
        )

    spec = importlib.util.spec_from_file_location("ubc_plugin", plugin_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ubc_plugin"] = module
    spec.loader.exec_module(module)
    # ubc.py uses @DATASETS.register_module() decorators — import is the registration


_register_ubc_plugin()


# ---------------------------------------------------------------------------
# Model cache — lazy-load each task's model on first use
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}


def get_model(task: str, device: str = "cuda:0"):
    """Return the MMDetection model for a given task, loading + caching on first call."""
    if task not in TASK_REGISTRY:
        raise KeyError(f"Unknown task {task!r}. Available: {list(TASK_REGISTRY)}")

    if task in _MODEL_CACHE:
        return _MODEL_CACHE[task]

    # Lazy import — keeps the module loadable without MMDetection if someone only
    # wants fetch_esri_image or visualize_predictions
    from mmdet.apis import init_detector

    reg = TASK_REGISTRY[task]
    print(f"[ubc_inference] Loading {task} model from {reg['checkpoint']}")
    if not reg["config"].exists():
        raise FileNotFoundError(f"Config not found: {reg['config']}")
    if not reg["checkpoint"].exists():
        raise FileNotFoundError(f"Checkpoint not found: {reg['checkpoint']}")

    model = init_detector(str(reg["config"]), str(reg["checkpoint"]), device=device)
    _MODEL_CACHE[task] = model
    print(f"[ubc_inference] {task} model ready")
    return model


def warmup_models(tasks: Optional[list] = None, device: str = "cuda:0"):
    """Eagerly load one or more task models (e.g., at server startup to avoid first-request latency)."""
    tasks = tasks or list(TASK_REGISTRY)
    for t in tasks:
        get_model(t, device=device)


def cached_tasks() -> list:
    return list(_MODEL_CACHE)


# ---------------------------------------------------------------------------
# Stage 1: Fetch satellite imagery from Esri
# ---------------------------------------------------------------------------
def fetch_esri_image(
    lat: float,
    lon: float,
    buffer_meters: float = 150.0,
    img_size: int = 600,
    timeout: float = 30.0,
) -> Optional[np.ndarray]:
    """Fetch Esri World Imagery centered at (lat, lon).

    Defaults match UBC training:
      buffer_meters=150, img_size=600 → 300m x 300m ground, 0.5 m/px GSD.

    Returns a BGR OpenCV image or None on failure.
    """
    endpoint = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/export")

    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * math.cos(math.radians(lat))
    lat_off = buffer_meters / lat_m_per_deg
    lon_off = buffer_meters / lon_m_per_deg

    bbox = f"{lon - lon_off},{lat - lat_off},{lon + lon_off},{lat + lat_off}"
    params = {
        "bbox":    bbox,
        "bboxSR":  "4326",
        "imageSR": "102100",
        "size":    f"{img_size},{img_size}",
        "format":  "png",
        "f":       "image",
    }
    headers = {"User-Agent": "Mozilla/5.0 (UBC-Inference-Pipeline/1.0)"}

    try:
        resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        print(f"[fetch_esri_image] Network error: {e}")
        return None

    if resp.status_code != 200:
        print(f"[fetch_esri_image] HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    arr = np.asarray(bytearray(resp.content), dtype="uint8")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[fetch_esri_image] cv2 failed to decode response")
        return None
    return img


# ---------------------------------------------------------------------------
# Stage 2: Inference
# ---------------------------------------------------------------------------
def run_ubc_inference(
    image_bgr: np.ndarray,
    task: str,
    score_threshold: float = 0.3,
    device: str = "cuda:0",
):
    """Run a cached UBC Cascade Mask R-CNN on a BGR image.

    Returns an MMDetection DetDataSample filtered to score >= threshold.
    """
    from mmdet.apis import inference_detector

    model = get_model(task, device=device)
    result = inference_detector(model, image_bgr)

    pred = result.pred_instances
    keep = pred.scores >= score_threshold
    result.pred_instances = pred[keep]
    return result


# ---------------------------------------------------------------------------
# Stage 3: Visualization
# ---------------------------------------------------------------------------
def visualize_predictions(
    image_bgr: np.ndarray,
    result,
    task: str,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Overlay predicted masks, boxes, labels, and a class legend on the image.

    Returns a new BGR image.
    """
    reg = TASK_REGISTRY[task]
    class_names = reg["classes"]
    colors = reg["colors"]

    vis = image_bgr.copy()
    pred = result.pred_instances
    if len(pred) == 0:
        cv2.putText(vis, "No buildings detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    def _tocpu(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else t

    masks  = _tocpu(pred.masks)
    # boxes  = _tocpu(pred.bboxes)
    scores = _tocpu(pred.scores)
    labels = _tocpu(pred.labels)

    # Masks (painted first, under boxes/labels for readability)
    overlay = vis.copy()
    for mask, label in zip(masks, labels):
        overlay[mask.astype(bool)] = colors[int(label) % len(colors)]
    vis = cv2.addWeighted(overlay, mask_alpha, vis, 1 - mask_alpha, 0)

    # Boxes + text
    # for box, score, label in zip(boxes, scores, labels):
    #     x1, y1, x2, y2 = map(int, box)
    #     color = colors[int(label) % len(colors)]
    #     cname = class_names[int(label)]
    #     cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    #     text = f"{cname}: {score:.2f}"
    #     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    #     cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    #     cv2.putText(vis, text, (x1 + 2, y1 - 4),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Legend
    unique = sorted(set(int(l) for l in labels))
    x0, y0 = vis.shape[1] - 180, 20
    cv2.rectangle(vis, (x0 - 8, y0 - 16),
                  (vis.shape[1] - 4, y0 + 20 * len(unique) - 4),
                  (0, 0, 0), -1)
    for i, l in enumerate(unique):
        y = y0 + i * 20
        cv2.rectangle(vis, (x0, y - 10), (x0 + 12, y), colors[l % len(colors)], -1)
        cv2.putText(vis, class_names[l], (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Convenience: structured predictions for JSON responses
# ---------------------------------------------------------------------------
def predictions_to_dict(result, task: str) -> list:
    """Serialize DetDataSample predictions to JSON-friendly list of dicts."""
    reg = TASK_REGISTRY[task]
    class_names = reg["classes"]

    pred = result.pred_instances
    if len(pred) == 0:
        return []

    def _tocpu(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else t

    boxes  = _tocpu(pred.bboxes)
    scores = _tocpu(pred.scores)
    labels = _tocpu(pred.labels)

    out = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(float, box)
        out.append({
            "class_id":   int(label),
            "class_name": class_names[int(label)],
            "score":      float(score),
            "bbox_xyxy":  [x1, y1, x2, y2],
        })
    return out
