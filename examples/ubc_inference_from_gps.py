"""
ubc_inference_from_gps.py
=========================
End-to-end demo of the UBC building classification stage of the fire detection
pipeline. Given a GPS coordinate (as a drone would provide), this script:

  1. Fetches a satellite image from Esri World Imagery at the UBC training
     resolution (~0.5 m/px GSD, 600x600 px, covering ~300x300 m ground area).
  2. Runs a trained Cascade Mask R-CNN on the image (one of three UBC tasks:
     roof_coarse, roof_fine, use_coarse).
  3. Overlays predicted building masks + class labels + confidence scores and
     saves the annotated output.

Designed as three modular functions (fetch_esri_image, run_ubc_inference,
visualize_predictions) that can be imported into the larger fire pipeline.

Usage
-----
  # Empire State Building, roof_coarse model
  python ubc_inference_from_gps.py \
      --lat 40.7484 --lon -73.9857 \
      --task roof_coarse

  # Munich city center, function classification
  python ubc_inference_from_gps.py \
      --lat 48.1374 --lon 11.5755 \
      --task use_coarse \
      --out-dir ./outputs/munich_center

  # Use a specific checkpoint explicitly
  python ubc_inference_from_gps.py \
      --lat 39.9349 --lon 116.3162 \
      --task roof_fine \
      --checkpoint /path/to/custom.pth

Requires the UBC dataset classes to be registered in MMDetection (i.e., the
Phase 1 setup with ~/mmdetection/mmdet/datasets/ubc.py installed).
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


# ---------------------------------------------------------------------------
# Task -> (config, best-checkpoint-glob) registry
# ---------------------------------------------------------------------------
# Default paths follow the Phase 1 README's layout. Override via --checkpoint
# or --config CLI flags if your layout differs.
MMDET_ROOT = Path.home() / "mmdetection"

TASK_REGISTRY = {
    "roof_coarse": {
        "config":       MMDET_ROOT / "configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse.py",
        "work_dir":     MMDET_ROOT / "work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_coarse",
        "classes":      ("flat", "gable", "hipped", "arched", "other"),
        "label_colors": [(239, 68, 68), (59, 130, 246), (168, 85, 247),
                         (251, 191, 36), (156, 163, 175)],
    },
    "roof_fine": {
        "config":       MMDET_ROOT / "configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_roof_fine.py",
        "work_dir":     MMDET_ROOT / "work_dirs/cascade-mask-rcnn_r50_fpn_ubc_roof_fine_bs6_wider_run3",
        "classes":      ("flat", "flat_roof_complex", "shed_roof", "gable_roof",
                         "gambrel_roof", "hipped_roof_v1", "hipped_roof_v2",
                         "mansard_roof", "pinnacle_roof", "arched", "other"),
        "label_colors": [(239, 68, 68), (248, 113, 113), (217, 70, 239),
                         (59, 130, 246), (37, 99, 235), (168, 85, 247),
                         (147, 51, 234), (251, 191, 36), (245, 158, 11),
                         (34, 197, 94), (156, 163, 175)],
    },
    "use_coarse": {
        "config":       MMDET_ROOT / "configs/ubc/cascade-mask-rcnn_r50_fpn_ubc_use_coarse.py",
        "work_dir":     MMDET_ROOT / "work_dirs/cascade-mask-rcnn_r50_fpn_ubc_use_coarse_run1",
        "classes":      ("residential", "commercial", "industrial", "public", "other"),
        "label_colors": [(34, 197, 94), (59, 130, 246), (168, 85, 247),
                         (251, 191, 36), (156, 163, 175)],
    },
}


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
    """Fetch a satellite image from Esri World Imagery centered at (lat, lon).

    Defaults match UBC v1's training distribution:
      * buffer_meters=150 => 300m x 300m ground area
      * img_size=600 => 600x600 px output
      * Combined: 300m / 600 px = 0.5 m/px GSD (matches SuperView bands in UBC)

    Returns a BGR OpenCV image, or None on failure.
    """
    endpoint = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/export")

    # 1 deg latitude ≈ 111,320 m; longitude scales with cos(lat)
    lat_m_per_deg = 111320.0
    lon_m_per_deg = 111320.0 * math.cos(math.radians(lat))
    lat_off = buffer_meters / lat_m_per_deg
    lon_off = buffer_meters / lon_m_per_deg

    # bbox: xmin, ymin, xmax, ymax — lon is X, lat is Y
    bbox = f"{lon - lon_off},{lat - lat_off},{lon + lon_off},{lat + lat_off}"

    params = {
        "bbox":     bbox,
        "bboxSR":   "4326",     # WGS84 / GPS coords
        "imageSR":  "102100",   # Web Mercator (Esri native)
        "size":     f"{img_size},{img_size}",
        "format":   "png",
        "f":        "image",
    }
    headers = {
        # Some Esri endpoints reject default python-requests user-agent with 403
        "User-Agent": "Mozilla/5.0 (UBC-Inference-Pipeline/1.0)",
    }

    print(f"[fetch_esri_image] Requesting {img_size}x{img_size} px image for "
          f"({lat:.5f}, {lon:.5f}), buffer {buffer_meters} m "
          f"(GSD ~{2 * buffer_meters / img_size:.2f} m/px)")

    try:
        resp = requests.get(endpoint, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        print(f"[fetch_esri_image] Network error: {e}")
        return None

    if resp.status_code != 200:
        print(f"[fetch_esri_image] Esri returned HTTP {resp.status_code}")
        print(f"[fetch_esri_image] Response: {resp.text[:200]}")
        return None

    # Decode PNG bytes into a BGR numpy array
    arr = np.asarray(bytearray(resp.content), dtype="uint8")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[fetch_esri_image] cv2 failed to decode response bytes")
        return None

    print(f"[fetch_esri_image] OK — got {img.shape[1]}x{img.shape[0]} px image")
    return img


# ---------------------------------------------------------------------------
# Stage 2: Run UBC Cascade Mask R-CNN inference
# ---------------------------------------------------------------------------
def _find_best_checkpoint(work_dir: Path) -> Optional[Path]:
    """Locate the `best_coco_segm_mAP_epoch_*.pth` saved by training.

    Returns the path with the highest epoch number, or None if no match.
    """
    if not work_dir.exists():
        return None
    candidates = list(work_dir.glob("best_coco_segm_mAP_epoch_*.pth"))
    if not candidates:
        # Fall back to latest `epoch_*.pth` if no best-tracked symlink exists
        candidates = sorted(work_dir.glob("epoch_*.pth"))
        return candidates[-1] if candidates else None
    # Pick the highest epoch number
    candidates.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return candidates[-1]


def run_ubc_inference(
    image_bgr: np.ndarray,
    task: str,
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda:0",
    score_threshold: float = 0.3,
):
    """Run Cascade Mask R-CNN on a BGR image for the selected UBC task.

    Returns a DetDataSample from MMDetection's API, which contains:
      * pred_instances.bboxes   (N, 4)  xyxy
      * pred_instances.scores   (N,)    [0, 1]
      * pred_instances.labels   (N,)    class indices
      * pred_instances.masks    (N, H, W) binary masks

    Filtered in-place to only predictions with score >= score_threshold.
    """
    # Lazy import — keeps the module loadable in environments without MMDetection
    # (e.g., a pipeline wrapper that dispatches tasks without always loading the model)
    from mmdet.apis import init_detector, inference_detector

    registry = TASK_REGISTRY[task]
    config_path = Path(config_path) if config_path else registry["config"]
    if checkpoint_path is None:
        checkpoint_path = _find_best_checkpoint(registry["work_dir"])
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {registry['work_dir']}. "
                f"Pass --checkpoint explicitly or train the {task} model first."
            )
    checkpoint_path = Path(checkpoint_path)

    print(f"[run_ubc_inference] Task: {task}")
    print(f"[run_ubc_inference] Config: {config_path}")
    print(f"[run_ubc_inference] Checkpoint: {checkpoint_path}")
    print(f"[run_ubc_inference] Device: {device}")

    model = init_detector(str(config_path), str(checkpoint_path), device=device)

    # MMDetection's inference_detector expects BGR (OpenCV-standard) — we're fine
    result = inference_detector(model, image_bgr)

    # Filter to confident predictions only
    pred = result.pred_instances
    keep = pred.scores >= score_threshold
    result.pred_instances = pred[keep]

    n = len(result.pred_instances)
    print(f"[run_ubc_inference] {n} buildings detected above score={score_threshold}")
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
    """Overlay predicted masks, class labels, and scores on the input image.

    Returns a new BGR image (does not modify input). Each building gets:
      * A colored translucent mask (colors come from TASK_REGISTRY)
      * A bounding box outline
      * A text label "class_name: 0.XX" near the top-left corner
    """
    registry = TASK_REGISTRY[task]
    class_names = registry["classes"]
    colors = registry["label_colors"]

    vis = image_bgr.copy()
    pred = result.pred_instances

    if len(pred) == 0:
        cv2.putText(vis, "No buildings detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis

    masks = pred.masks.cpu().numpy() if hasattr(pred.masks, "cpu") else pred.masks
    boxes = pred.bboxes.cpu().numpy() if hasattr(pred.bboxes, "cpu") else pred.bboxes
    scores = pred.scores.cpu().numpy() if hasattr(pred.scores, "cpu") else pred.scores
    labels = pred.labels.cpu().numpy() if hasattr(pred.labels, "cpu") else pred.labels

    # Draw masks first (painted under boxes and labels for readability)
    overlay = vis.copy()
    for mask, label in zip(masks, labels):
        color = colors[int(label) % len(colors)]
        overlay[mask.astype(bool)] = color
    vis = cv2.addWeighted(overlay, mask_alpha, vis, 1 - mask_alpha, 0)

    # Then boxes + text
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = colors[int(label) % len(colors)]
        cname = class_names[int(label)]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{cname}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        # Text background for legibility against varied imagery
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                    cv2.LINE_AA)

    # Add a small legend in the top-right corner showing which classes appear
    unique_labels = sorted(set(int(l) for l in labels))
    legend_lines = [(class_names[l], colors[l % len(colors)]) for l in unique_labels]
    x0, y0 = vis.shape[1] - 180, 20
    cv2.rectangle(vis, (x0 - 8, y0 - 16),
                  (vis.shape[1] - 4, y0 + 20 * len(legend_lines) - 4),
                  (0, 0, 0), -1)
    for i, (name, color) in enumerate(legend_lines):
        y = y0 + i * 20
        cv2.rectangle(vis, (x0, y - 10), (x0 + 12, y), color, -1)
        cv2.putText(vis, name, (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return vis


# ---------------------------------------------------------------------------
# Glue: single-image demo entry point
# ---------------------------------------------------------------------------
def run_single_inference(
    lat: float,
    lon: float,
    task: str,
    out_dir: Path,
    buffer_meters: float = 150.0,
    img_size: int = 600,
    config_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    score_threshold: float = 0.3,
    device: str = "cuda:0",
) -> bool:
    """Full pipeline: fetch, infer, visualize, save. Returns True on success."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch
    img = fetch_esri_image(lat, lon, buffer_meters=buffer_meters, img_size=img_size)
    if img is None:
        return False

    stem = f"{lat:.5f}_{lon:.5f}_{task}".replace("-", "n")  # 'n' for negative
    raw_path = out_dir / f"{stem}_raw.png"
    cv2.imwrite(str(raw_path), img)
    print(f"[pipeline] Saved raw satellite image to {raw_path}")

    # 2. Infer
    result = run_ubc_inference(
        img, task,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        score_threshold=score_threshold,
    )

    # 3. Visualize + save
    vis = visualize_predictions(img, result, task)
    vis_path = out_dir / f"{stem}_annotated.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"[pipeline] Saved annotated image to {vis_path}")

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run a trained UBC Cascade Mask R-CNN on Esri satellite imagery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat",   type=float, required=True, help="Latitude (WGS84)")
    p.add_argument("--lon",   type=float, required=True, help="Longitude (WGS84)")
    p.add_argument("--task",
                   choices=list(TASK_REGISTRY.keys()),
                   default="roof_coarse",
                   help="Which UBC task to run")
    p.add_argument("--out-dir",       type=Path, default=Path("./inference_out"))
    p.add_argument("--buffer-meters", type=float, default=150.0,
                   help="Half-side of ground area in meters. 150 = 300x300 m tile.")
    p.add_argument("--img-size",      type=int, default=600,
                   help="Output image side in pixels. 600 matches UBC's 0.5 m/px training GSD.")
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--device",        default="cuda:0",
                   help="PyTorch device. Use 'cpu' if no GPU available.")
    p.add_argument("--config",        type=Path, default=None,
                   help="Override the default config path for the chosen task.")
    p.add_argument("--checkpoint",    type=Path, default=None,
                   help="Override the auto-discovered best checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    ok = run_single_inference(
        lat=args.lat,
        lon=args.lon,
        task=args.task,
        out_dir=args.out_dir,
        buffer_meters=args.buffer_meters,
        img_size=args.img_size,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        score_threshold=args.score_threshold,
        device=args.device,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
