"""
ubc_client.py — standalone Windows client for the UBC classification server.

Sends a GPS coordinate + task to the laptop server, displays the annotated
satellite image in an OpenCV window, and prints a summary of detected
buildings to the console.

Usage
-----
    # Demo: Munich city center, building functions
    python ubc_client.py --lat 48.1374 --lon 11.5755 --task use_coarse

    # Roof types in Beijing Chaoyang
    python ubc_client.py --lat 39.9200 --lon 116.4500 --task roof_coarse

    # Hit a different server host/port
    python ubc_client.py --lat 48.1374 --lon 11.5755 \
        --server http://192.168.1.11:8000

    # Save annotated output instead of (or in addition to) displaying
    python ubc_client.py --lat 48.1374 --lon 11.5755 --save munich.png

Requirements (Windows venv):
    pip install requests opencv-python numpy
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests


DEFAULT_SERVER = "http://192.168.1.11:8000"


def check_health(server: str, timeout: float = 5.0) -> dict:
    """Call /health to verify server is reachable. Returns JSON or raises."""
    r = requests.get(f"{server}/health", timeout=timeout)
    r.raise_for_status()
    return r.json()


def classify_gps(
    server: str,
    lat: float,
    lon: float,
    task: str,
    buffer_meters: float = 150.0,
    img_size: int = 600,
    score_threshold: float = 0.3,
    timeout: float = 60.0,
) -> dict:
    """POST to /classify/gps, return parsed JSON response."""
    payload = {
        "lat": lat,
        "lon": lon,
        "task": task,
        "buffer_meters": buffer_meters,
        "img_size": img_size,
        "score_threshold": score_threshold,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{server}/classify/gps", json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    data["_client_roundtrip_sec"] = round(time.perf_counter() - t0, 3)
    return data


def decode_b64_png(b64: str) -> np.ndarray:
    """Decode a base64-encoded PNG into a BGR OpenCV image."""
    png_bytes = base64.b64decode(b64)
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def summarize_detections(detections: list) -> None:
    """Print a by-class count + top-N detection table to stdout."""
    if not detections:
        print("  (no buildings detected above threshold)")
        return

    # Count per class
    by_class = {}
    for d in detections:
        by_class.setdefault(d["class_name"], []).append(d["score"])

    # Sorted by count descending
    print(f"  {len(detections)} total buildings detected:")
    for cname, scores in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
        max_s = max(scores)
        mean_s = sum(scores) / len(scores)
        print(f"    {cname:20s}  n={len(scores):3d}  "
              f"max={max_s:.2f}  mean={mean_s:.2f}")


def show_image(img: np.ndarray, window_title: str) -> None:
    """Display image in a resizable OpenCV window until user presses any key."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, min(img.shape[1], 900), min(img.shape[0], 900))
    cv2.imshow(window_title, img)
    print("\n[display] Window open — press any key (or close window) to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(
        description="Standalone client for the UBC classification server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--lat", type=float, required=True, help="Latitude (WGS84)")
    p.add_argument("--lon", type=float, required=True, help="Longitude (WGS84)")
    p.add_argument("--task", default="use_coarse",
                   choices=["roof_coarse", "roof_fine", "use_coarse"],
                   help="Which UBC task to run")
    p.add_argument("--server", default=DEFAULT_SERVER,
                   help="Server URL (default: laptop on LAN)")
    p.add_argument("--buffer-meters", type=float, default=150.0)
    p.add_argument("--img-size", type=int, default=600)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--save", type=Path, default=None,
                   help="Save annotated image to this path")
    p.add_argument("--no-display", action="store_true",
                   help="Don't open an OpenCV window (useful with --save)")
    args = p.parse_args()

    # Step 1: health check
    print(f"[client] Pinging {args.server}/health ...")
    try:
        health = check_health(args.server)
        print(f"[client] Server OK. Tasks loaded: {health.get('tasks_loaded', [])}")
    except requests.RequestException as e:
        print(f"[client] ERROR: could not reach server — {e}", file=sys.stderr)
        sys.exit(2)

    # Step 2: classify
    print(f"[client] Classifying ({args.lat}, {args.lon}) as '{args.task}' ...")
    try:
        resp = classify_gps(
            args.server, args.lat, args.lon, args.task,
            buffer_meters=args.buffer_meters,
            img_size=args.img_size,
            score_threshold=args.score_threshold,
        )
    except requests.RequestException as e:
        print(f"[client] ERROR during classification: {e}", file=sys.stderr)
        sys.exit(3)

    # Step 3: print summary
    meta = resp["meta"]
    print(f"\n[result] task={meta['task']}  "
          f"inference={meta['inference_sec']}s  "
          f"total_server={meta['total_sec']}s  "
          f"roundtrip={resp['_client_roundtrip_sec']}s")
    summarize_detections(resp["detections"])

    # Step 4: decode + display/save annotated image
    annotated = decode_b64_png(resp["annotated_image_b64"])

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), annotated)
        print(f"\n[client] Annotated image saved to {args.save}")

    if not args.no_display:
        title = f"UBC {args.task} @ ({args.lat:.4f}, {args.lon:.4f})"
        show_image(annotated, title)


if __name__ == "__main__":
    main()