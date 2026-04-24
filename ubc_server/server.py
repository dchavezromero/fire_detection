"""
server.py — FastAPI server for UBC building classification.

Runs on the laptop, listens on 0.0.0.0:PORT so the Windows desktop can reach it
over LAN. Windows fire pipeline posts GPS coordinates; server fetches Esri
imagery, runs Cascade Mask R-CNN, returns JSON with detections and a base64
annotated image.

Launch:
    cd ~/ubc_server
    python server.py               # binds to 0.0.0.0:8000
    python server.py --port 9000   # custom port
    python server.py --warmup use_coarse roof_coarse   # preload models at startup

Endpoints:
    GET  /health                 → {"status": "ok", "cached_tasks": [...]}
    GET  /tasks                  → list of available task names
    POST /classify/gps           → body {lat, lon, task[, buffer_meters, img_size, score_threshold]}
                                   returns {detections, annotated_image_b64, raw_image_b64, meta}
    POST /classify/image         → multipart image upload + task query param
                                   returns same shape as /classify/gps
"""

from __future__ import annotations

import argparse
import base64
import io
import time
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import ubc_inference as ui


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ClassifyGPSRequest(BaseModel):
    lat: float = Field(..., description="Latitude (WGS84)")
    lon: float = Field(..., description="Longitude (WGS84)")
    task: str = Field(
        "use_coarse",
        description="Which UBC task to run: roof_coarse, roof_fine, use_coarse",
    )
    buffer_meters: float = Field(150.0, description="Half-side of ground area in meters")
    img_size: int = Field(600, description="Output image side in pixels")
    score_threshold: float = Field(0.3, description="Confidence filter for detections")


class Detection(BaseModel):
    class_id: int
    class_name: str
    score: float
    bbox_xyxy: list


class ClassifyResponse(BaseModel):
    detections: list[Detection]
    annotated_image_b64: str
    raw_image_b64: str
    meta: dict


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="UBC Building Classification Server",
    description="Cascade Mask R-CNN inference for the fire-detection pipeline",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tasks_available": list(ui.TASK_REGISTRY.keys()),
        "tasks_loaded": ui.cached_tasks(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        task: {
            "classes": list(reg["classes"]),
            "loaded": task in ui.cached_tasks(),
        }
        for task, reg in ui.TASK_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _encode_png_b64(img_bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR image as base64 PNG."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _run_pipeline(img_bgr: np.ndarray, task: str, score_threshold: float, t0: float) -> dict:
    """Stages 2+3: inference + visualization + response assembly."""
    t_inf_start = time.perf_counter()
    result = ui.run_ubc_inference(img_bgr, task, score_threshold=score_threshold)
    t_inf = time.perf_counter() - t_inf_start

    annotated = ui.visualize_predictions(img_bgr, result, task)
    detections = ui.predictions_to_dict(result, task)

    return {
        "detections":          detections,
        "annotated_image_b64": _encode_png_b64(annotated),
        "raw_image_b64":       _encode_png_b64(img_bgr),
        "meta": {
            "task":              task,
            "num_detections":    len(detections),
            "score_threshold":   score_threshold,
            "inference_sec":     round(t_inf, 3),
            "total_sec":         round(time.perf_counter() - t0, 3),
        },
    }


# ---------------------------------------------------------------------------
# /classify/gps — primary endpoint for the fire pipeline
# ---------------------------------------------------------------------------
@app.post("/classify/gps", response_model=ClassifyResponse)
def classify_gps(req: ClassifyGPSRequest):
    if req.task not in ui.TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")

    t0 = time.perf_counter()
    img = ui.fetch_esri_image(
        lat=req.lat,
        lon=req.lon,
        buffer_meters=req.buffer_meters,
        img_size=req.img_size,
    )
    if img is None:
        raise HTTPException(status_code=502, detail="Esri image fetch failed")

    resp = _run_pipeline(img, req.task, req.score_threshold, t0)
    resp["meta"].update({
        "lat":           req.lat,
        "lon":           req.lon,
        "buffer_meters": req.buffer_meters,
        "img_size":      req.img_size,
    })
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# /classify/image — bring-your-own-image (useful for testing without Esri)
# ---------------------------------------------------------------------------
@app.post("/classify/image", response_model=ClassifyResponse)
async def classify_image(
    file: UploadFile = File(...),
    task: str = Query("use_coarse"),
    score_threshold: float = Query(0.3),
):
    if task not in ui.TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    contents = await file.read()
    arr = np.asarray(bytearray(contents), dtype="uint8")
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")

    t0 = time.perf_counter()
    resp = _run_pipeline(img, task, score_threshold, t0)
    resp["meta"]["source"] = f"upload:{file.filename}"
    return JSONResponse(resp)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0", help="Bind address")
    p.add_argument("--port", type=int, default=8000, help="Listen port")
    p.add_argument("--warmup", nargs="*", default=[],
                   help="Task names to preload at startup (e.g. --warmup use_coarse)")
    p.add_argument("--workers", type=int, default=1,
                   help="Uvicorn worker count (keep 1 — models aren't fork-safe)")
    args = p.parse_args()

    if args.warmup:
        print(f"[server] Warming up models: {args.warmup}")
        ui.warmup_models(args.warmup)

    print(f"[server] Starting on http://{args.host}:{args.port}")
    print(f"[server] Available tasks: {list(ui.TASK_REGISTRY)}")
    print("[server] Endpoints: GET /health, GET /tasks, "
          "POST /classify/gps, POST /classify/image")

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
