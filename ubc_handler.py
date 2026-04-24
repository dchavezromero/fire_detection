"""
UBC Building Classification Handler
====================================
Manages the 'U' key workflow for the fire detection pipeline: fetch Esri
satellite imagery at the demo GPS, run a trained Cascade Mask R-CNN model,
and display the annotated result in its own OpenCV window.

Runs in-process (no HTTP server) — the pipeline and classifier share one
Python process and one GPU.

The Esri fetch is ~1-1.5s of network I/O, so queries run in a background
thread to keep the video loop responsive. Inference itself is ~100 ms on
an RTX 5080 once the model is loaded.
"""

import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig


# Make ubc_inference importable — it lives in ubc_server/ not at repo root
UBC_SERVER_DIR = Path(__file__).resolve().parent / "ubc_classifier"
if str(UBC_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(UBC_SERVER_DIR))
import ubc_inference as ubc   # noqa: E402


class UBCQueryHandler:
    """Owns the 'press U to classify buildings' workflow.

    The pipeline owns one instance of this class and:
      1. Calls `warmup()` once at startup (preloads the model, ~3 s).
      2. Calls `trigger()` on each 'U' keypress (starts a background query).
      3. Calls `repaint_if_needed()` once per video loop iteration
         (updates the UBC window if the background query finished).
    """

    WINDOW_NAME = "UBC Building Classification"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest_result = None
        self._pending = False
        self._last_error = None
        self._needs_repaint = False

    # ----- Public API ------------------------------------------------------

    def warmup(self):
        """Eagerly load the configured UBC model so the first keypress is fast."""
        print(f"[UBC] Warming up '{self.cfg.ubc_task}' model...")
        ubc.get_model(self.cfg.ubc_task)
        print(f"[UBC] Ready. Press 'U' to classify buildings at "
              f"({self.cfg.ubc_demo_lat}, {self.cfg.ubc_demo_lon})")

    def trigger(self):
        """Kick off a background UBC query. No-op if one is already in flight."""
        if self._pending:
            print("[UBC] Query already in flight, ignoring keypress")
            return
        self._pending = True
        self._last_error = None
        print(f"[UBC] Classifying ({self.cfg.ubc_demo_lat}, {self.cfg.ubc_demo_lon}) "
              f"as '{self.cfg.ubc_task}' ...")
        threading.Thread(target=self._run_query, daemon=True).start()

    def repaint_if_needed(self):
        """Called each main-loop iteration. Shows new results when they arrive."""
        with self._lock:
            if not self._needs_repaint:
                return
            self._needs_repaint = False
            img = self._latest_result
            err = self._last_error

        if err is not None:
            self._show_error_card(err)
        elif img is not None:
            cv2.imshow(self.WINDOW_NAME, img)

    # ----- Internal --------------------------------------------------------

    def _run_query(self):
        """Thread target: fetch → infer → visualize, then hand off to repaint."""
        try:
            t0 = time.perf_counter()

            img = ubc.fetch_esri_image(
                lat=self.cfg.ubc_demo_lat,
                lon=self.cfg.ubc_demo_lon,
                buffer_meters=self.cfg.ubc_buffer_meters,
                img_size=self.cfg.ubc_img_size,
            )
            if img is None:
                raise RuntimeError("Esri fetch failed (network error or rate limit)")

            result = ubc.run_ubc_inference(
                img, self.cfg.ubc_task,
                score_threshold=self.cfg.ubc_score_threshold,
            )
            annotated = ubc.visualize_predictions(img, result, self.cfg.ubc_task)

            n = len(result.pred_instances)
            ms = int((time.perf_counter() - t0) * 1000)
            print(f"[UBC] {n} buildings detected in {ms} ms")

            with self._lock:
                self._latest_result = annotated
                self._needs_repaint = True
        except Exception as e:
            msg = f"UBC query failed: {e}"
            print(f"[UBC] ERROR: {msg}")
            with self._lock:
                self._last_error = msg
                self._needs_repaint = True
        finally:
            self._pending = False

    def _show_error_card(self, err_msg: str):
        """Render a small error card into the UBC window."""
        card = np.zeros((200, 600, 3), dtype=np.uint8)
        cv2.rectangle(card, (0, 0), (600, 200), (0, 0, 80), -1)
        cv2.putText(card, "UBC Query Error", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(card, err_msg[:70], (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow(self.WINDOW_NAME, card)