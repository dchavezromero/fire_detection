"""
Fire Detection Pipeline
========================
Two-stage fire detection combining FFireNet (binary classifier) and
YOLOv8 (object detector) with cross-correlation and temporal smoothing.

Stage 1: FFireNet classifies each frame (fire / no-fire probability).
Stage 2: YOLO runs conditionally based on FFireNet confidence.
Stage 3: Cross-correlation resolves disagreements between models.
Stage 4: Temporal smoothing over a rolling window prevents flickering.

Usage:
    python fire_pipeline.py
"""

import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from torchvision import transforms, models
from ultralytics import YOLO


# ============================================================
# Pipeline Configuration
# ============================================================
@dataclass
class PipelineConfig:
    """All tunable thresholds in one place."""

    # --- Paths ---
    ffirenet_model_path: str = "models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth"
    yolo_model_path: str = "models/26m_1280imgsz_200epochs/weights/best.pt"
    video_path: str = "sample_videos/fire3.mp4"

    # --- FFireNet gating thresholds ---
    # FFireNet sigmoid output: 0.0 = fire, 1.0 = no fire
    # Frames below ffirenet_fire_thresh    -> high confidence fire
    # Frames above ffirenet_nofire_thresh  -> high confidence no fire
    # Frames in between                   -> uncertain, always run YOLO
    ffirenet_fire_thresh: float = 0.3      # below this = confident fire
    ffirenet_nofire_thresh: float = 0.7    # above this = confident no fire

    # --- YOLO settings ---
    yolo_conf: float = 0.3                 # base YOLO confidence threshold
    yolo_recheck_conf: float = 0.15        # lower threshold when cross-checking disagreements
    yolo_device: int = 0                   # GPU device (0) or "cpu"

    # --- Cross-correlation weights ---
    # Final frame score = (w_ffirenet * ffirenet_score) + (w_yolo * yolo_score)
    # Both scores normalized to 0.0 = no fire, 1.0 = fire
    w_ffirenet: float = 0.4
    w_yolo: float = 0.6

    # --- Temporal smoothing ---
    window_size: int = 8                   # rolling window frame count
    alert_threshold: float = 0.55          # fused score above this = fire alert
    min_frames_for_alert: int = 4          # need at least N frames in window above threshold

    # --- Display ---
    show_display: bool = True
    save_output: bool = False
    output_path: str = "output_pipeline.mp4"


# ============================================================
# FFireNet Model (must match training)
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
# Preprocessing (must match FFireNet training)
# ============================================================
ffirenet_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# Pipeline
# ============================================================
class FireDetectionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load FFireNet
        self.ffirenet = FFireNet().to(self.device)
        checkpoint = torch.load(cfg.ffirenet_model_path, map_location=self.device, weights_only=False)
        self.ffirenet.load_state_dict(checkpoint["model_state_dict"])
        self.ffirenet.eval()
        print(f"[PIPELINE] FFireNet loaded from {cfg.ffirenet_model_path}")

        # Load YOLO
        self.yolo = YOLO(cfg.yolo_model_path)
        print(f"[PIPELINE] YOLO loaded from {cfg.yolo_model_path}")
        print(f"[PIPELINE] Device: {self.device}")

        # Temporal smoothing buffer: stores per-frame fused scores
        self.score_window = deque(maxlen=cfg.window_size)

        # Stats
        self.stats = {
            "total_frames": 0,
            "yolo_runs": 0,
            "alerts": 0,
        }

    # ----------------------------------------------------------
    # Stage 1: FFireNet inference
    # ----------------------------------------------------------
    def run_ffirenet(self, frame_rgb: np.ndarray) -> float:
        """Returns fire probability (0 = no fire, 1 = fire)."""
        tensor = ffirenet_preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.ffirenet(tensor).squeeze()
            prob = torch.sigmoid(logit).item()
        # FFireNet: prob close to 0 = fire, close to 1 = nofire
        # Invert so our score means: 1.0 = fire, 0.0 = no fire
        return 1.0 - prob

    # ----------------------------------------------------------
    # Stage 2: YOLO inference (conditional)
    # ----------------------------------------------------------
    def run_yolo(self, frame, conf: float) -> tuple[list, float]:
        """
        Returns (detections_list, yolo_fire_score).
        yolo_fire_score: 0.0 = nothing found, scales up with count & confidence.
        """
        results = self.yolo.predict(
            source=frame,
            conf=conf,
            show=False,
            save=False,
            device=self.cfg.yolo_device,
            verbose=False,
        )
        self.stats["yolo_runs"] += 1

        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                det_conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": det_conf,
                    "class_id": cls_id,
                })

        # YOLO fire score: combine detection count and max confidence
        if detections:
            max_conf = max(d["confidence"] for d in detections)
            count_factor = min(len(detections) / 3.0, 1.0)  # saturates at 3 detections
            yolo_score = 0.5 * max_conf + 0.5 * count_factor
        else:
            yolo_score = 0.0

        return detections, yolo_score

    # ----------------------------------------------------------
    # Stage 3: Cross-correlation & fusion
    # ----------------------------------------------------------
    def fuse_scores(
        self,
        ffirenet_score: float,
        yolo_score: float | None,
        yolo_ran: bool,
    ) -> float:
        """
        Combine model outputs into a single fire score [0, 1].
        If YOLO didn't run, rely on FFireNet alone.
        """
        if not yolo_ran or yolo_score is None:
            return ffirenet_score

        fused = (self.cfg.w_ffirenet * ffirenet_score) + (self.cfg.w_yolo * yolo_score)

        # Disagreement penalty: if models strongly disagree, dampen confidence
        disagreement = abs(ffirenet_score - yolo_score)
        if disagreement > 0.5:
            # Pull score toward 0.5 (uncertain) proportionally to disagreement
            penalty = 0.2 * (disagreement - 0.5)
            fused = fused * (1 - penalty) + 0.5 * penalty

        return np.clip(fused, 0.0, 1.0)

    # ----------------------------------------------------------
    # Stage 4: Temporal smoothing
    # ----------------------------------------------------------
    def temporal_decision(self, fused_score: float) -> tuple[bool, float]:
        """
        Push score into rolling window. Returns (fire_alert, smoothed_score).
        """
        self.score_window.append(fused_score)

        if len(self.score_window) < 2:
            return False, fused_score

        smoothed = np.mean(self.score_window)
        frames_above = sum(1 for s in self.score_window if s > self.cfg.alert_threshold)
        alert = (smoothed > self.cfg.alert_threshold) and (frames_above >= self.cfg.min_frames_for_alert)

        return alert, smoothed

    # ----------------------------------------------------------
    # Process single frame
    # ----------------------------------------------------------
    def process_frame(self, frame) -> dict:
        """Full pipeline for one frame. Returns info dict."""
        self.stats["total_frames"] += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Stage 1: FFireNet
        ffirenet_score = self.run_ffirenet(frame_rgb)

        # Stage 2: Decide whether to run YOLO
        yolo_ran = False
        yolo_score = None
        detections = []

        if ffirenet_score > (1.0 - self.cfg.ffirenet_fire_thresh):
            # High confidence fire -> run YOLO at normal conf to localize
            detections, yolo_score = self.run_yolo(frame, self.cfg.yolo_conf)
            yolo_ran = True

        elif ffirenet_score > (1.0 - self.cfg.ffirenet_nofire_thresh):
            # Uncertain zone -> run YOLO to help decide
            detections, yolo_score = self.run_yolo(frame, self.cfg.yolo_conf)
            yolo_ran = True

        else:
            # High confidence no fire -> but occasionally spot-check
            # Run YOLO every 30 frames as a safety net
            if self.stats["total_frames"] % 30 == 0:
                detections, yolo_score = self.run_yolo(frame, self.cfg.yolo_recheck_conf)
                yolo_ran = True

                # If YOLO finds something FFireNet missed, flag it
                if yolo_score > 0.3:
                    ffirenet_score = max(ffirenet_score, 0.5)  # bump FFireNet up

        # Stage 3: Fuse
        fused_score = self.fuse_scores(ffirenet_score, yolo_score, yolo_ran)

        # Stage 4: Temporal
        alert, smoothed = self.temporal_decision(fused_score)
        if alert:
            self.stats["alerts"] += 1

        return {
            "ffirenet_score": ffirenet_score,
            "yolo_score": yolo_score,
            "yolo_ran": yolo_ran,
            "detections": detections,
            "fused_score": fused_score,
            "smoothed_score": smoothed,
            "alert": alert,
        }

    # ----------------------------------------------------------
    # Drawing
    # ----------------------------------------------------------
    def draw_overlay(self, frame, info: dict, dt_ms: float):
        h, w = frame.shape[:2]

        # YOLO bounding boxes
        for det in info["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"fire {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Alert banner
        if info["alert"]:
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 180), -1)
            cv2.putText(frame, "FIRE ALERT", (w // 2 - 120, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

        # Scores panel (bottom-left)
        panel_y = h - 130
        cv2.rectangle(frame, (0, panel_y), (340, h), (0, 0, 0), -1)

        lines = [
            f"FFireNet: {info['ffirenet_score']:.2f}",
            f"YOLO:     {info['yolo_score']:.2f}" if info["yolo_ran"] else "YOLO:     skipped",
            f"Fused:    {info['fused_score']:.2f}",
            f"Smoothed: {info['smoothed_score']:.2f}",
            f"{dt_ms:.0f}ms | YOLO calls: {self.stats['yolo_runs']}",
        ]

        for i, line in enumerate(lines):
            color = (0, 255, 0) if info["smoothed_score"] < self.cfg.alert_threshold else (0, 0, 255)
            cv2.putText(frame, line, (10, panel_y + 22 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        return frame

    # ----------------------------------------------------------
    # Run on video
    # ----------------------------------------------------------
    def run(self):
        cap = cv2.VideoCapture(self.cfg.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.cfg.video_path}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[PIPELINE] Video: {self.cfg.video_path} ({total} frames @ {fps_video:.0f} FPS)")

        writer = None
        if self.cfg.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.cfg.output_path, fourcc, fps_video, (w, h))

        print("[PIPELINE] Running... press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            info = self.process_frame(frame)
            dt_ms = (time.perf_counter() - t0) * 1000

            frame = self.draw_overlay(frame, info, dt_ms)

            if writer:
                writer.write(frame)

            if self.cfg.show_display:
                cv2.imshow("Fire Detection Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Summary
        total_f = self.stats["total_frames"]
        yolo_r = self.stats["yolo_runs"]
        pct = (yolo_r / total_f * 100) if total_f > 0 else 0
        print(f"\n[PIPELINE] Done. {total_f} frames processed.")
        print(f"[PIPELINE] YOLO ran on {yolo_r}/{total_f} frames ({pct:.1f}%)")
        print(f"[PIPELINE] Alert frames: {self.stats['alerts']}")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    cfg = PipelineConfig(
        # --- Paths (edit these) ---
        ffirenet_model_path="models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth",
        yolo_model_path="models/26m_1280imgsz_200epochs/weights/best.pt",
        video_path="sample_videos/fire4.mp4",

        # --- Tune these ---
        ffirenet_fire_thresh=0.3,
        ffirenet_nofire_thresh=0.7,
        yolo_conf=0.3,
        yolo_recheck_conf=0.15,
        w_ffirenet=0.4,
        w_yolo=0.6,
        window_size=8,
        alert_threshold=0.55,
        min_frames_for_alert=4,
    )

    pipeline = FireDetectionPipeline(cfg)
    pipeline.run()