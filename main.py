"""
Fire Detection Pipeline
========================
Two-stage fire detection: FFireNet acts as a lightweight gate that trips
on possible fire. Once tripped, YOLO26 takes over for a configurable
number of frames to localize and confirm. Conviction tracking on YOLO's
output drives the final fire alert.

Press 'U' during playback to classify buildings at the demo GPS using the
UBC Cascade Mask R-CNN model (see ubc_handler.py and PipelineConfig.ubc_*).

Usage:
    python main.py
"""

import re
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from ultralytics import YOLO

from config import PipelineConfig
from fusion import ConvictionTracker
from display import draw_overlay
from ubc_handler import UBCQueryHandler


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
# Utilities
# ============================================================
def parse_imgsz(model_path: str) -> int:
    """
    Extract image size from the FFireNet model folder name.

    Expected pattern: mobilenet_v2_{N}imgsz_...
    """
    match = re.search(r"(\d+)imgsz", model_path)
    if not match:
        raise ValueError(
            f"Cannot parse image size from model path: {model_path}\n"
            f"Expected folder name containing '<N>imgsz' (e.g. '640imgsz')"
        )
    size = int(match.group(1))
    print(f"[PIPELINE] Parsed FFireNet image size: {size}x{size}")
    return size


def build_ffirenet_preprocess(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_ffirenet(model_path: str, device: torch.device) -> FFireNet:
    model = FFireNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ============================================================
# Pipeline
# ============================================================
class FireDetectionPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ffirenet = load_ffirenet(cfg.ffirenet_model_path, self.device)
        img_size = parse_imgsz(cfg.ffirenet_model_path)
        self.ffirenet_preprocess = build_ffirenet_preprocess(img_size)
        print(f"[PIPELINE] FFireNet loaded from {cfg.ffirenet_model_path}")

        self.yolo = YOLO(cfg.yolo_model_path)
        print(f"[PIPELINE] YOLO loaded from {cfg.yolo_model_path}")
        print(f"[PIPELINE] Device: {self.device}")

        self.tracker = ConvictionTracker(cfg)
        self.gate_countdown = 0
        self.stats = {"total_frames": 0, "yolo_runs": 0, "alerts": 0}

        # UBC building classifier (press 'U' to query)
        self.ubc = UBCQueryHandler(cfg)
        self.ubc.warmup()

    # ----------------------------------------------------------
    # FFireNet inference
    # ----------------------------------------------------------
    def run_ffirenet(self, frame_rgb: np.ndarray) -> float:
        tensor = self.ffirenet_preprocess(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.ffirenet(tensor).squeeze()
            prob = torch.sigmoid(logit).item()
        return prob

    # ----------------------------------------------------------
    # YOLO inference
    # ----------------------------------------------------------
    def run_yolo(self, frame) -> tuple[list, float]:
        results = self.yolo.predict(
            source=frame,
            conf=self.cfg.yolo_conf,
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
                cls_name = self.cfg.class_names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": det_conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        fire_dets = [d for d in detections if d["class_name"] == "fire"]
        smoke_dets = [d for d in detections if d["class_name"] == "smoke"]

        score = 0.0
        if fire_dets:
            max_fire_conf = max(d["confidence"] for d in fire_dets)
            fire_count = min(len(fire_dets) / 3.0, 1.0)
            score += 0.5 * max_fire_conf + 0.3 * fire_count
        if smoke_dets:
            max_smoke_conf = max(d["confidence"] for d in smoke_dets)
            score += 0.2 * max_smoke_conf

        yolo_score = min(score, 1.0)
        return detections, yolo_score

    # ----------------------------------------------------------
    # Process single frame
    # ----------------------------------------------------------
    def process_frame(self, frame) -> dict:
        self.stats["total_frames"] += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ffirenet_sigmoid = self.run_ffirenet(frame_rgb)

        if ffirenet_sigmoid < self.cfg.gate_thresh:
            self.gate_countdown = self.cfg.gate_frames

        yolo_ran = False
        yolo_score = 0.0
        detections = []
        gate_open = self.gate_countdown > 0

        if gate_open:
            detections, yolo_score = self.run_yolo(frame)
            yolo_ran = True
            has_detections = len(detections) > 0

            if has_detections:
                self.gate_countdown = self.cfg.gate_frames
            else:
                self.gate_countdown -= 1

            alert, conviction = self.tracker.update(yolo_score, has_detections)
        else:
            alert, conviction = self.tracker.update(0.0, False)
            self.gate_countdown = 0

        if alert:
            self.stats["alerts"] += 1

        return {
            "ffirenet_score": ffirenet_sigmoid,
            "gate_open": gate_open,
            "gate_remaining": self.gate_countdown,
            "yolo_score": yolo_score,
            "yolo_ran": yolo_ran,
            "detections": detections,
            "conviction": conviction,
            "streak": self.tracker.streak,
            "alert": alert,
        }

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
        print(f"[PIPELINE] Keys: 'q' to quit, 'U' to classify buildings")

        writer = None
        if self.cfg.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.cfg.output_path, fourcc, fps_video, (w, h))

        print("[PIPELINE] Running...\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            info = self.process_frame(frame)
            dt_ms = (time.perf_counter() - t0) * 1000

            frame = draw_overlay(frame, info, self.stats, self.cfg, dt_ms)

            if writer:
                writer.write(frame)

            if self.cfg.show_display:
                cv2.imshow("Fire Detection Pipeline", frame)
                self.ubc.repaint_if_needed()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("u"), ord("U")):
                    self.ubc.trigger()

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

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
        ffirenet_model_path="models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth",
        yolo_model_path="models/26m_640imgsz_200epochs/weights/best.pt",
        video_path="sample_videos/fire1.mp4",

        ubc_demo_lat = 22.44647,           # Wang Fuk Court fire (default demo location)
        ubc_demo_lon = 114.17627,
        ubc_task = "roof_coarse",            # or "use_coarse", "roof_fine"

        gate_thresh=0.8,
        gate_frames=60,
        yolo_conf=0.3,
        conviction_rise=0.15,
        conviction_decay=0.3,
        flicker_penalty=0.10,
        alert_conviction=0.60,
        clear_conviction=0.20,
    )

    pipeline = FireDetectionPipeline(cfg)
    pipeline.run()