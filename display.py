"""
On-screen overlay drawing for the pipeline.
"""

import cv2
from config import PipelineConfig

# BGR colors per class
CLASS_COLORS = {
    "fire":  (0, 0, 255),     # red
    "smoke": (200, 150, 0),   # teal/blue-ish
}
DEFAULT_COLOR = (255, 255, 255)


def draw_overlay(frame, info: dict, stats: dict, cfg: PipelineConfig, dt_ms: float):
    """Draw bounding boxes, alert banner, and score panel onto the frame."""
    h, w = frame.shape[:2]

    # YOLO bounding boxes — colored by class
    for det in info["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cls_name = det.get("class_name", "unknown")
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # Alert banner
    if info["alert"]:
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.putText(frame, "FIRE ALERT", (w // 2 - 120, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # Detection counts
    fire_count = sum(1 for d in info["detections"] if d.get("class_name") == "fire")
    smoke_count = sum(1 for d in info["detections"] if d.get("class_name") == "smoke")

    # Scores panel (bottom-left)
    panel_y = h - 180
    cv2.rectangle(frame, (0, panel_y), (380, h), (0, 0, 0), -1)

    gate_status = f"OPEN ({info['gate_remaining']})" if info["gate_open"] else "CLOSED"

    lines = [
        f"FFireNet:   {info['ffirenet_score']:.2f}",
        f"Gate:       {gate_status}",
        f"YOLO:       {info['yolo_score']:.2f}" if info["yolo_ran"] else "YOLO:       --",
        f"Fire:       {fire_count}  Smoke: {smoke_count}" if info["yolo_ran"] else "Fire:       --  Smoke: --",
        f"Conviction: {info['conviction']:.2f} (streak {info['streak']})",
        f"State:      {'FIRE' if info['alert'] else 'CLEAR'}",
        f"{dt_ms:.0f}ms | YOLO calls: {stats['yolo_runs']}",
    ]

    panel_color = (0, 255, 0) if not info["alert"] else (0, 0, 255)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, panel_y + 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, panel_color, 1, cv2.LINE_AA)

    return frame