"""
On-screen overlay drawing for the pipeline.
"""

import cv2
from config import PipelineConfig


def draw_overlay(frame, info: dict, stats: dict, cfg: PipelineConfig, dt_ms: float):
    """Draw bounding boxes, alert banner, and score panel onto the frame."""
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
    panel_y = h - 160
    cv2.rectangle(frame, (0, panel_y), (380, h), (0, 0, 0), -1)

    gate_status = f"OPEN ({info['gate_remaining']})" if info["gate_open"] else "CLOSED"

    lines = [
        f"FFireNet:   {info['ffirenet_score']:.2f}",
        f"Gate:       {gate_status}",
        f"YOLO:       {info['yolo_score']:.2f}" if info["yolo_ran"] else "YOLO:       --",
        f"Conviction: {info['conviction']:.2f} (streak {info['streak']})",
        f"State:      {'FIRE' if info['alert'] else 'CLEAR'}",
        f"{dt_ms:.0f}ms | YOLO calls: {stats['yolo_runs']}",
    ]

    color = (0, 255, 0) if not info["alert"] else (0, 0, 255)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, panel_y + 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame