"""
Conviction tracker — drives fire alerts based on YOLO detections.
"""

import numpy as np
from config import PipelineConfig


class ConvictionTracker:
    """
    Builds conviction based on YOLO detection results while the gate is open.

    - YOLO finding fire raises conviction (accelerates with streak length).
    - YOLO finding nothing decays conviction.
    - State transitions (fire<->nofire) apply a flicker penalty.
    - Hysteresis: alert triggers at alert_conviction, clears at clear_conviction.

    Conviction is clamped to [0.0, 1.0].
    """

    def __init__(self, cfg: PipelineConfig):
        self.rise = cfg.conviction_rise
        self.decay = cfg.conviction_decay
        self.flicker_penalty = cfg.flicker_penalty
        self.alert_level = cfg.alert_conviction
        self.clear_level = cfg.clear_conviction

        self.conviction = 0.0
        self.alert = False
        self.prev_has_fire: bool | None = None
        self.streak = 0

    def update(self, yolo_score: float, has_detections: bool) -> tuple[bool, float]:
        """
        Push a YOLO result. Returns (fire_alert, conviction).

        Args:
            yolo_score: Combined YOLO fire score [0, 1].
            has_detections: Whether YOLO found any fire bounding boxes.
        """
        # Detect state transition -> flicker penalty
        if self.prev_has_fire is not None and has_detections != self.prev_has_fire:
            self.conviction = max(0.0, self.conviction - self.flicker_penalty)
            self.streak = 0

        self.streak += 1

        if has_detections:
            streak_bonus = 1.0 + 0.1 * min(self.streak, 10)
            self.conviction += self.rise * streak_bonus * yolo_score
        else:
            self.conviction -= self.decay * (1.0 + (1.0 - self.conviction) * 0.5)

        self.conviction = float(np.clip(self.conviction, 0.0, 1.0))

        # Hysteresis
        if not self.alert and self.conviction >= self.alert_level:
            self.alert = True
        elif self.alert and self.conviction <= self.clear_level:
            self.alert = False

        self.prev_has_fire = has_detections

        return self.alert, self.conviction

    def reset(self):
        self.conviction = 0.0
        self.alert = False
        self.prev_has_fire = None
        self.streak = 0