"""
Pipeline configuration — all tunable thresholds in one place.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- Paths ---
    ffirenet_model_path: str = "models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth"
    yolo_model_path: str = "models/26m_1280imgsz_200epochs/weights/best.pt"
    video_path: str = "sample_videos/fire3.mp4"

    # --- YOLO class mapping ---
    # Class IDs from training: 0 = smoke, 1 = fire
    class_names: dict = field(default_factory=lambda: {0: "smoke", 1: "fire"})

    # --- FFireNet gate ---
    # FFireNet sigmoid: 0.0 = fire, 1.0 = no fire
    # Any frame where sigmoid < gate_thresh trips the gate and activates YOLO.
    gate_thresh: float = 0.5                # FFireNet sigmoid below this = possible fire, open gate

    # --- Gate duration ---
    # Once FFireNet trips the gate, YOLO takes over for this many frames.
    # If YOLO keeps finding fire, the countdown resets each time.
    gate_frames: int = 30                   # how many frames YOLO stays active after gate trips

    # --- YOLO settings ---
    yolo_conf: float = 0.3                  # YOLO confidence threshold
    yolo_device: int = 0                    # GPU device (0) or "cpu"

    # --- Conviction tracker (driven by YOLO when gate is open) ---
    conviction_rise: float = 0.15           # conviction added per YOLO fire frame
    conviction_decay: float = 0.08          # conviction removed per YOLO no-fire frame
    flicker_penalty: float = 0.10           # conviction drop on each fire<->nofire transition
    alert_conviction: float = 0.60          # conviction level that triggers a fire alert
    clear_conviction: float = 0.20          # conviction level below which alert is cleared

    # --- Display ---
    show_display: bool = True
    save_output: bool = False
    output_path: str = "output_pipeline.mp4"