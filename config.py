"""
Pipeline configuration — all tunable thresholds in one place.
"""

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # --- Paths ---
    ffirenet_model_path: str = "models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth"
    yolo_model_path: str = "models/26m_1280imgsz_200epochs/weights/best.pt"
    video_path: str = "sample_videos/fire2.mp4"

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

    # --- UBC building classification (manual trigger via 'U' key) ---
    # While the pipeline is running, pressing 'U' queries the UBC server for
    # building classification at the demo GPS. The server must be running at
    ubc_demo_lat: float = 22.44647           # Munich Altstadt (default demo location)
    ubc_demo_lon: float = 114.17627
    ubc_task: str = "roof_coarse"            # or "roof_coarse", "roof_fine"
    ubc_buffer_meters: float = 150.0
    ubc_img_size: int = 600
    ubc_score_threshold: float = 0.3