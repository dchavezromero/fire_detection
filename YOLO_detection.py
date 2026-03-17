from ultralytics import YOLO

model = YOLO("models/26m_1280imgsz_200epochs/weights/best.pt") # choose model to use from models folders

results = model.predict(
    source="sample_videos/fire2.mp4",
    conf=0.3,
    show=True,
    save=False,
    device=0,
    stream=True,
)

for r in results:
    pass