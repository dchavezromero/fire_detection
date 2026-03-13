from ultralytics import YOLO

#model = YOLO("runs/detect/v26n_640imgsz_100epochs/weights/best.pt") #v26n 640 imgsz / 100 epochs
#model = YOLO("runs/detect/v8m_640imgsz_100epochs/weights/best.pt") #v8m 640 imgsz / 100 epochs
# model = YOLO("runs/detect/trainRTX5080/best.pt") #v26m 1280 imgsz / 100 epochs
model = YOLO("runs/detect/trainRTX5080_new/best.pt") #v26m 640 imgsz / 200 epochs

results = model.predict(
    source="sample_videos/fire1.mp4",
    conf=0.3,
    show=True,
    save=False,
    device=0,
    stream=True,
)

for r in results:
    pass