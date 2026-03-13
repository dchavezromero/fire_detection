from ultralytics import YOLO

# YOLO v8m
# model = YOLO("yolov8m.pt")  # downloads pretrained nano model automatically
# model.train(data="training_datasets/Fire And Smoke 5.v1i.yolov8/data.yaml", epochs=50, imgsz=640, device=0)

# YOLO v26m
model = YOLO("yolov26m.pt")  # downloads pretrained nano model automatically
model.train(data="training_datasets/Fire And Smoke 5.v1i.yolov26/data.yaml", epochs=200, imgsz=640, device=0)
model.val()  # prints mAP, precision, recall