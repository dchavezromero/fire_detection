import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    # YOLO v8m
    # model = YOLO("yolov8m.pt")  # downloads pretrained medium model automatically
    # model.train(data="training_datasets/Fire And Smoke 5.v1i.yolov8/data.yaml", epochs=50, imgsz=640, device=0)

    # # YOLO v26n
    # model = YOLO("yolo26n.pt")  # downloads pretrained nano model automatically
    # model.train(data="training_datasets/Fire And Smoke 5.v1i.yolo26/data.yaml", epochs=5, imgsz=640, device=0)

    # # YOLO v26n
    # model = YOLO("yolo26n.pt")  # downloads pretrained nano model automatically
    # model.train(data="training_datasets/Fire And Smoke 5.v1i.yolo26/data.yaml", epochs=5, imgsz=640, device=0)

    # YOLO v26l
    model = YOLO("yolov8m.pt")  # downloads pretrained large model automatically

    model.train(
    data="training_datasets/Fire And Smoke 5.v1i.yolov8/data.yaml",
    epochs=200,
    patience=50,
    imgsz=640,
    batch=40,
    device=0
    )


    model.val() # run validation pass | prints mAP, precision, recall
    # --- Post-training cleanup ---
    detect_dir = Path("runs/detect")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Find the latest trainN folder
    train_dirs = sorted(detect_dir.glob("train*"), key=lambda p: p.stat().st_mtime)
    latest_train = train_dirs[-1]

    # Find the latest valN folder (if it exists)
    val_dirs = sorted(detect_dir.glob("val*"), key=lambda p: p.stat().st_mtime)
    if val_dirs:
        shutil.move(str(val_dirs[-1]), str(latest_train / "val"))

    # Read args.yaml to build folder name
    args_path = latest_train / "args.yaml"
    with open(args_path) as f:
        args = yaml.safe_load(f)

    model_name = Path(args["model"]).stem          # e.g. "yolov26m" or "yolov8m"
    version = model_name.replace("yolo", "")       # e.g. "26m" or "v8m"
    imgsz = args["imgsz"]
    epochs = args["epochs"]
    folder_name = f"{version}_{imgsz}imgsz_{epochs}epochs"

    # Handle duplicate names
    dest = models_dir / folder_name
    counter = 2
    while dest.exists():
        dest = models_dir / f"{folder_name}_{counter}"
        counter += 1

    shutil.move(str(latest_train), str(dest))

    print(f"Training results moved to: {dest}")

if __name__ == '__main__':
    main()