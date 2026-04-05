"""
FFireNet: Video Inference Script (PyTorch)
===========================================
Load a trained FFireNet model and run inference on a video file.
Displays each frame with fire/nofire prediction overlay.

Usage:
    python ffirenet_test.py
"""

import cv2
import time
import torch
import torch.nn as nn
from torchvision import transforms, models


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/mobilenet_v2_640imgsz_100epochs_0.01lr/ffirenet.pth"  # TODO: set your model path
VIDEO_PATH = "sample_videos/fire7.mp4"                        # TODO: set your video path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model Definition (must match training)
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
# Preprocessing (must match training)
# ============================================================
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# Main
# ============================================================
def main():
    # Load model
    model = FFireNet().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[INFO] Model loaded from {MODEL_PATH}")
    print(f"[INFO] Device: {DEVICE}")

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {VIDEO_PATH}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {VIDEO_PATH}")
    print(f"[INFO] {total_frames} frames @ {fps_video:.1f} FPS")
    print("[INFO] Press 'q' to quit\n")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.perf_counter()

        # Preprocess: BGR (OpenCV) -> RGB -> tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(frame_rgb).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logit = model(img_tensor).squeeze()
            prob = torch.sigmoid(logit).item()

        dt = (time.perf_counter() - t0) * 1000  # ms

        # Class mapping: fire=0, nofire=1 (alphabetical)
        # sigmoid close to 0 = fire, close to 1 = nofire
        if prob < 0.5:
            label = "FIRE DETECTED"
            confidence = (1 - prob) * 100
            color = (0, 0, 255)      # red in BGR
        else:
            label = "NO FIRE"
            confidence = prob * 100
            color = (0, 255, 0)      # green in BGR

        # Draw overlay
        text = f"{label} ({confidence:.1f}%)"
        fps_text = f"{1000/dt:.0f} FPS ({dt:.1f}ms)"

        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame {frame_count}/{total_frames}", (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Show
        cv2.imshow("FFireNet - Fire Detection", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Processed {frame_count} frames")


if __name__ == "__main__":
    main()