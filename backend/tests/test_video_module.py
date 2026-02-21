import torch
import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import DEVICE
from backend.video_service import analyze_video, detect_face, load_model

MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"

print("\n===== DEVICE TEST =====")
print("Device:", DEVICE)

print("\n===== MODEL TEST =====")
model,threshold = load_model()
model.to(DEVICE)
model.eval()

print(f"Threshold: {threshold}")

dummy = torch.randn(1,3,224,224).to(DEVICE)
output = model(dummy)
print("Model output shape:", output.shape)

print("\n===== MTCNN TEST =====")
img_path = str(MEDIA_DIR / "test.jpg")
img = cv2.imread(img_path)

if img is None:
    print(f"{img_path} not found.")
else:
    face = detect_face(img)
    print("Face detected:", face is not None)

print("\n===== FULL PIPELINE TEST =====")
video_path = str(MEDIA_DIR / "sample4.mp4")
result = analyze_video(video_path)
print("Final Result:", result)
