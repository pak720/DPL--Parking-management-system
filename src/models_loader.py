import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# ===============================
# DEVICE CONFIG
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================
# FACE EMBEDDING MODEL (FaceNet)
# ===============================

print("Loading FaceNet...")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# ===============================
# YOLO MODELS
# ===============================

print("Loading YOLO models...")

# Face detection model
face_yolo = YOLO("models/face_verify.pt")

# Clothes detection model
clothes_yolo = YOLO("models/clothes.pt")

# Helmet detection model
helmet_model = YOLO("models/helmet_yesno.pt")

# Helmet color detection model
helmet_color_model = YOLO("models/helmet_color.pt")

print("All models loaded successfully!")