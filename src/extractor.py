import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models_loader import *

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def crop_expand(frame, box, expand=0.1):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)

    dx = int((x2 - x1) * expand)
    dy = int((y2 - y1) * expand)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    return frame[y1:y2, x1:x2]


def get_face_embedding(face_img):

    if face_img is None or face_img.size == 0:
        return None

    img = Image.fromarray(face_img)
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = facenet(tensor)

    emb = emb.cpu().numpy()

    if np.isnan(emb).any():
        return None

    return emb.reshape(1, -1).astype(np.float32)


def get_clothes_embedding(clothes_img):

    if clothes_img is None or clothes_img.size == 0:
        return None

    clothes_img = cv2.resize(clothes_img, (128, 256))
    clothes_img = clothes_img.astype(np.float32) / 255.0

    emb = clothes_img.flatten()

    if emb.size == 0 or np.isnan(emb).any():
        return None

    return emb.reshape(1, -1).astype(np.float32)


def detect_face(frame):
    results = face_yolo(frame, conf=0.5)[0]
    if len(results.boxes) == 0:
        return None, None

    box = results.boxes.xyxy[0].cpu().numpy()
    cls = int(results.boxes.cls[0].item())

    crop = crop_expand(frame, box)
    emb = get_face_embedding(crop)

    return emb, cls


def detect_clothes(frame):
    results = clothes_yolo(frame, conf=0.5)[0]
    if len(results.boxes) == 0:
        return None

    box = results.boxes.xyxy[0].cpu().numpy()
    crop = crop_expand(frame, box)

    return get_clothes_embedding(crop)


def detect_helmet(frame):
    results = helmet_model(frame, conf=0.5)[0]
    if len(results.boxes) == 0:
        return "no"
    return "yes"


def detect_helmet_color(frame):
    results = helmet_color_model(frame, conf=0.5)[0]
    if len(results.boxes) == 0:
        return "unknown"

    cls = int(results.boxes.cls[0].item())
    return helmet_color_model.names[cls]


def run_plate_demo():
    return "51A-12345"


def extract_all(frame):

    face_emb, _ = detect_face(frame)
    clothes_emb = detect_clothes(frame)
    helmet = detect_helmet(frame)
    helmet_color = detect_helmet_color(frame)
    plate = run_plate_demo()

    if face_emb is None:
        return None

    return {
        "face_emb": face_emb,
        "clothes_emb": clothes_emb,
        "helmet": helmet,
        "helmet_color": helmet_color,
        "plate": plate
    }