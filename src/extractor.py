import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models_loader import *
import sys
import os

# Add License-Plate-Recognition functions to path
lp_func_path = os.path.join(os.path.dirname(__file__), "../License-Plate-Recognition/function")
sys.path.insert(0, lp_func_path)

try:
    import helper
    import utils_rotate
    LP_MODULES_AVAILABLE = True
except:
    LP_MODULES_AVAILABLE = False
    print("⚠ Warning: LP helper modules not found, using demo mode")

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


def normalize_plate_text(text):
    if text is None:
        return "unknown"

    cleaned = "".join(ch for ch in str(text).upper() if ch.isalnum())
    if not (6 <= len(cleaned) <= 10):
        return "unknown"

    # Ưu tiên format prefix 4 ký tự như yêu cầu:
    # - xxAx-xxxxx (4-5)
    # - xxAx-xxxx  (4-4)
    if len(cleaned) >= 9 and cleaned[-5:].isdigit() and len(cleaned[:-5]) == 4:
        return f"{cleaned[:-5]}-{cleaned[-5:]}"

    if len(cleaned) >= 8 and cleaned[-4:].isdigit() and len(cleaned[:-4]) == 4:
        return f"{cleaned[:-4]}-{cleaned[-4:]}"

    # Fallback (giữ tương thích nếu OCR ra độ dài khác)
    if len(cleaned) >= 7 and cleaned[-5:].isdigit():
        return f"{cleaned[:-5]}-{cleaned[-5:]}"

    if len(cleaned) >= 6 and cleaned[-4:].isdigit():
        return f"{cleaned[:-4]}-{cleaned[-4:]}"

    return "unknown"


def build_plate_variants(frame):
    enhanced = enhance_plate_image(frame)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_otsu = cv2.cvtColor(th_otsu, cv2.COLOR_GRAY2BGR)

    return [frame, enhanced, th_otsu]


def enhance_plate_image(frame):
    """
    Tiền xử lý ảnh biển số để robust với rung camera, góc độ
    Cải thiện:
    - Contrast (dễ nhận diện ký tự)
    - Blur giảm noise (rung camera)
    - Sharpening để làm rõ ký tự
    """
    # 1. Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # 2. Giảm noise từ rung camera (bilateral blur)
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 3. Sharpening kernel để làm rõ ký tự
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]]) / 1.0
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


def read_plate_from_crop(crop_img):
    h, w = crop_img.shape[:2]
    if h < 8 or w < 16:
        return "unknown"

    # upscale giúp OCR ổn định hơn khi camera rung nhẹ
    scale = 2
    up = cv2.resize(crop_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # deskew theo nhiều hướng
    for cc in range(0, 2):
        for ct in range(0, 2):
            try_img = utils_rotate.deskew(up, cc, ct)
            lp = normalize_plate_text(helper.read_plate(lp_ocr, try_img))
            if lp != "unknown":
                return lp

    # fallback đọc trực tiếp
    lp = normalize_plate_text(helper.read_plate(lp_ocr, up))
    return lp


def detect_license_plate(frame):
    """
    Detect and recognize license plate from frame
    Returns: license plate text or "unknown"
    Robust với rung camera, góc độ, lighting
    """
    if lp_detector is None or lp_ocr is None or not LP_MODULES_AVAILABLE:
        return "unknown"
    
    try:
        variants = build_plate_variants(frame)

        # chạy detector nhiều size để tăng độ nhạy ở nhiều khoảng cách
        detect_sizes = [640, 960]

        for variant in variants:
            vh, vw = variant.shape[:2]

            for detect_size in detect_sizes:
                plates = lp_detector(variant, size=detect_size)
                list_plates = plates.pandas().xyxy[0].values.tolist()

                for plate in list_plates:
                    x1 = int(max(0, plate[0]))
                    y1 = int(max(0, plate[1]))
                    x2 = int(min(vw, plate[2]))
                    y2 = int(min(vh, plate[3]))

                    # padding để giữ đủ ký tự khi camera rung
                    pad_x = int((x2 - x1) * 0.15)
                    pad_y = int((y2 - y1) * 0.20)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(vw, x2 + pad_x)
                    y2 = min(vh, y2 + pad_y)

                    crop_img = variant[y1:y2, x1:x2]
                    lp = read_plate_from_crop(crop_img)
                    if lp != "unknown":
                        return lp

            # fallback OCR trực tiếp trên toàn frame variant
            lp = normalize_plate_text(helper.read_plate(lp_ocr, variant))
            if lp != "unknown":
                return lp

        return "unknown"
    
    except Exception as e:
        print(f"Error in license plate detection: {e}")
        return "unknown"


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


def extract_license_plate(frame):
    """
    Extract only license plate from frame (for separate camera)
    """
    plate = detect_license_plate(frame)
    return plate