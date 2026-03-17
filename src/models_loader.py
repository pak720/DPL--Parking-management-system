import torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
import sys
import os
from config import YOLOV5_PATH, LP_DETECTOR_MODEL, LP_OCR_MODEL, LP_DETECTOR_CONFIDENCE, LP_OCR_CONFIDENCE

# ===============================
# DEVICE CONFIG
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_half = torch.cuda.is_available()  # FP16 chỉ khi có GPU — tăng tốc ~2x trên NVIDIA
print(f"Using device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | VRAM: {props.total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True  # tự chọn kernel tối ưu cho input size cố định
else:
    print("GPU not found — running on CPU")

# ===============================
# FACE EMBEDDING MODEL (FaceNet)
# ===============================

print("Loading FaceNet...")
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
if use_half:
    facenet = facenet.half()
    print("✓ FaceNet → GPU (FP16)")

# ===============================
# YOLO MODELS
# ===============================

print("Loading YOLO models...")
_yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Face detection model
face_yolo = YOLO("src/models/best_face.pt")
face_yolo.to(_yolo_device)

# Clothes detection model
clothes_yolo = YOLO("src/models/clothes.pt")
clothes_yolo.to(_yolo_device)

# Helmet detection model
helmet_model = YOLO("src/models/helmet_yesno.pt")
helmet_model.to(_yolo_device)

# Helmet color detection model
helmet_color_model = YOLO("src/models/helmet_color.pt")
helmet_color_model.to(_yolo_device)

print(f"✓ YOLO models → {_yolo_device}")

# ===============================
# LICENSE PLATE MODELS
# ===============================

print("Loading License Plate Recognition models...")

# Resolve absolute paths from src/
base_dir = os.path.dirname(__file__)
yolov5_path = os.path.abspath(os.path.join(base_dir, YOLOV5_PATH))
lp_detector_path = os.path.abspath(os.path.join(base_dir, LP_DETECTOR_MODEL))
lp_ocr_path = os.path.abspath(os.path.join(base_dir, LP_OCR_MODEL))

# Use local full YOLOv5 source (already cloned in workspace)
hub_repo = None
hub_source = 'local'

if not os.path.isdir(yolov5_path):
    print(f"⚠ Warning: YOLOV5_PATH is not a directory: {yolov5_path}")
else:
    hubconf_path = os.path.join(yolov5_path, "hubconf.py")
    if not os.path.exists(hubconf_path):
        print(f"⚠ Warning: Missing hubconf.py in YOLOV5_PATH: {hubconf_path}")
    else:
        hub_repo = yolov5_path
        if yolov5_path not in sys.path:
            sys.path.insert(0, yolov5_path)
        print(f"Using local YOLOv5 source: {yolov5_path}")

# License plate detector
try:
    if hub_repo is None:
        raise RuntimeError("Local YOLOv5 source is not available")
    if not os.path.exists(lp_detector_path):
        raise FileNotFoundError(f"LP detector model not found: {lp_detector_path}")

    lp_detector = torch.hub.load(
        hub_repo,
        'custom', 
        path=lp_detector_path,
        force_reload=False, 
        source=hub_source
    )
    lp_detector.conf = LP_DETECTOR_CONFIDENCE
    lp_detector.to(device)
    if use_half:
        lp_detector.half()
    print(f"✓ LP Detector loaded → {device}{' (FP16)' if use_half else ''}")
except Exception as e:
    print(f"⚠ Warning: Could not load LP detector: {e}")
    print("  → Check src/config.py -> YOLOV5_PATH, LP_DETECTOR_MODEL")
    lp_detector = None

# License plate OCR
try:
    if hub_repo is None:
        raise RuntimeError("Local YOLOv5 source is not available")
    if not os.path.exists(lp_ocr_path):
        raise FileNotFoundError(f"LP OCR model not found: {lp_ocr_path}")
    
    lp_ocr = torch.hub.load(
        hub_repo,
        'custom', 
        path=lp_ocr_path,
        force_reload=False, 
        source=hub_source
    )
    lp_ocr.conf = LP_OCR_CONFIDENCE
    lp_ocr.to(device)
    if use_half:
        lp_ocr.half()
    print(f"✓ LP OCR loaded → {device}{' (FP16)' if use_half else ''}")
except Exception as e:
    print(f"⚠ Warning: Could not load LP OCR: {e}")
    print("  → Check src/config.py -> YOLOV5_PATH, LP_OCR_MODEL")
    lp_ocr = None

print("All models loaded successfully!")