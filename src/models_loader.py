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
    print("✓ LP Detector loaded")
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
    print("✓ LP OCR loaded")
except Exception as e:
    print(f"⚠ Warning: Could not load LP OCR: {e}")
    print("  → Check src/config.py -> YOLOV5_PATH, LP_OCR_MODEL")
    lp_ocr = None

print("All models loaded successfully!")