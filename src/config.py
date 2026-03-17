# Configuration file for camera setup

# ===============================
# CAMERA CONFIGURATION
# ===============================

# Camera for person detection (face, clothes, helmet)
# Use 0 for laptop webcam
PERSON_CAMERA = 1

# Camera for license plate detection
# Option 1: If using DroidCam, set to 1 or 2 (depending on your system)
# Option 2: If using IP Webcam app, set to "http://YOUR_IP:8080/video"
# Example: PLATE_CAMERA = "http://192.168.1.100:8080/video"

PLATE_CAMERA = 0  # Change this to your phone camera index or IP

# ===============================
# DATABASE CONFIGURATION
# ===============================

DB_PATH = "parking_db.csv"

# ===============================
# LICENSE PLATE RECOGNITION CONFIG
# ===============================

# Path to yolov5 source OR yolov5 model (.pt)
YOLOV5_PATH = "../License-Plate-Recognition/yolov5"

# Model paths
LP_DETECTOR_MODEL = "../License-Plate-Recognition/model/LP_detector_nano_61.pt"
LP_OCR_MODEL = "../License-Plate-Recognition/model/LP_ocr_nano_62.pt"

# Confidence threshold for LP detector
LP_DETECTOR_CONFIDENCE = 0.20

# Confidence threshold for LP OCR
LP_OCR_CONFIDENCE = 0.50
