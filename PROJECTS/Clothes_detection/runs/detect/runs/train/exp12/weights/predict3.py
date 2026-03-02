import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(r"D:\DPL_PROJECT\clothes classfy.v1i.yolov11\runs\detect\runs\train\exp12\weights\best.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📷 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Improve image quality slightly ---
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)  # brighter & clearer

    # Run prediction (better settings)
    results = model(
        frame,
        conf=0.35,          # higher = less false detections
        iou=0.5,            # better box filtering
        imgsz=640,
        device=0,          # GPU if available
        stream=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Improved YOLO Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
