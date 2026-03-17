import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(r"D:\DPL_PROJECT\PROJECTS\R1\kaggle\working\runs\detect\runs\train\exp1\weights\best.pt")

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

    # Improve image slightly
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

    # 🔥 Run detection (use CPU for now if CUDA broken)
    results = model(frame, conf=0.35, device="cpu")

    # Draw boxes
    annotated_frame = results[0].plot()

    # 🔥 SHOW FRAME (THIS WAS MISSING)
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()