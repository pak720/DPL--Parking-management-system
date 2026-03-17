import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"D:\DPL_PROJECT\clothes classfy.v1i.yolov11\runs\detect\runs\train\exp12\weights\best.pt")  # change path if needed

# Open laptop camera (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📷 Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO prediction
    results = model.predict(frame, conf=0.25, imgsz=640)

    # Draw detections on frame
    annotated_frame = results[0].plot()

    # Show window
    cv2.imshow("YOLOv11 Webcam Detection", annotated_frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
