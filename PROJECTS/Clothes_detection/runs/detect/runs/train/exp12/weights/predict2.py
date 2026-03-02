from ultralytics import YOLO

# Load model đã train
model = YOLO(r"D:\DPL_PROJECT\clothes classfy.v1i.yolov11\runs\detect\runs\train\exp12\weights\best.pt")

# Predict một ảnh
results = model.predict(
    source=r"D:\DPL_PROJECT\clothes classfy.v1i.yolov11\runs\detect\runs\train\exp12\weights\89d9202ca81e5cf7e80a37555b62b128.jpg",
    conf=0.25,      # confidence threshold
    save=True,     # lưu ảnh kết quả
    show=True      # hiện ảnh ra màn hình
)

print("Done predicting!")
