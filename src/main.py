import cv2
import os
import csv
import numpy as np
import time
from collections import Counter, deque
from extractor import extract_all, extract_license_plate
from matcher import find_match, verify_checkout
from database import *
from config import PERSON_CAMERA, PLATE_CAMERA, DB_PATH

os.makedirs("captures", exist_ok=True)
init_db()

print("=== HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE - 2 CAMERA ===")
print("Camera 1: Thân người (Face, Clothes, Helmet)")
print("Camera 2: Biển số xe (License Plate)")
print("=" * 50)
print("Nhấn 'i' để CHECK-IN")
print("Nhấn 'o' để CHECK-OUT")
print("Nhấn 'q' để THOÁT")
print("=" * 50)


def optimize_camera(cap, width=None, height=None):
    # Giảm buffer để bớt trễ
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Set resolution nếu cung cấp
    if width is not None and height is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def get_camera_info(cap):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    return w, h, cam_fps


def enhance_image_quality(frame):
    """
    Tối ưu hóa chất lượng ảnh bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


# Khởi tạo ticket counter
ticket_counter = 1
if os.path.exists("parking_db.csv"):
    with open("parking_db.csv") as f:
        rows = list(csv.reader(f))
        if len(rows) > 1:
            ticket_counter = int(rows[-1][0]) + 1

# Mở 2 cameras
print("\n>>> Đang mở camera...")
cap_person = cv2.VideoCapture(PERSON_CAMERA)
cap_plate = cv2.VideoCapture(PLATE_CAMERA)
# Set độ phân giải cao cho Camera 1 (1920x1080) và giữ native cho Camera 2
optimize_camera(cap_person, width=1920, height=1080)
optimize_camera(cap_plate)

# Ưu tiên tự lấy nét / phơi sáng cho camera biển số
cap_plate.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap_plate.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Kiểm tra kết nối camera
if not cap_person.isOpened():
    print("❌ Không thể mở camera thân người (PERSON_CAMERA)")
    exit()

if not cap_plate.isOpened():
    print("⚠ Cảnh báo: Không thể mở camera biển số xe")
    print("   Hệ thống sẽ chạy với 1 camera (demo mode)")
    use_dual_camera = False
    person_w, person_h, person_cam_fps = get_camera_info(cap_person)
    person_res_text = f"{person_w}x{person_h}"
    print(f"ℹ Camera 1 native: {person_res_text} | FPS báo cáo: {person_cam_fps:.1f}")
else:
    use_dual_camera = True
    print("✅ Đã kết nối 2 camera thành công!")
    person_w, person_h, person_cam_fps = get_camera_info(cap_person)
    plate_w, plate_h, plate_cam_fps = get_camera_info(cap_plate)
    person_res_text = f"{person_w}x{person_h}"
    plate_res_text = f"{plate_w}x{plate_h}"
    print(f"ℹ Camera 1 native: {person_res_text} | FPS báo cáo: {person_cam_fps:.1f}")
    print(f"ℹ Camera 2 native: {plate_res_text} | FPS báo cáo: {plate_cam_fps:.1f}")

# Biến lưu biển số xe hiện tại
current_plate = "unknown"
frame_plate = None
plate_history = deque(maxlen=7)

# Chạy OCR liên tục để tối đa hóa chất lượng nhận diện
plate_ocr_every_n_frames = 1
plate_frame_idx = 0

# FPS monitor
prev_time = time.time()
fps = 0.0

while True:
    # Đọc frame từ camera thân người
    ret1, frame_person = cap_person.read()
    if not ret1:
        print("Không thể đọc frame từ camera thân người")
        break
    
    # Đọc frame từ camera biển số (nếu có)
    if use_dual_camera:
        ret2, frame_plate = cap_plate.read()
        if ret2:
            # Tự động detect biển số, nhưng giảm tần suất để tăng FPS
            plate_frame_idx += 1
            if plate_frame_idx % plate_ocr_every_n_frames == 0:
                plate_candidate = extract_license_plate(frame_plate)
                if plate_candidate != "unknown":
                    plate_history.append(plate_candidate)
                
                if len(plate_history) > 0:
                    current_plate = Counter(plate_history).most_common(1)[0][0]
                else:
                    current_plate = "unknown"
            
            # Hiển thị biển số đã detect trên frame
            plate_color = (0, 255, 0) if current_plate != "unknown" else (0, 0, 255)
            cv2.putText(frame_plate, f"Plate: {current_plate}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plate_color, 2)
            cv2.putText(frame_plate, f"License Plate Camera ({plate_res_text})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_plate, f"OCR vote window: {len(plate_history)}/7", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame_plate, f"FPS: {fps:.1f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Resize cửa sổ để nhỏ hơn (vẫn giữ resolution capture)
            display_plate = cv2.resize(frame_plate, (800, 540))
            cv2.imshow("Camera 2 - License Plate", display_plate)
    
    # Hiển thị hướng dẫn trên frame thân người
    cv2.putText(frame_person, f"Camera 1 - Person Detection ({person_res_text})", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame_person, "Press: [i]=Check-in | [o]=Check-out | [q]=Quit", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_person, f"FPS: {fps:.1f} | Quality: Enhanced", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Resize cửa sổ để nhỏ hơn (vẫn giữ resolution capture)
    display_person = cv2.resize(frame_person, (800, 540))
    cv2.imshow("Camera 1 - Person Detection", display_person)
    
    key = cv2.waitKey(1)

    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 1.0 / dt
    prev_time = now

    # CHECK-IN
    if key == ord('i'):
        print("\n" + "="*50)
        print(">>> Đang xử lý CHECK-IN...")

        # Chụp lại frame mới từ cả 2 camera tại thời điểm nhấn nút
        ret_person_snap, person_frame_snap = cap_person.read()
        if not ret_person_snap:
            print("❌ Không đọc được camera thân người")
            continue
        
        # Cải thiện chất lượng ảnh trước xử lý
        person_frame_snap = enhance_image_quality(person_frame_snap)

        plate_frame_snap = None
        plate_from_phone = "unknown"
        if use_dual_camera:
            ret_plate_snap, plate_frame_snap = cap_plate.read()
            if ret_plate_snap:
                plate_from_phone = extract_license_plate(plate_frame_snap)
        
        # Extract data từ camera thân người
        data = extract_all(person_frame_snap)
        if data is None:
            print("❌ Không detect được mặt")
            continue
        
        # Xác thực biển số (rất quan trọng cho check-in)
        if use_dual_camera:
            if plate_from_phone != "unknown":
                data["plate"] = plate_from_phone
                print(f"✅ Biển số từ camera 2 (realtime): {plate_from_phone}")
            elif current_plate != "unknown":
                data["plate"] = current_plate
                print(f"⚠ Biển số từ stream (có thể cũ): {current_plate}")
            else:
                print("❌ Không detect được biển số xe")
                continue
        else:
            print(f"ℹ Demo mode: Sử dụng biển số mặc định: {data['plate']}")

        # Xác thực mũ bảo hiểm
        if data["helmet"] == "no":
            print("⚠ Cảnh báo: Người này KHÔNG đội mũ bảo hiểm")
        else:
            print(f"✓ Mũ bảo hiểm: {data['helmet']} ({data['helmet_color']})")

        # Lưu ảnh từ cả 2 camera
        image_path_person = f"captures/in_{ticket_counter}_person.jpg"
        cv2.imwrite(image_path_person, person_frame_snap)
        
        if use_dual_camera and plate_frame_snap is not None:
            image_path_plate = f"captures/in_{ticket_counter}_plate.jpg"
            cv2.imwrite(image_path_plate, plate_frame_snap)

        add_checkin(
            ticket_counter,
            data["plate"],
            data["helmet"],
            data["helmet_color"],
            data["face_emb"],
            data["clothes_emb"],
            image_path_person
        )

        print(f"✅ Check-in thành công! ID = {ticket_counter}")
        print(f"   - Biển số: {data['plate']}")
        print(f"   - Mũ bảo hiểm: {data['helmet']} ({data['helmet_color']})")
        print("="*50)
        ticket_counter += 1

    # CHECK-OUT
    elif key == ord('o'):
        print("\n" + "="*50)
        print(">>> Đang xử lý CHECK-OUT...")

        # Chụp lại frame mới từ cả 2 camera tại thời điểm nhấn nút
        ret_person_snap, person_frame_snap = cap_person.read()
        if not ret_person_snap:
            print("❌ Không đọc được camera thân người")
            continue

        # Cải thiện chất lượng ảnh trước xử lý
        person_frame_snap = enhance_image_quality(person_frame_snap)

        plate_frame_snap = None
        plate_from_phone = "unknown"
        if use_dual_camera:
            ret_plate_snap, plate_frame_snap = cap_plate.read()
            if ret_plate_snap:
                plate_from_phone = extract_license_plate(plate_frame_snap)
        
        # Extract data từ camera thân người
        data = extract_all(person_frame_snap)
        if data is None:
            print("❌ Không detect được mặt")
            continue
        
        # Xác thực biển số (rất quan trọng cho check-out matching)
        if use_dual_camera:
            if plate_from_phone != "unknown":
                data["plate"] = plate_from_phone
                print(f"✅ Biển số từ camera 2 (realtime): {plate_from_phone}")
            elif current_plate != "unknown":
                data["plate"] = current_plate
                print(f"⚠ Biển số từ stream (có thể cũ): {current_plate}")
            else:
                print("❌ Không detect được biển số xe")
                continue
        else:
            print(f"ℹ Demo mode: Sử dụng biển số mặc định: {data['plate']}")

        # Xác thực mũ bảo hiểm
        if data["helmet"] == "no":
            print("⚠ Cảnh báo: Người này KHÔNG đội mũ bảo hiểm")
        else:
            print(f"✓ Mũ bảo hiểm: {data['helmet']} ({data['helmet_color']})")

        # Load database mỗi lần check-out để có data mới nhất
        database_embeddings = load_active_tickets()
        
        if len(database_embeddings) == 0:
            print("❌ Không có vé nào đang hoạt động trong hệ thống")
            continue

        query = {
            "face": data["face_emb"],
            "clothes": data["clothes_emb"],
            "helmet": data["helmet"],
            "plate": data["plate"]
        }

        print("\n📋 Đang so khớp với dữ liệu check-in...")
        match_id = find_match(query, database_embeddings)

        if match_id is None:
            print("❌ Không tìm thấy vé phù hợp")
            print("   (Có thể người/xe khác hoặc dữ liệu check-in không chính xác)")
            print("="*50)
            continue

        # **NEW: Xác thực dữ liệu check-in vs check-out**
        print("\n🔄 Bước 2: Xác thực dữ liệu check-in vs check-out...")
        
        # Tải dữ liệu check-in từ database
        with open(DB_PATH, mode="r") as f:
            reader = csv.DictReader(f)
            checkin_data = None
            for row in reader:
                if row["status"] == "ACTIVE" and int(row["ticket_id"]) == match_id:
                    checkin_data = {
                        "face": np.load(row["face_embedding_path"]),
                        "clothes": np.load(row["clothes_embedding_path"]) if row["clothes_embedding_path"] != "" else None,
                        "helmet": row["helmet"],
                        "plate": row["plate"]
                    }
                    break
        
        if checkin_data is None:
            print(f"❌ Không tìm thấy dữ liệu check-in cho ticket {match_id}")
            print("="*50)
            continue
        
        # Xác thực checkout
        is_valid, verify_score, verify_message = verify_checkout(checkin_data, query)
        print(verify_message)
        
        if not is_valid:
            print(f"❌ XÁC THỰC THẤT BẠI - Từ chối checkout")
            print("="*50)
            continue

        print(f"✅ XÁC THỰC THÀNH CÔNG (Score: {verify_score:.3f})")

        # Lưu ảnh từ cả 2 camera
        image_path_person = f"captures/out_{match_id}_person.jpg"
        cv2.imwrite(image_path_person, person_frame_snap)
        
        if use_dual_camera and plate_frame_snap is not None:
            image_path_plate = f"captures/out_{match_id}_plate.jpg"
            cv2.imwrite(image_path_plate, plate_frame_snap)

        update_checkout(match_id, image_path_person)

        print(f"✅ Check-out thành công! ID = {match_id}")
        print("="*50)

    # THOÁT
    elif key == ord('q'):
        print("\n>>> Đang thoát hệ thống...")
        break

cap_person.release()
if use_dual_camera:
    cap_plate.release()
cv2.destroyAllWindows()
print("Hệ thống đã đóng!")