import cv2
import os
import csv
import numpy as np
import time
import re
import threading
from collections import Counter, deque
from extractor import extract_all, extract_license_plate
from matcher import find_match, verify_checkout
from database import *
from config import PERSON_CAMERA, PLATE_CAMERA, DB_PATH


class CameraThread:
    """Thread đọc camera liên tục, tránh buffer delay làm lag display."""
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ret = False
        self._lock = threading.Lock()
        self._running = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self._lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self._running = False

os.makedirs("captures", exist_ok=True)
init_db()

print("=== HE THONG QUAN LY BAI DO XE - 2 CAMERA ===")
print("Camera 1: Thân người (Face, Clothes, Helmet)")
print("Camera 2: Biển số xe (License Plate)")
print("=" * 50)
print("Che do tu dong: vao bai = CHECK-IN | ra bai = CHECK-OUT")
print("Dieu kien: phai detect duoc ca mat + bien so")
print("Quy tac an toan: sau CHECK-IN phai cho 30 giay moi duoc CHECK-OUT")
print("Nhan 'q' de THOAT")
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


_clahe_main = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def enhance_image_quality(frame):
    """
    Tối ưu hóa chất lượng ảnh bằng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe_main.apply(l)  # dùng object cache
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


PLATE_FORMAT_REGEXES = [
    re.compile(r"^[A-Z0-9]{4}-\d{4,5}$"),      # dang hien tai (vd: 29T8-2843)
    re.compile(r"^\d{2}MD[A-Z0-9]-\d{5}$"),   # dang moi: xxMDx-xxxxx
]


def is_valid_plate_format(plate_text):
    if plate_text is None:
        return False
    plate_text = str(plate_text).upper().strip()
    return any(rx.fullmatch(plate_text) for rx in PLATE_FORMAT_REGEXES)


# Khởi tạo ticket counter
ticket_counter = 1
if os.path.exists("parking_db.csv"):
    with open("parking_db.csv") as f:
        rows = list(csv.reader(f))
        if len(rows) > 1:
            ticket_counter = int(rows[-1][0]) + 1

# Mở 2 cameras
print("\n>>> Dang mo camera...")
cap_person = cv2.VideoCapture(PERSON_CAMERA)
cap_plate = cv2.VideoCapture(PLATE_CAMERA)

# Hạ xuống 1280x720 để giảm lag — vẫn đủ chất lượng cho infer
optimize_camera(cap_person, width=1280, height=720)
optimize_camera(cap_plate)

# Ưu tiên tự lấy nét / phơi sáng cho camera biển số
cap_plate.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap_plate.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Kiểm tra kết nối camera
if not cap_person.isOpened():
    print("[ERROR] Khong the mo camera than nguoi (PERSON_CAMERA)")
    exit()

if not cap_plate.isOpened():
    print("[WARN] Khong the mo camera bien so xe")
    print("   He thong se chay voi 1 camera (demo mode)")
    use_dual_camera = False
    person_w, person_h, person_cam_fps = get_camera_info(cap_person)
    person_res_text = f"{person_w}x{person_h}"
    print(f"[INFO] Camera 1 native: {person_res_text} | FPS bao cao: {person_cam_fps:.1f}")
else:
    use_dual_camera = True
    print("[OK] Da ket noi 2 camera thanh cong!")
    person_w, person_h, person_cam_fps = get_camera_info(cap_person)
    plate_w, plate_h, plate_cam_fps = get_camera_info(cap_plate)
    person_res_text = f"{person_w}x{person_h}"
    plate_res_text = f"{plate_w}x{plate_h}"
    print(f"[INFO] Camera 1 native: {person_res_text} | FPS bao cao: {person_cam_fps:.1f}")
    print(f"[INFO] Camera 2 native: {plate_res_text} | FPS bao cao: {plate_cam_fps:.1f}")

# Khởi động camera threads để tránh buffer delay
cam_person_thread = CameraThread(cap_person)
cam_plate_thread = CameraThread(cap_plate) if use_dual_camera else None

# Biến lưu biển số xe hiện tại
current_plate = "unknown"
frame_plate = None
plate_history = deque(maxlen=5)

# **Tự động hóa check-in/check-out (không dùng countdown)**
last_action_time = 0.0
MIN_ACTION_GAP = 1.0  # chống trigger lặp liên tục
CHECKOUT_DELAY_AFTER_CHECKIN = 30.0  # chỉ cho checkout sau 30 giây kể từ check-in
last_checkin_by_plate = {}  # plate -> timestamp check-in gần nhất trong phiên chạy
status_text = "Cho detect mat + bien so..."
status_color = (0, 255, 0)
MIN_CHECKIN_GAP = 2.0  # delay giữa 2 lần check-in
STABLE_REQUIRED_SECONDS = 2.0  # phải đứng yên 2 giây mới thao tác
EVENT_MESSAGE_HOLD_SECONDS = 3  # giu thong bao CHECK-IN/CHECK-OUT tren man hinh
status_hold_until = 0.0
PERSON_INFER_SCALE = 0.7  # giam kich thuoc frame cho infer de tang toc
FACE_PLATE_LOCK_THRESHOLD = 0.78  # cung 1 khuon mat chi duoc gan 1 bien so

# Trạng thái ổn định người + biển số
stable_start_time = None
stable_plate = None
stable_face_emb = None
last_checkin_time = 0.0

# Giảm tần suất OCR — chạy mỗi 2 frame thay vì mỗi frame
plate_ocr_every_n_frames = 2
plate_frame_idx = 0

# Skip person inference mỗi 2 frame để giảm tải 4 YOLO models
PERSON_INFER_EVERY_N = 2
person_infer_idx = 0
last_person_data = None

# FPS monitor
prev_time = time.time()
fps = 0.0


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    try:
        va = np.asarray(a, dtype=np.float32).reshape(-1)
        vb = np.asarray(b, dtype=np.float32).reshape(-1)
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))
    except:
        return 0.0


def set_status(message, color, hold_seconds=0.0):
    global status_text, status_color, status_hold_until
    status_text = message
    status_color = color
    if hold_seconds > 0:
        status_hold_until = time.time() + hold_seconds


def has_face_plate_conflict(query_face_emb, query_plate, active_tickets):
    """
    Kiem tra rang buoc: moi khuon mat chi checkin/checkout voi 1 bien so.
    Neu thay 1 ticket ACTIVE co face tuong tu cao nhung bien so khac => conflict.
    """
    if query_face_emb is None:
        return None, 0.0

    best_conflict_plate = None
    best_sim = 0.0
    for _, stored in active_tickets.items():
        stored_plate = str(stored.get("plate", "")).upper().strip()
        if stored_plate == query_plate:
            continue
        sim = cosine_sim(query_face_emb, stored.get("face"))
        if sim > FACE_PLATE_LOCK_THRESHOLD and sim > best_sim:
            best_sim = sim
            best_conflict_plate = stored_plate

    return best_conflict_plate, best_sim

while True:
    # Đọc frame từ camera thread (không bị block bởi buffer)
    ret1, frame_person = cam_person_thread.read()
    if frame_person is None:
        # Thread chưa capture được frame đầu tiên — chờ tiếp
        continue
    if not ret1:
        print("Khong the doc frame tu camera than nguoi")
        break

    # Đọc frame từ camera biển số (nếu có)
    if use_dual_camera:
        ret2, frame_plate = cam_plate_thread.read()
        if ret2:
            plate_frame_idx += 1
            if plate_frame_idx % plate_ocr_every_n_frames == 0:
                plate_candidate = extract_license_plate(frame_plate)
                if plate_candidate != "unknown":
                    plate_history.append(plate_candidate)
                if len(plate_history) > 0:
                    current_plate = Counter(plate_history).most_common(1)[0][0]
                else:
                    current_plate = "unknown"

            plate_color = (0, 255, 0) if current_plate != "unknown" else (0, 0, 255)
            cv2.putText(frame_plate, f"Plate: {current_plate}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, plate_color, 2)
            cv2.putText(frame_plate, f"License Plate Camera ({plate_res_text})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_plate, f"OCR vote window: {len(plate_history)}/5",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame_plate, f"FPS: {fps:.1f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            display_plate = cv2.resize(frame_plate, (800, 540))
            cv2.imshow("Camera 2 - License Plate", display_plate)

    # Detect người từ camera 1 — chỉ infer mỗi PERSON_INFER_EVERY_N frame
    person_infer_idx += 1
    if person_infer_idx % PERSON_INFER_EVERY_N == 0:
        infer_person_frame = cv2.resize(
            frame_person,
            None,
            fx=PERSON_INFER_SCALE,
            fy=PERSON_INFER_SCALE,
            interpolation=cv2.INTER_LINEAR,
        )
        last_person_data = extract_all(infer_person_frame)
    person_data = last_person_data

    now = time.time()
    can_trigger_action = (now - last_action_time) >= MIN_ACTION_GAP
    can_overwrite_status = now >= status_hold_until

    # Điều kiện ổn định 5 giây: cùng biển số + cùng người (face similarity cao)
    stable_ready = False
    if person_data is not None and current_plate != "unknown" and is_valid_plate_format(current_plate):
        if stable_plate == current_plate and cosine_sim(stable_face_emb, person_data.get("face_emb")) >= 0.80:
            if stable_start_time is None:
                stable_start_time = now
            if (now - stable_start_time) >= STABLE_REQUIRED_SECONDS:
                stable_ready = True
                if can_overwrite_status:
                    set_status("OK: Nguoi + bien so on dinh, san sang xu ly", (0, 255, 0))
            else:
                if can_overwrite_status:
                    set_status("Cho dung yen de xac thuc", (0, 200, 255))
        else:
            stable_plate = current_plate
            stable_face_emb = person_data.get("face_emb")
            stable_start_time = now
            if can_overwrite_status:
                set_status("Bat dau theo doi on dinh nguoi + bien so", (0, 200, 255))
    else:
        stable_start_time = None
        stable_plate = None
        stable_face_emb = None

    # Chỉ thực hiện khi có đủ cả mặt + biển số đúng format
    if current_plate != "unknown" and not is_valid_plate_format(current_plate):
        if can_overwrite_status:
            set_status(
                f"Canh bao: Bien so sai format: {current_plate} | KHONG CHECK-IN/CHECK-OUT, vui long quet lai",
                (0, 0, 255),
                EVENT_MESSAGE_HOLD_SECONDS,
            )

        # Bat buoc lam lai: xoa trang thai on dinh va bo ket qua OCR sai
        stable_start_time = None
        stable_plate = None
        stable_face_emb = None
        plate_history.clear()
        current_plate = "unknown"

    if person_data is not None and current_plate != "unknown" and is_valid_plate_format(current_plate) and stable_ready and can_trigger_action:
        active_tickets = load_active_tickets()
        matched_ticket_id = None
        checkin_data = None

        for ticket_id, stored in active_tickets.items():
            stored_plate = stored.get("plate")
            if not is_valid_plate_format(stored_plate):
                continue
            if stored_plate == current_plate:
                matched_ticket_id = ticket_id
                checkin_data = stored
                break

        if matched_ticket_id is None:
            # NGƯỜI VÀO -> CHECK-IN
            # Rule: 1 khuon mat <-> 1 bien so, chi ap dung voi VE ACTIVE
            # Neu da CHECK-OUT thanh cong thi ticket khong con ACTIVE va duoc phep CHECK-IN bien so moi
            conflict_plate, conflict_sim = has_face_plate_conflict(
                person_data.get("face_emb"),
                current_plate,
                active_tickets,
            )
            if conflict_plate is not None:
                set_status(
                    f"Tu choi CHECK-IN: khuon mat dang gan voi bien so ACTIVE {conflict_plate} (sim={conflict_sim:.2f})",
                    (0, 0, 255),
                    EVENT_MESSAGE_HOLD_SECONDS,
                )
                last_action_time = now
                continue

            if (now - last_checkin_time) < MIN_CHECKIN_GAP:
                wait_checkin = int(MIN_CHECKIN_GAP - (now - last_checkin_time))
                if can_overwrite_status:
                    set_status(f"Vui long cho {wait_checkin}s truoc khi CHECK-IN tiep", (0, 140, 255))
                continue

            image_path_person = f"captures/in_{ticket_counter}_person.jpg"
            cv2.imwrite(image_path_person, frame_person)

            if use_dual_camera and frame_plate is not None:
                image_path_plate = f"captures/in_{ticket_counter}_plate.jpg"
                cv2.imwrite(image_path_plate, frame_plate)

            add_checkin(
                ticket_counter,
                current_plate,
                person_data["helmet"],
                person_data["helmet_color"],
                person_data["face_emb"],
                person_data["clothes_emb"],
                image_path_person
            )

            last_checkin_by_plate[current_plate] = now
            last_checkin_time = now
            set_status(
                f"CHECK-IN OK | ID {ticket_counter} | {current_plate}",
                (0, 255, 0),
                EVENT_MESSAGE_HOLD_SECONDS
            )
            print(f"\n[OK] CHECK-IN thanh cong | ID={ticket_counter} | Plate={current_plate}")
            ticket_counter += 1
            last_action_time = now
        else:
            # NGƯỜI RA -> CHECK-OUT (phải qua xác thực + rule 30 giây)
            last_in_time = last_checkin_by_plate.get(current_plate)
            if last_in_time is not None and (now - last_in_time) < CHECKOUT_DELAY_AFTER_CHECKIN:
                wait_seconds = int(CHECKOUT_DELAY_AFTER_CHECKIN - (now - last_in_time))
                if can_overwrite_status:
                    set_status(f"Chua du 30s de CHECK-OUT | Con {wait_seconds}s", (0, 140, 255))
            else:
                checkout_data = {
                    "face": person_data["face_emb"],
                    "clothes": person_data["clothes_emb"],
                    "helmet": person_data["helmet"],
                    "plate": current_plate
                }

                is_valid, verify_score, verify_message = verify_checkout(checkin_data, checkout_data)
                if is_valid:
                    image_path_person = f"captures/out_{matched_ticket_id}_person.jpg"
                    cv2.imwrite(image_path_person, frame_person)

                    if use_dual_camera and frame_plate is not None:
                        image_path_plate = f"captures/out_{matched_ticket_id}_plate.jpg"
                        cv2.imwrite(image_path_plate, frame_plate)

                    update_checkout(matched_ticket_id, image_path_person)
                    set_status(
                        f"CHECK-OUT OK | ID {matched_ticket_id} | Score {verify_score:.2f}",
                        (0, 255, 0),
                        EVENT_MESSAGE_HOLD_SECONDS
                    )
                    print(f"\n[OK] CHECK-OUT thanh cong | ID={matched_ticket_id} | Plate={current_plate}")
                    # Sau khi checkout thanh cong, giai phong lock theo bien so de cho phep checkin bien so moi
                    if current_plate in last_checkin_by_plate:
                        del last_checkin_by_plate[current_plate]
                    # reset trạng thái ổn định sau khi xử lý xong
                    stable_start_time = None
                    stable_plate = None
                    stable_face_emb = None
                    # reset plate tracking de tranh dinh bien so cu, cho phep check-in bien so moi nhanh hon
                    plate_history.clear()
                    current_plate = "unknown"
                else:
                    set_status(
                        "CHECK-OUT bi tu choi: khong khop mat/bien so",
                        (0, 0, 255),
                        EVENT_MESSAGE_HOLD_SECONDS
                    )
                    print(f"\n[FAIL] CHECK-OUT tu choi | Plate={current_plate} | {verify_message}")

                last_action_time = now

    # Overlay camera 1
    cv2.putText(frame_person, f"Camera 1 - Person Detection ({person_res_text})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame_person, "Press: [q]=Quit | Auto Mode",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame_person, f"FPS: {fps:.1f} | Plate: {current_plate}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame_person, status_text,
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    display_person = cv2.resize(frame_person, (800, 540))
    cv2.imshow("Camera 1 - Person Detection", display_person)

    key = cv2.waitKey(1)

    dt = now - prev_time
    if dt > 0:
        fps = 1.0 / dt
    prev_time = now

    if key == ord('q'):
        print("\n>>> Dang thoat he thong...")
        break

cam_person_thread.stop()
if cam_plate_thread is not None:
    cam_plate_thread.stop()
cap_person.release()
if use_dual_camera:
    cap_plate.release()
cv2.destroyAllWindows()
print("He thong da dong!")