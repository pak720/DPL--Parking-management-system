import sys
import cv2
import time
import os
from collections import Counter, deque
import numpy as np
from PIL import ImageFont, ImageDraw, Image

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer

# Import modules từ project của bạn
from extractor import extract_all, extract_license_plate
from matcher import find_match, verify_checkout
from database import init_db, load_active_tickets, add_checkin, update_checkout
from config import PERSON_CAMERA, PLATE_CAMERA


def put_text_vietnamese(img, text, position, text_color=(0, 255, 0), font_size=42):
    """
    Hàm hỗ trợ viết Tiếng Việt có dấu lên ảnh OpenCV (thông qua Pillow)
    Có gài thêm viền đen chìm xung quanh chữ để đổi màu sáng không bị chìm nền.
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Load font mặc định của Windows hỗ trợ Tiếng Việt (Arial) - ưu tiên chữ đậm
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
            
    r, g, b = text_color[2], text_color[1], text_color[0]
    x, y = position
    
    # Vẽ viền đen shadow dày 2px
    stroke = 2
    for dx in range(-stroke, stroke + 1):
        for dy in range(-stroke, stroke + 1):
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
            
    # Vẽ text chính
    draw.text((x, y), text, font=font, fill=(r, g, b))
    
    # Kéo ngược lại thành ma trận OpenCV
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class CameraWorker(QThread):
    # Signals truyền kết quả vể luồng UI chính
    change_pixmap_person = pyqtSignal(QImage)
    change_pixmap_plate = pyqtSignal(QImage)
    update_status = pyqtSignal(str, str) # text, color_hex
    update_metrics = pyqtSignal(str) # thong so confidence/accuracy
    update_current_plate = pyqtSignal(str) # biển số hiện tại

    def __init__(self):
        super().__init__()
        self.running = True
        
        # --- CONSTANTS COPIED FROM MAIN ---
        self.MIN_ACTION_GAP = 0.8  
        self.CHECKOUT_DELAY_AFTER_CHECKIN = 30.0  
        self.CHECKIN_DELAY_AFTER_CHECKOUT = 10.0
        self.MIN_CHECKIN_GAP = 5.0  
        self.STABLE_REQUIRED_SECONDS = 5.0  
        self.EVENT_MESSAGE_HOLD_SECONDS = 5.0  
        self.PERSON_INFER_SCALE = 0.7  
        self.FACE_PLATE_LOCK_THRESHOLD = 0.78  
        
    def optimize_camera(self, cap, width=None, height=None):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width is not None and height is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def draw_alignment_boxes(self, frame, is_person_cam=True):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw translucent guide box
        if is_person_cam:
            # Camera 1: bo khung, cho phep quet toan bo khung hinh
            return frame
        else:
            # Vùng canh biển số xe (Landscape/Wide shape) TO HƠN
            box_w, box_h = int(w * 0.65), int(h * 0.45)
            x1, y1 = int((w - box_w) / 2), int(h * 0.35) # Đặt cao vừa phải
            cv2.rectangle(overlay, (x1, y1), (x1 + box_w, y1 + box_h), (255, 255, 0), 3) # Xanh Cyan mờ
            overlay = put_text_vietnamese(overlay, "Đưa biển số xe vào ô", (x1 - 20, y1 - 50), text_color=(0, 255, 0), font_size=42)

        alpha = 0.5  # Đô mờ đậm hơn 1 tí để rõ màu khung
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def get_plate_roi(self, frame):
        """Camera 2 chi detect trong ROI khung huong dan."""
        h, w = frame.shape[:2]
        box_w, box_h = int(w * 0.65), int(h * 0.45)
        x1, y1 = int((w - box_w) / 2), int(h * 0.35)
        x2, y2 = x1 + box_w, y1 + box_h
        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)
        
    def emit_status(self, text, color_rgb=(0, 255, 0)):
        # Convert RGB to HEX for Qt Stylesheet
        hex_color = "#{:02x}{:02x}{:02x}".format(color_rgb[0], color_rgb[1], color_rgb[2])
        self.update_status.emit(text, hex_color)

    def build_confidence_text(self, plate_history, current_plate, stable_face_emb, current_face_emb, stable_start_time):
        # Plate confidence from voting window ratio
        plate_conf = 0.0
        if current_plate != "unknown" and len(plate_history) > 0:
            same = sum(1 for p in plate_history if p == current_plate)
            plate_conf = same / float(len(plate_history))

        # Face confidence from cosine similarity with tracked stable face
        face_conf = 0.0
        if stable_face_emb is not None and current_face_emb is not None:
            try:
                va = np.asarray(stable_face_emb, dtype=np.float32).reshape(-1)
                vb = np.asarray(current_face_emb, dtype=np.float32).reshape(-1)
                na = np.linalg.norm(va)
                nb = np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    face_conf = float(np.dot(va, vb) / (na * nb))
            except:
                face_conf = 0.0

        # Stability progress (0..1)
        stable_progress = 0.0
        if stable_start_time is not None:
            stable_progress = min(1.0, max(0.0, (time.time() - stable_start_time) / self.STABLE_REQUIRED_SECONDS))

        # Overall confidence (weighted)
        overall = 0.45 * face_conf + 0.35 * plate_conf + 0.20 * stable_progress
        overall = max(0.0, min(1.0, overall))

        return (
            f"Face: {face_conf * 100:5.1f}%  |  "
            f"Plate: {plate_conf * 100:5.1f}%  |  "
            f"Stable: {stable_progress * 100:5.1f}%  |  "
            f"Overall: {overall * 100:5.1f}%"
        )

    def run(self):
        # Setup logic tu main.py
        import re
        import csv
        
        os.makedirs("captures", exist_ok=True)
        init_db()
        
        PLATE_FORMAT_REGEXES = [
            re.compile(r"^[A-Z0-9]{4}-\d{4,5}$"),
            re.compile(r"^\d{2}MD[A-Z0-9]-\d{5}$"),
        ]

        def is_valid_plate_format(plate_text):
            if plate_text is None: return False
            return any(rx.fullmatch(str(plate_text).upper().strip()) for rx in PLATE_FORMAT_REGEXES)

        def cosine_sim(a, b):
            if a is None or b is None: return 0.0
            try:
                va = np.asarray(a, dtype=np.float32).reshape(-1)
                vb = np.asarray(b, dtype=np.float32).reshape(-1)
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na == 0 or nb == 0: return 0.0
                return float(np.dot(va, vb) / (na * nb))
            except:
                return 0.0

        def has_face_plate_conflict(query_face_emb, query_plate, active_tickets):
            if query_face_emb is None: return None, 0.0
            best_p, best_s = None, 0.0
            for _, stored in active_tickets.items():
                stored_plate = str(stored.get("plate", "")).upper().strip()
                if stored_plate == query_plate: continue
                sim = cosine_sim(query_face_emb, stored.get("face"))
                if sim > self.FACE_PLATE_LOCK_THRESHOLD and sim > best_s:
                    best_s, best_p = sim, stored_plate
            return best_p, best_s

        ticket_counter = 1
        if os.path.exists("parking_db.csv"):
            with open("parking_db.csv") as f:
                rows = list(csv.reader(f))
                if len(rows) > 1: ticket_counter = int(rows[-1][0]) + 1

        self.emit_status("Đang mở camera...", (255, 255, 255))
        cap_person = cv2.VideoCapture(PERSON_CAMERA)
        cap_plate = cv2.VideoCapture(PLATE_CAMERA)
        
        self.optimize_camera(cap_person, width=1920, height=1080)
        self.optimize_camera(cap_plate)
        cap_plate.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap_plate.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        use_dual_camera = cap_plate.isOpened()
        
        if not cap_person.isOpened():
            self.emit_status("ERROR: Không thể mở camera thân người!", (255, 0, 0))
            return
            
        self.emit_status("Hệ thống sẵn sàng!", (0, 255, 0))

        # Variables
        current_plate = "unknown"
        plate_history = deque(maxlen=5)
        last_action_time = 0.0
        last_checkin_by_plate = {}
        recently_checkout_plates = {}
        status_hold_until = 0.0
        
        stable_start_time = None
        stable_plate = None
        stable_face_emb = None
        last_checkin_time = 0.0
        
        plate_ocr_every_n_frames = 1
        plate_frame_idx = 0
        
        while self.running:
            now = time.time()
            can_trigger_action = (now - last_action_time) >= self.MIN_ACTION_GAP
            can_overwrite_status = now >= status_hold_until

            ret1, frame_person = cap_person.read()
            if not ret1: break
            
            frame_plate = None
            if use_dual_camera:
                ret2, frame_plate = cap_plate.read()
                if ret2:
                    plate_frame_idx += 1
                    if plate_frame_idx % plate_ocr_every_n_frames == 0:
                        # Chi quet bien so trong khung ROI cua Camera 2
                        plate_roi, _ = self.get_plate_roi(frame_plate)
                        p_cnd = extract_license_plate(plate_roi)
                        if p_cnd != "unknown": plate_history.append(p_cnd)
                        current_plate = Counter(plate_history).most_common(1)[0][0] if plate_history else "unknown"

                    # Convert and emit Plate Frame
                    display_frame_plate = self.draw_alignment_boxes(frame_plate, False)
                    rgb_plate = cv2.cvtColor(display_frame_plate, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_plate.shape
                    qimg_plate = QImage(rgb_plate.data, w, h, ch * w, QImage.Format_RGB888)
                    self.change_pixmap_plate.emit(qimg_plate)
            
            self.update_current_plate.emit(current_plate)

            infer_person_frame = cv2.resize(frame_person, None, fx=self.PERSON_INFER_SCALE, fy=self.PERSON_INFER_SCALE, interpolation=cv2.INTER_LINEAR)
            person_data = extract_all(infer_person_frame)

            if person_data is not None and person_data.get("face_count", 1) > 1 and can_overwrite_status:
                self.emit_status(
                    f"Phat hien {person_data.get('face_count')} khuon mat - dang chon rider chinh (mat lon nhat)",
                    (0, 200, 255),
                )

            # Emit confidence metrics for UI
            current_face_emb = person_data.get("face_emb") if person_data is not None else None
            metrics_text = self.build_confidence_text(
                plate_history,
                current_plate,
                stable_face_emb,
                current_face_emb,
                stable_start_time,
            )
            self.update_metrics.emit(metrics_text)

            # ----- LOGIC STATUS CHUYỂN QUA GỌI HÀM emit_status -----
            stable_ready = False
            if person_data is not None and current_plate != "unknown" and is_valid_plate_format(current_plate):
                if stable_plate == current_plate and cosine_sim(stable_face_emb, person_data.get("face_emb")) >= 0.80:
                    if stable_start_time is None: stable_start_time = now
                    if (now - stable_start_time) >= self.STABLE_REQUIRED_SECONDS:
                        stable_ready = True
                        if can_overwrite_status:
                            self.emit_status("OK: Người + biển số ổn định, sẵn sàng xử lý ", (0, 255, 0))
                    else:
                        if can_overwrite_status:
                            self.emit_status("Chỗ đứng yên để xác thực...", (255, 200, 0))
                else:
                    stable_plate, stable_face_emb, stable_start_time = current_plate, person_data.get("face_emb"), now
                    if can_overwrite_status:
                        self.emit_status("Bắt đầu theo dõi ổn định...", (255, 200, 0))
            else:
                stable_start_time, stable_plate, stable_face_emb = None, None, None
                if can_overwrite_status:
                    if current_plate == "unknown":
                        self.emit_status("Chỗ hiển thị rõ khuôn mặt và biển số...", (150, 150, 150))
                    else:
                        self.emit_status("Chổ hiển thị rõ khuôn mặt...", (150, 150, 150))
                        
            if current_plate != "unknown" and not is_valid_plate_format(current_plate):
                if can_overwrite_status:
                    self.emit_status(f"Biển số sai format: {current_plate}. Quét lại!", (255, 0, 0))
                    status_hold_until = now + self.EVENT_MESSAGE_HOLD_SECONDS
                stable_start_time, stable_plate, stable_face_emb = None, None, None
                plate_history.clear()
                current_plate = "unknown"

            # ----- LOGIC CHECKIN/OUT CORE -----
            if person_data is not None and current_plate != "unknown" and is_valid_plate_format(current_plate) and stable_ready and can_trigger_action:
                active_tickets = load_active_tickets()
                matched_ticket_id = None
                checkin_data = None

                for ticket_id, stored in active_tickets.items():
                    stored_plate = stored.get("plate")
                    if not is_valid_plate_format(stored_plate): continue
                    if stored_plate == current_plate:
                        matched_ticket_id = ticket_id
                        checkin_data = stored
                        break

                if matched_ticket_id is None:
                    last_out_time = recently_checkout_plates.get(current_plate)
                    if last_out_time is not None and (now - last_out_time) < self.CHECKIN_DELAY_AFTER_CHECKOUT:
                        wait_leave = int(self.CHECKIN_DELAY_AFTER_CHECKOUT - (now - last_out_time))
                        if can_overwrite_status:
                            self.emit_status(f"Mời Check-out , vui lòng đợi {wait_leave}s...", (255, 140, 0))
                        continue

                    conflict_plate, conflict_sim = has_face_plate_conflict(person_data.get("face_emb"), current_plate, active_tickets)
                    if conflict_plate is not None:
                        # Tự động phân biệt: khuôn mặt này ĐÃ có xe trong bãi
                        # => Người dùng đang cố CHECK-IN xe mới hoặc đưa sai biển số khi muốn CHECK-OUT
                        self.emit_status(
                            f"⚠️ Bạn đang có xe [{conflict_plate}] trong bãi. Hãy CHECK-OUT xe cũ trước hoặc đưa đúng biển số!",
                            (255, 165, 0)  # Màu cam cảnh báo
                        )
                        status_hold_until = now + self.EVENT_MESSAGE_HOLD_SECONDS
                        last_action_time = now
                        continue

                    if (now - last_checkin_time) < self.MIN_CHECKIN_GAP:
                        wait_checkin = int(self.MIN_CHECKIN_GAP - (now - last_checkin_time))
                        if can_overwrite_status:
                            self.emit_status(f"Vui lòng đợi {wait_checkin}s để tạo mẫ...", (255, 140, 0))
                        continue

                    image_path_person = f"captures/in_{ticket_counter}_person.jpg"
                    cv2.imwrite(image_path_person, frame_person)
                    if use_dual_camera and frame_plate is not None:
                        image_path_plate = f"captures/in_{ticket_counter}_plate.jpg"
                        cv2.imwrite(image_path_plate, frame_plate)

                    add_checkin(ticket_counter, current_plate, person_data["helmet"], person_data["helmet_color"], person_data["face_emb"], person_data["clothes_emb"], image_path_person)
                    last_checkin_by_plate[current_plate] = now
                    last_checkin_time = now
                    
                    self.emit_status(f"✅ CHECK-IN OK | ID {ticket_counter} | {current_plate}", (0, 255, 0))
                    status_hold_until = now + self.EVENT_MESSAGE_HOLD_SECONDS
                    print(f"\n[OK] CHECK-IN id={ticket_counter} plate={current_plate}")
                    
                    ticket_counter += 1
                    last_action_time = now
                else:
                    last_in_time = last_checkin_by_plate.get(current_plate)
                    if last_in_time is not None and (now - last_in_time) < self.CHECKOUT_DELAY_AFTER_CHECKIN:
                        wait_seconds = int(self.CHECKOUT_DELAY_AFTER_CHECKIN - (now - last_in_time))
                        if can_overwrite_status:
                            self.emit_status(f"Chưa đủ 30 giây để Check-out | Còn {wait_seconds}s", (255, 140, 0))
                    else:
                        checkout_data = {"face": person_data["face_emb"], "clothes": person_data["clothes_emb"], "helmet": person_data["helmet"], "plate": current_plate}
                        is_valid, verify_score, verify_message = verify_checkout(checkin_data, checkout_data)
                        
                        if is_valid:
                            image_path_person = f"captures/out_{matched_ticket_id}_person.jpg"
                            cv2.imwrite(image_path_person, frame_person)

                            if use_dual_camera and frame_plate is not None:
                                image_path_plate = f"captures/out_{matched_ticket_id}_plate.jpg"
                                cv2.imwrite(image_path_plate, frame_plate)

                            update_checkout(matched_ticket_id, image_path_person)
                            
                            self.emit_status(f"✅ CHECK-OUT OK | ID {matched_ticket_id}", (0, 255, 0))
                            status_hold_until = now + self.EVENT_MESSAGE_HOLD_SECONDS
                            print(f"\n[OK] CHECK-OUT id={matched_ticket_id} plate={current_plate}")
                            
                            recently_checkout_plates[current_plate] = now
                            if current_plate in last_checkin_by_plate: del last_checkin_by_plate[current_plate]
                            stable_start_time, stable_plate, stable_face_emb = None, None, None
                            plate_history.clear()
                            current_plate = "unknown"
                        else:
                            self.emit_status("❌ CHECK-OUT không thành công: Khuôn mặt và biển số không khớp!", (255, 0, 0))
                            status_hold_until = now + self.EVENT_MESSAGE_HOLD_SECONDS
                        last_action_time = now

            # Emit clean person frame with basic guide lines, NOT text status
            display_frame_person = self.draw_alignment_boxes(frame_person, True)
            rgb_person = cv2.cvtColor(display_frame_person, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_person.shape
            qimg_person = QImage(rgb_person.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_person.emit(qimg_person)

        cap_person.release()
        if use_dual_camera: cap_plate.release()

    def stop(self):
        self.running = False
        self.wait()

class ParkingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE THÔNG MINH")
        self.setMinimumSize(1280, 800)
        self.showMaximized()

        # ===== GLOBAL STYLESHEET =====
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d1117;
            }
            QLabel {
                color: #e6edf3;
            }
        """)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(12)

        # ===== HEADER BAR =====
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1f36, stop:0.5 #0f2b46, stop:1 #1a1f36);
                border-radius: 12px;
                padding: 8px;
            }
        """)
        header_layout = QHBoxLayout(header_frame)

        # Logo icon text
        logo_label = QLabel("🅿️")
        logo_label.setFont(QFont("Segoe UI Emoji", 32))
        logo_label.setFixedWidth(60)
        logo_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(logo_label)

        # Title
        title_label = QLabel("HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE THÔNG MINH")
        title_label.setFont(QFont("Arial", 22, QFont.Bold))
        title_label.setStyleSheet("color: #58a6ff; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title_label, stretch=1)

        # Time label (live clock)
        self.time_label = QLabel("")
        self.time_label.setFont(QFont("Consolas", 14, QFont.Bold))
        self.time_label.setStyleSheet("color: #8b949e; background: transparent;")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setFixedWidth(200)
        header_layout.addWidget(self.time_label)

        # Timer to update clock
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        main_layout.addWidget(header_frame)

        # ===== CAMERAS AREA =====
        cam_layout = QHBoxLayout()
        cam_layout.setSpacing(16)

        # --- Camera 1: Nhận diện khuôn mặt ---
        cam1_card = QFrame()
        cam1_card.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 2px solid #00bcd4;
                border-radius: 12px;
            }
        """)
        cam1_inner = QVBoxLayout(cam1_card)
        cam1_inner.setContentsMargins(8, 8, 8, 8)
        cam1_inner.setSpacing(6)

        cam1_title = QLabel("📷  CAMERA 1 — NHẬN DIỆN KHUÔN MẶT")
        cam1_title.setFont(QFont("Arial", 13, QFont.Bold))
        cam1_title.setStyleSheet("""
            color: #00e5ff;
            background-color: #0d2137;
            padding: 8px 12px;
            border-radius: 8px;
        """)
        cam1_title.setAlignment(Qt.AlignCenter)
        cam1_inner.addWidget(cam1_title)

        self.label_person_cam = QLabel("Đang kết nối camera...")
        self.label_person_cam.setMinimumSize(500, 380)
        self.label_person_cam.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #30363d;
            border-radius: 8px;
            color: #484f58;
            font-size: 14px;
        """)
        self.label_person_cam.setAlignment(Qt.AlignCenter)
        cam1_inner.addWidget(self.label_person_cam, stretch=1)

        cam_layout.addWidget(cam1_card, stretch=1)

        # --- Camera 2: Nhận diện biển số ---
        cam2_card = QFrame()
        cam2_card.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 2px solid #ffc107;
                border-radius: 12px;
            }
        """)
        cam2_inner = QVBoxLayout(cam2_card)
        cam2_inner.setContentsMargins(8, 8, 8, 8)
        cam2_inner.setSpacing(6)

        cam2_title = QLabel("📷  CAMERA 2 — NHẬN DIỆN BIỂN SỐ XE")
        cam2_title.setFont(QFont("Arial", 13, QFont.Bold))
        cam2_title.setStyleSheet("""
            color: #ffca28;
            background-color: #2b2000;
            padding: 8px 12px;
            border-radius: 8px;
        """)
        cam2_title.setAlignment(Qt.AlignCenter)
        cam2_inner.addWidget(cam2_title)

        self.label_plate_cam = QLabel("Đang kết nối camera...")
        self.label_plate_cam.setMinimumSize(500, 380)
        self.label_plate_cam.setStyleSheet("""
            background-color: #000000;
            border: 1px solid #30363d;
            border-radius: 8px;
            color: #484f58;
            font-size: 14px;
        """)
        self.label_plate_cam.setAlignment(Qt.AlignCenter)
        cam2_inner.addWidget(self.label_plate_cam, stretch=1)

        cam_layout.addWidget(cam2_card, stretch=1)

        main_layout.addLayout(cam_layout, stretch=1)

        # ===== BOTTOM INFO PANEL =====
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)

        # --- Status Banner ---
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 1px solid #30363d;
                border-radius: 10px;
            }
        """)
        status_inner = QVBoxLayout(status_card)
        status_inner.setContentsMargins(12, 8, 12, 8)
        status_inner.setSpacing(4)

        status_header = QLabel("📊 TRẠNG THÁI HỆ THỐNG")
        status_header.setFont(QFont("Arial", 10))
        status_header.setStyleSheet("color: #8b949e; background: transparent; border: none;")
        status_inner.addWidget(status_header)

        self.status_label = QLabel("⏳ Đang khởi động hệ thống...")
        self.status_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #8b949e; background: transparent; border: none; padding: 4px 0px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(60)
        status_inner.addWidget(self.status_label)

        self.metrics_label = QLabel("Face: -- | Plate: -- | Stable: -- | Overall: --")
        self.metrics_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("color: #58a6ff; background: transparent; border: none; padding: 2px 0px;")
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.metrics_label.setMinimumHeight(36)
        status_inner.addWidget(self.metrics_label)

        bottom_layout.addWidget(status_card, stretch=3)

        # --- Plate Display Card ---
        plate_card = QFrame()
        plate_card.setStyleSheet("""
            QFrame {
                background-color: #161b22;
                border: 2px solid #ffc107;
                border-radius: 10px;
            }
        """)
        plate_inner = QVBoxLayout(plate_card)
        plate_inner.setContentsMargins(12, 8, 12, 8)
        plate_inner.setSpacing(4)

        plate_header = QLabel("🔢 BIỂN SỐ HIỆN TẠI")
        plate_header.setFont(QFont("Arial", 10))
        plate_header.setStyleSheet("color: #8b949e; background: transparent; border: none;")
        plate_inner.addWidget(plate_header)

        self.plate_label = QLabel("---")
        self.plate_label.setFont(QFont("Consolas", 28, QFont.Bold))
        self.plate_label.setStyleSheet("color: #ffd54f; background: transparent; border: none;")
        self.plate_label.setAlignment(Qt.AlignCenter)
        self.plate_label.setMinimumHeight(50)
        plate_inner.addWidget(self.plate_label)

        bottom_layout.addWidget(plate_card, stretch=1)

        main_layout.addLayout(bottom_layout)

        # ===== EXIT BUTTON =====
        self.btn_exit = QPushButton("⏻  THOÁT HỆ THỐNG")
        self.btn_exit.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.setFixedHeight(45)
        self.btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                color: #f85149;
                border: 1px solid #f8514933;
                border-radius: 8px;
                padding: 8px 20px;
            }
            QPushButton:hover {
                background-color: #da3633;
                color: white;
                border-color: #da3633;
            }
        """)
        self.btn_exit.clicked.connect(self.close)
        main_layout.addWidget(self.btn_exit)

        # ===== START CAMERA WORKER =====
        self.worker = CameraWorker()
        self.worker.change_pixmap_person.connect(self.update_image_person)
        self.worker.change_pixmap_plate.connect(self.update_image_plate)
        self.worker.update_status.connect(self.update_ui_status)
        self.worker.update_current_plate.connect(self.update_ui_plate)
        self.worker.update_metrics.connect(self.update_ui_metrics)
        self.worker.start()

    def _update_clock(self):
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S  •  %d/%m/%Y")
        self.time_label.setText(now)

    def update_image_person(self, image):
        self.label_person_cam.setPixmap(QPixmap.fromImage(image).scaled(
            self.label_person_cam.width(), self.label_person_cam.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_image_plate(self, image):
        self.label_plate_cam.setPixmap(QPixmap.fromImage(image).scaled(
            self.label_plate_cam.width(), self.label_plate_cam.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_ui_status(self, text, color_hex):
        self.status_label.setText(text)
        self.status_label.setStyleSheet(
            f"color: {color_hex}; background: transparent; border: none; padding: 4px 0px;"
        )

    def update_ui_plate(self, plate_text):
        if plate_text == "unknown":
            self.plate_label.setText("---")
            self.plate_label.setStyleSheet("color: #484f58; background: transparent; border: none;")
        else:
            self.plate_label.setText(plate_text)
            self.plate_label.setStyleSheet("color: #ffd54f; background: transparent; border: none;")

    def update_ui_metrics(self, metrics_text):
        self.metrics_label.setText(metrics_text)

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # High DPI support
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True) if hasattr(Qt, 'AA_EnableHighDpiScaling') else None
    window = ParkingApp()
    window.show()
    sys.exit(app.exec_())
''