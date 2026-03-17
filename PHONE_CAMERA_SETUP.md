# Hướng dẫn sử dụng Điện thoại làm Camera thứ 2

## 🍎 Phương pháp cho iOS (iPhone/iPad)

### Phương pháp 1: iVCam (Khuyến nghị nhất cho iOS)

#### Bước 1: Cài đặt App và Driver
- **iPhone**: Tải "iVCam Webcam" từ App Store (miễn phí)
- **PC**: Tải iVCam Client từ https://www.e2esoft.com/ivcam/

#### Bước 2: Kết nối
1. Cài đặt driver trên PC và khởi động lại máy
2. Mở app iVCam trên iPhone
3. Mở phần mềm iVCam trên PC
4. Đảm bảo iPhone và PC cùng mạng WiFi
5. App tự động kết nối (hoặc dùng USB cable cho ổn định hơn)

#### Bước 3: Sử dụng trong Python
```python
# iPhone qua iVCam sẽ trở thành camera index 1
cap1 = cv2.VideoCapture(0)  # Laptop
cap2 = cv2.VideoCapture(1)  # iPhone qua iVCam
```

#### Lưu ý:
- ✅ Chất lượng tốt, ổn định
- ✅ Hỗ trợ cả WiFi và USB
- ✅ Không có watermark
- ⚠️ Cần cài driver trên PC

---

### Phương pháp 2: EpocCam (Dễ dùng)

#### Bước 1: Cài đặt
- **iPhone**: Tải "EpocCam Webcam" từ App Store
- **PC**: Tải driver từ https://www.kinoni.com/

#### Bước 2: Kết nối
1. Cài driver trên PC
2. Mở app trên iPhone
3. Tự động kết nối khi cùng WiFi

#### Lưu ý:
- ⚠️ Bản miễn phí có watermark nhỏ ở góc
- ✅ Hỗ trợ HD với bản Pro
- ✅ Kết nối ổn định

---

### Phương pháp 3: DroidCam (iOS version)

#### Bước 1: Cài đặt
- **iPhone**: Tải "DroidCam Webcam OBS Camera" từ App Store
- **PC**: Tải từ https://www.dev47apps.com/

#### Bước 2: Kết nối tương tự iVCam

---

## 🤖 Phương pháp cho Android

### Phương pháp 1: IP Webcam App (Khuyến nghị - Đơn giản nhất)

#### Bước 1: Cài đặt App
- **Android**: Tải "IP Webcam" từ Google Play Store

### Bước 2: Cấu hình IP Webcam (Android)
1. Mở app "IP Webcam"
2. Kéo xuống cuối và nhấn "Start server"
3. App sẽ hiển thị địa chỉ IP, ví dụ: `http://192.168.1.100:8080`
4. Ghi lại địa chỉ này

### Bước 3: Lấy URL cho OpenCV
- URL để mở trong OpenCV: `http://192.168.1.100:8080/video`
- Hoặc: `http://192.168.1.100:8080/videofeed`

### Bước 4: Kết nối trong Python
```python
import cv2

# Camera laptop (mặc định)
cap1 = cv2.VideoCapture(0)

# Camera điện thoại (thay IP của bạn)
cap2 = cv2.VideoCapture("http://192.168.1.100:8080/video")

# Hoặc
cap2 = cv2.VideoCapture("http://192.168.1.100:8080/videofeed")
```

#### Lưu ý:
- ✅ Không cần cài driver trên PC
- ✅ Hoàn toàn miễn phí
- ⚠️ Cần biết địa chỉ IP
- ⚠️ Có thể bị lag nếu WiFi yếu

---

## 📋 So sánh các phương pháp

| App | Platform | Miễn phí | Cài Driver | Độ khó | Chất lượng |
|-----|----------|----------|------------|---------|-----------|
| **iVCam** | iOS | ✅ | ✅ Cần | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **EpocCam** | iOS | Có watermark | ✅ Cần | ⭐⭐ | ⭐⭐⭐⭐ |
| **DroidCam** | iOS/Android | ✅ | ✅ Cần | ⭐⭐ | ⭐⭐⭐⭐ |
| **IP Webcam** | Android | ✅ | ❌ Không | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 Khuyến nghị theo thiết bị

### iPhone/iPad:
1. **Tốt nhất**: iVCam (chất lượng cao, ổn định)
2. **Thay thế**: EpocCam (nếu không ngại watermark)

### Android:
1. **Đơn giản nhất**: IP Webcam (không cần driver)
2. **Ổn định nhất**: DroidCam (có driver support)

---

## ⚙️ Cấu hình trong Project

### Nếu dùng iVCam / EpocCam / DroidCam (có driver):
File [config.py](src/config.py):
```python
# Camera cho thân người (laptop webcam)
PERSON_CAMERA = 0

# Camera cho biển số xe (phone with driver)
PLATE_CAMERA = 1
```

### Nếu dùng IP Webcam (không driver):
File [config.py](src/config.py):
```python
# Camera cho thân người (laptop webcam)
PERSON_CAMERA = 0

# Camera cho biển số xe (IP Webcam)
PLATE_CAMERA = "http://192.168.1.100:8080/video"  # Thay IP của bạn
```

---

## 💡 Lưu ý quan trọng

### 1. Cùng mạng WiFi
- Laptop và điện thoại **PHẢI** kết nối cùng một mạng WiFi
- Tắt dữ liệu di động trên điện thoại để chắc chắn dùng WiFi

### 2. Kiểm tra kết nối trước khi chạy
```python
import cv2

# Test camera laptop
cap = cv2.VideoCapture(0)
print(f"Camera laptop (0): {cap.isOpened()}")
cap.release()

# Test camera điện thoại (nếu dùng driver)
cap2 = cv2.VideoCapture(1)
print(f"Camera điện thoại (1): {cap2.isOpened()}")
cap2.release()

# Test camera điện thoại (nếu dùng IP Webcam)
cap2 = cv2.VideoCapture("http://192.168.1.100:8080/video")
print(f"Camera IP Webcam: {cap2.isOpened()}")
cap2.release()
```

### 3. Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|-------------|-----------|
| Không kết nối được | Firewall chặn | Tắt firewall tạm thời |
| Video lag/giật | WiFi yếu | Đứng gần router hoặc dùng USB |
| Camera index sai | Nhiều webcam | Thử index 1, 2, 3... |
| IP thay đổi | DHCP tự động | Cố định IP trong router |
| Driver không nhận | Chưa restart | Restart PC sau khi cài driver |

### 4. Tips tối ưu

**Cho iOS (iVCam):**
- ✅ Dùng USB cable cho kết nối ổn định nhất
- ✅ Đặt Resolution 720p để cân bằng tốc độ/chất lượng
- ✅ Tắt Auto-lock trên iPhone

**Cho Android (IP Webcam):**
- ✅ Chọn Resolution 640x480 trong Settings của app
- ✅ Tắt các hiệu ứng không cần thiết
- ✅ Bật "Stay awake" trong app

---

## 🚀 Test nhanh setup của bạn

Chạy lệnh sau để test:

```bash
cd src
python -c "import cv2; c1=cv2.VideoCapture(0); c2=cv2.VideoCapture(1); print(f'Cam0: {c1.isOpened()}, Cam1: {c2.isOpened()}')"
```

Kết quả mong muốn:
```
Cam0: True, Cam1: True
```

---

## Demo Layout

```
┌─────────────────────┐  ┌─────────────────────┐
│   Camera Laptop     │  │  Camera Phone       │
│   (Thân người)      │  │  (Biển số xe)       │
│                     │  │                     │
│   - Face            │  │   - License Plate   │
│   - Clothes         │  │   - OCR             │
│   - Helmet          │  │                     │
└─────────────────────┘  └─────────────────────┘
```
