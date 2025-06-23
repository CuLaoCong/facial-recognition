# Hệ Thống Nhận Diện Khuôn Mặt - Face Recognition System

Hệ thống chấm công tự động sử dụng AI nhận diện khuôn mặt, hỗ trợ cả webcam và ESP32-CAM.

## �� Tính năng chính

- ✅ Nhận diện khuôn mặt real-time
- ✅ Chấm công tự động
- ✅ Hỗ trợ webcam và ESP32-CAM
- ✅ Giao diện web thân thiện
- ✅ Xuất báo cáo Excel
- ✅ Quản lý nhân viên
- ✅ Lưu trữ ảnh chấm công
- ✅ API RESTful

## ��️ Công nghệ sử dụng

- **Backend**: FastAPI, Python
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: Face Recognition, YOLO
- **Database**: SQLAlchemy, SQLite
- **Hardware**: ESP32-CAM (tùy chọn)

## �� Yêu cầu hệ thống

- Python 3.8+
- Webcam hoặc ESP32-CAM
- RAM: 4GB+
- Storage: 2GB+

## ⚙️ Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/CuLaoCong/facial-recognition.git
cd facial-recognition
```

### 2. Tạo virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Tải model YOLO (nếu chưa có)
```bash
# Model sẽ được tải tự động khi chạy lần đầu
# Hoặc tải thủ công:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## �� Cấu hình

### 1. Cấu hình nguồn camera
Mở file `app.py` và thay đổi:
```python
WEBCAMOPTION = 'webcam'      # Sử dụng webcam máy tính
# hoặc
WEBCAMOPTION = 'ESP32-CAM'   # Sử dụng ESP32-CAM
```

### 2. Cấu hình ESP32-CAM (nếu sử dụng)
```python
ESP32_CAM_URL = "http://YOUR_ESP32_IP/stream"
INFO_URL = "http://YOUR_ESP32_IP/info"
```

### 3. Cấu hình WiFi cho ESP32
Mở file `Nhan_dien_khuon_mat.ino` và thay đổi:
```cpp
const char* ssid = "YOUR_WIFI_NAME";
const char* password = "YOUR_WIFI_PASSWORD";
```

## �� Chạy ứng dụng

### 1. Khởi động server
```bash
python app.py
```

### 2. Truy cập ứng dụng
Mở trình duyệt và truy cập: `http://localhost:8001`

## 📖 Hướng dẫn sử dụng

### 1. Đăng ký nhân viên mới
- Truy cập trang "Đăng Ký Chấm Công"
- Chụp 5 ảnh khuôn mặt
- Điền thông tin nhân viên
- Hoàn tất đăng ký

### 2. Chấm công tự động
- Truy cập trang chính
- Hệ thống sẽ tự động nhận diện và chấm công
- Xem thông tin nhận diện real-time

### 3. Xem báo cáo
- Truy cập "Danh Sách Điểm Danh"
- Xem lịch sử chấm công
- Xuất báo cáo Excel

## �� Cấu trúc thư mục
