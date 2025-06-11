import cv2
import os
import time
import subprocess
import shutil
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import engine, SessionLocal, Base
from models.user import Employee

# Khởi tạo bảng


def init_db():
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError as e:
        print(f"Error initializing database: {e}")


init_db()

# Tạo thư mục nếu chưa có
os.makedirs("known_faces", exist_ok=True)

# Nhập thông tin cá nhân
id_code = input("🔢 Nhập mã ID: ").strip()
person_name = input("👤 Nhập tên người dùng: ").strip()
dob = input("📅 Nhập ngày sinh (dd-mm-yyyy): ").strip()
position = input("💼 Nhập vị trí công việc: ").strip()

# Kiểm tra ID có trùng không
with SessionLocal() as db:
    emp = db.query(Employee).filter_by(employee_id=id_code).first()
    if emp:
        overwrite = input(
            f"[!] Mã ID '{id_code}' đã tồn tại. Ghi đè thông tin? (y/n): ").strip().lower()
        if overwrite == "y":
            old_dirs = [d for d in os.listdir(
                "known_faces") if d.startswith(id_code)]
            for d in old_dirs:
                shutil.rmtree(os.path.join("known_faces", d))
        else:
            print("[❌] Đã huỷ.")
            exit()

# Tạo thư mục lưu ảnh
dataset_path = os.path.join("known_faces", f"{id_code}_{person_name}")
os.makedirs(dataset_path, exist_ok=True)

# Chọn phương thức thêm ảnh
print("\nChọn cách thêm ảnh:")
print("1. 📷 Chụp ảnh bằng webcam")
print("2. 📁 Thêm ảnh thủ công vào thư mục")
choice = input("👉 Nhập 1 hoặc 2: ").strip()

if choice == "1":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở webcam.")
        exit()

    print("\n[INFO] Bắt đầu chụp ảnh...")
    print("👉 Nhìn vào camera. Nhấn SPACE để chụp, ESC để thoát.")
    time.sleep(1)

    count = 0
    max_images = 10
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Không đọc được từ webcam.")
            break

        frame_display = frame.copy()
        cv2.putText(frame_display, f"Ảnh: {count+1}/{max_images} | SPACE: Chụp | ESC: Thoát",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Chụp ảnh khuôn mặt", frame_display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] Hủy chụp.")
            break
        elif key == 32:
            for i in range(3, 0, -1):
                print(f"Chụp sau {i}...", end="\r")
                time.sleep(1)

            img_path = os.path.join(
                dataset_path, f"{person_name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] ✅ Đã lưu: {img_path}")
            count += 1
            time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

elif choice == "2":
    print(f"\n👉 Hãy thêm ảnh thủ công vào thư mục: {dataset_path}")
    input("⏳ Nhấn Enter khi đã thêm xong...")
else:
    print("❌ Lựa chọn không hợp lệ.")
    exit()

# Kiểm tra thư mục có ảnh chưa
if not os.listdir(dataset_path):
    print("⚠️ Không có ảnh trong thư mục. Thoát.")
    exit()

# Lưu thông tin nhân viên vào DB
try:
    with SessionLocal() as db:
        emp = db.query(Employee).filter_by(employee_id=id_code).first()
        if emp:
            emp.name = person_name
            emp.dob = dob
            emp.position = position
        else:
            emp = Employee(
                employee_id=id_code,
                name=person_name,
                dob=dob,
                position=position
            )
            db.add(emp)
        db.commit()
    print("[✅] Đã lưu thông tin nhân viên vào CSDL.")
except SQLAlchemyError as e:
    print(f"[ERROR] Lưu thông tin thất bại: {e}")
    exit()

# Gọi update_encodings.py
print("[INFO] Đang cập nhật encodings.pickle...")
result = subprocess.run(["python", "update_encodings.py"])
if result.returncode == 0:
    print("[INFO] ✅ Encode thành công.")
else:
    print("[ERROR] ❌ Encode thất bại. Kiểm tra update_encodings.py.")
