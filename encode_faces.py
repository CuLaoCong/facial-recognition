import os
import cv2
import face_recognition
import pickle

KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'encodings.pickle'

known_encodings = []
known_names = []

print("[INFO] Bắt đầu xử lý khuôn mặt từ thư mục known_faces...")

# Duyệt qua từng thư mục con (mỗi người)
for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    
    if not os.path.isdir(person_dir):
        continue  # Bỏ qua nếu không phải thư mục

    # Duyệt qua từng file ảnh của người đó
    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)

        # Đọc ảnh
        image = cv2.imread(filepath)
        if image is None:
            print(f"[WARNING] Không đọc được ảnh: {filepath}")
            continue

        # Chuyển ảnh sang RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Tìm vị trí khuôn mặt
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Nếu có ít nhất 1 khuôn mặt
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"[OK] Mã hóa khuôn mặt từ: {filename} (người: {name})")
        else:
            print(f"[WARNING] Không tìm thấy khuôn mặt trong ảnh: {filename}")

# Lưu kết quả nếu có dữ liệu
if known_encodings:
    data = {"encodings": known_encodings, "names": known_names}
    try:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print(f"[SUCCESS] Đã lưu dữ liệu mã hóa vào {ENCODINGS_FILE}")
    except Exception as e:
        print(f"[ERROR] Lỗi khi lưu file {ENCODINGS_FILE}: {e}")
else:
    print("[ERROR] Không có dữ liệu khuôn mặt để lưu! Vui lòng kiểm tra ảnh.")

print("✅ Hoàn tất quá trình xử lý!")
