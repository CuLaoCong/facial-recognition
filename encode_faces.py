import face_recognition
import os
import pickle

# Thư mục chứa ảnh nhân viên
image_dir = "known_faces"

known_encodings = []
known_names = []

for filename in os.listdir(image_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(image_dir, filename)
        name = os.path.splitext(filename)[0]

        print(f"Đang xử lý ảnh {filename} cho nhân viên {name}")

        # Đọc ảnh và tạo encoding khuôn mặt
        image = face_recognition.load_image_file(path)
        boxes = face_recognition.face_locations(image, model='hog')
        encodings = face_recognition.face_encodings(image, boxes)

        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(name)
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh {filename}")

# Lưu dữ liệu encoding ra file pickle
data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Đã tạo file encodings.pickle thành công!")
