from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle

app = Flask(__name__)

# Load mã hóa khuôn mặt đã lưu
with open("../encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Dữ liệu nhân viên (ví dụ: {Tên: [ID, Chức vụ]})
staff_info = {
    "Nguyen Van A": ["NV001", "Nhân viên kỹ thuật"],
    "Tran Thi B": ["NV002", "Kế toán"],
    # ... thêm nữa nếu có
}
def gen_frames():
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize để nhận diện nhanh hơn
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Không xác định"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            names.append(name)

        # Hiển thị tên trên frame
        for ((top, right, bottom, left), name) in zip(face_locations, names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode lại cho Flask trả về
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # Mặc định hiển thị thông tin trống
    info = {
        "name": "",
        "employee_id": "",
        "position": ""
    }
    return render_template('home.html', **info)

if __name__ == '__main__':
    app.run(debug=True)
