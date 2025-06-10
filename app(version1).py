import cv2
import face_recognition
import numpy as np
import pickle
from ultralytics import YOLO
from flask import Flask, request, Response, jsonify, render_template_string
import datetime
import pandas as pd
import os
import threading
import time
from functools import lru_cache

app = Flask(__name__)

# Cấu hình
WEBCAM_INDEX = 0  # Sử dụng webcam máy tính, thường là 0
ATTENDANCE_FILE = "attendance.csv"
UPLOAD_DIR = "uploads"

# Load encodings
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
except FileNotFoundError:
    print("Error: encodings.pickle not found")
    known_encodings = []
    known_names = []

# Load YOLO
yolo_model = YOLO("yolov8n.pt")

# Bộ nhớ đệm để tránh nhận diện lặp lại
last_detected = {}
CACHE_TIMEOUT = 10  # Giây

# Tạo thư mục uploads nếu chưa có
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Hàm nhận diện khuôn mặt
@lru_cache(maxsize=100)
def recognize_face(encoding_tuple):
    encoding = np.array(encoding_tuple)
    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for i, b in enumerate(matches) if b]
        counts = {known_names[i]: face_recognition.face_distance(known_encodings, encoding)[i] for i in matched_idxs}
        name = min(counts, key=counts.get)  # Chọn tên có khoảng cách nhỏ nhất
    return name

# Hàm xử lý stream webcam
def process_stream():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Dò khuôn mặt bằng YOLO (class 0 là "person")
        results = yolo_model(frame, conf=0.5)[0]
        face_locations = []
        for box in results.boxes:
            if box.cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # face_recognition dùng định dạng (top, right, bottom, left)
                face_locations.append((y1, x2, y2, x1))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        names = []
        confidences = []

        for encoding in encodings:
            current_time = time.time()
            encoding_tuple = tuple(encoding)
            name = recognize_face(encoding_tuple)

            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                confidence = last_detected[name]["confidence"]
            else:
                distances = face_recognition.face_distance(known_encodings, encoding)
                confidence = 1 - min(distances) if distances.size > 0 else 0
                last_detected[name] = {"name": name, "time": current_time, "confidence": confidence}

            names.append(name)
            confidences.append(confidence)

            if name != "Unknown":
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data = {"Name": name, "Time": timestamp, "Confidence": confidence}
                df = pd.DataFrame([data])
                df.to_csv(ATTENDANCE_FILE, mode='a', index=False, header=not os.path.exists(ATTENDANCE_FILE))

                # Lưu ảnh
                safe_timestamp = timestamp.replace(":", "-")
                img_path = os.path.join(UPLOAD_DIR, f"{name}_{safe_timestamp}.jpg")
                cv2.imwrite(img_path, frame)

        # Vẽ khung và tên lên frame
        for (top, right, bottom, left), name in zip(face_locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Webcam Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm tạo stream MJPEG
def generate_stream():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ret2, jpeg = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head><title>Webcam Stream</title></head>
    <body>
        <h1>Live Stream from Webcam</h1>
        <img src="{{ url_for('video_feed') }}" width="640" height="480" />
        <p>Nhấn 'q' để đóng cửa sổ video stream.</p>
    </body>
    </html>
    '''
    return render_template_string(html)

if __name__ == '__main__':
    threading.Thread(target=process_stream, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
