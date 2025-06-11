import signal
import threading
from time import sleep, time
import cv2
import face_recognition
import numpy as np
import pickle
from ultralytics import YOLO
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
import datetime
import os
import requests
import asyncio
import pandas as pd
import io
from functools import lru_cache
from threading import Lock
from typing import List
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from database import engine, SessionLocal, Base, get_db
from models.user import Employee
from models.attendance import Attendance, AttendanceRecord
from contextlib import asynccontextmanager
import uvicorn



app = FastAPI()

# Cấu hình
ESP32_CAM_URL = "http://192.168.1.105/stream"
INFO_URL = "http://192.168.1.105/info"
UPLOAD_DIR = "uploads"
API_TOKEN = "123456"
WEBCAMOPTION = 'webcam' #webcam or ESP32-CAM

# Tạo thư mục uploads
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Khởi tạo bảng


def init_db():
    try:
        Base.metadata.create_all(engine)
    except SQLAlchemyError as e:
        print(f"Error initializing database: {e}")


init_db()

# Load thông tin nhân viên từ CSV vào databse (chạy một lần hoặc nếu chưa có)
if os.path.exists("employee_info.csv"):
    df = pd.read_csv("employee_info.csv")
    with SessionLocal() as db:
        for _, row in df.iterrows():
            emp = db.query(Employee).filter_by(employee_id=row['ID']).first()
            if emp:
                emp.name = row['Name']
                emp.dob = row['DOB']
                emp.position = row['Position']
            else:
                emp = Employee(
                    employee_id=row['ID'],
                    name=row['Name'],
                    dob=row['DOB'],
                    position=row['Position']
                )
                db.add(emp)
            db.commit()

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

# Bộ nhớ đệm
last_detected = {}
CACHE_TIMEOUT = 10
output_frame = None
frame_lock = Lock()
last_detected_lock = Lock()

# Pydantic model cho response
class RecognitionResponse(BaseModel):
    names: List[str]





@lru_cache(maxsize=100)
def recognize_face(encoding_tuple):
    if not known_encodings:
        return "Unknown"
    encoding = np.array(encoding_tuple)
    matches = face_recognition.compare_faces(
        known_encodings, encoding, tolerance=0.45)
    name = "Unknown"
    if True in matches:
        matched_idxs = [i for i, b in enumerate(matches) if b]
        counts = {known_names[i]: face_recognition.face_distance(
            known_encodings, encoding)[i] for i in matched_idxs}
        name = min(counts, key=counts.get)
    return name


def send_to_esp32(name: str, info: str) -> bool:
    try:
        payload = {"name": name, "info": info}
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.post(
            INFO_URL, data=payload, headers=headers, timeout=5)
        return response.status_code == 200
    except requests.RequestException as e:
        print(f"Failed to send info to ESP32-CAM: {e}")
        return False


@app.post("/upload", response_model=RecognitionResponse)
async def upload_image(request: Request, db: Session = Depends(get_db)):
    if request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    file_bytes = await request.body()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    results = yolo_model(frame, conf=0.5)[0]
    face_locations = [(int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.xyxy[0][0]))
                      for box in results.boxes if int(box.cls[0]) == 0]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    names = []

    for encoding in encodings:
        encoding_tuple = tuple(encoding)
        name = recognize_face(encoding_tuple)
        current_time = time()

        with last_detected_lock:
            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                continue
            last_detected[name] = {"time": current_time}

        names.append(name)

        if name != "Unknown":
            timestamp = datetime.datetime.now()
            emp = db.query(Employee).filter_by(name=name.lower()).first()
            if emp:
                confidence = 1 - \
                    min(face_recognition.face_distance(
                        known_encodings, encoding)) if known_encodings else 0
                attendance = Attendance(
                    name=name,
                    time=timestamp,
                    confidence=confidence,
                    employee_id=emp.employee_id,
                    dob=emp.dob,
                    position=emp.position
                )
                db.add(attendance)
                db.commit()
                info = f"ID: {emp.employee_id}, DOB: {emp.dob}, Position: {emp.position}, Confidence: {confidence:.2f}"
                send_to_esp32(name, info)
            img_path = os.path.join(
                UPLOAD_DIR, f"{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(img_path, frame)

    return {"names": names}


@app.get("/report", response_model=List[AttendanceRecord])
async def get_report(db: Session = Depends(get_db)):
    try:
        records = db.query(Attendance).order_by(Attendance.time.desc()).limit(100).all()
        return [
            {
                "id": record.id,
                "name": record.name,
                "time": record.time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": record.confidence,
                "employee_id": record.employee_id,
                "dob": record.dob,
                "position": record.position
            } for record in records
        ]
    except SQLAlchemyError as e:
        print(f"Report error: {e}")
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/export")
async def export_to_excel(db: Session = Depends(get_db)):
    try:
        records = db.query(Attendance).order_by(Attendance.time.desc()).all()
        data = [
            {
                "id": record.id,
                "name": record.name,
                "time": record.time.strftime("%Y-%m-%d %H:%M:%S"),
                "confidence": record.confidence,
                "employee_id": record.employee_id,
                "dob": record.dob,
                "position": record.position
            } for record in records
        ]

        if not data:
            raise HTTPException(
                status_code=404, detail="No attendance data to export")

        df = pd.DataFrame(data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attendance_export_{timestamp}.xlsx"

        return FileResponse(
            output,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=filename
        )
    except SQLAlchemyError as e:
        print(f"Export error: {e}")
        raise HTTPException(
            status_code=500, detail="Server error while exporting")



@app.get("/video_feed")
async def video_feed():
    async def generate():
        global output_frame, frame_lock
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def index(request: Request):
    # hàm này em có thể truyền file trang chủ của tụi em vào
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/camera")
async def camera_demo(request: Request):
    # hàm này sẽ là file html mà mình muốn chuyển đến
    return templates.TemplateResponse("index2.html", {"request": request})

@app.get("/users")
async def get_all_user(request: Request, db: Session = Depends(get_db)):
    # khi em bấm vào user thì nó sẽ chạy tới đây
    # lúc này em mới lấy dữ liệu trong db ra để gửi trả lại cho file html
    records = db.query(Employee).order_by(Employee.name).limit(100).all()
    data = [
            {
                "name": record.name,
                "employee_id": record.employee_id,
                "dob": record.dob,
                "position": record.position,
                "ahihi": "ahehe"
            } for record in records
        ]
    # trả lại thì nó ntn đầu tiên là tên file html tiếp theo là dữ liệu muốn gửi lên
    return templates.TemplateResponse("user.html", {"request": request, "records_html": data})

def process_stream():
    global output_frame, frame_lock
    src_cam = 0 if WEBCAMOPTION == "webcam" else ESP32_CAM_URL
    while True:
        print("src_cam: ", src_cam)
        cap = cv2.VideoCapture(src_cam)
        if not cap.isOpened():
            print(f"Error: Cannot open {WEBCAMOPTION} stream. Retrying...")
            sleep(5)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Stream disconnected. Retrying...")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations)

            db = SessionLocal()
            try:
                for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                    encoding_tuple = tuple(encoding)
                    name = recognize_face(encoding_tuple)
                    name = name.split('_')[-1]

                    # Lấy thông tin nhân viên từ database
                    emp = db.query(Employee).filter_by(name=name).first()
                    
                    # Ghi thông tin điểm danh vào database nếu nhận diện thành công
                    if name != "Unknown" and emp:
                        current_time = time()
                        with last_detected_lock:
                            if name in last_detected and current_time - last_detected[name]["time"] < CACHE_TIMEOUT:
                                continue
                            last_detected[name] = {"time": current_time}

                        confidence = 1 - \
                            min(face_recognition.face_distance(
                                known_encodings, encoding)) if known_encodings else 0
                        timestamp = datetime.datetime.now()
                        attendance = Attendance(
                            name=name,
                            time=timestamp,
                            confidence=confidence,
                            employee_id=emp.employee_id,
                            dob=emp.dob,
                            position=emp.position
                        )
                        db.add(attendance)
                        db.commit()
                        # Log debug
                        # print(f"Đã ghi vào database: {name}, {timestamp}, {confidence}")

                        # Lưu ảnh (tùy chọn)
                        img_path = os.path.join(
                            UPLOAD_DIR, f"{name}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
                        cv2.imwrite(img_path, frame)

                        info = f"ID: {emp.employee_id}, DOB: {emp.dob}, Position: {emp.position}, Confidence: {confidence:.2f}"
                        if src_cam != 0:
                            send_to_esp32(name, info)

                    # Vẽ khung và nhãn
                    cv2.rectangle(frame, (left, top),
                                    (right, bottom), (0, 255, 0), 2)
                    label = name + (f" ({emp.employee_id})" if emp else "")
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                with frame_lock:
                    output_frame = frame.copy()

            finally:
                db.close()

            sleep(0.05)

        cap.release()
        sleep(5)


if __name__ == "__main__":
    threading.Thread(target=process_stream, daemon=True).start()
    uvicorn.run(app, port=8001)
