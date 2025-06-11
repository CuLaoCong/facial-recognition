import pyodbc

def get_connection():
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost\\SQLEXPRESS;"  # ⚠️ Đổi thành tên server của bạn
            "DATABASE=FaceAttendance;"
            "UID=sa;"                        # ⚠️ Đổi nếu bạn không dùng user 'sa'
            "PWD=your_password"             # ⚠️ Đổi thành mật khẩu thật
        )
        print("[✅] Kết nối thành công tới SQL Server")
        return conn
    except Exception as e:
        print("[❌] Kết nối thất bại:", e)
        return None
