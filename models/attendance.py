
from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
from database import Base
from pydantic import BaseModel

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    time = Column(DateTime)
    confidence = Column(Float)
    employee_id = Column(String(50), ForeignKey("employees.employee_id"))
    dob = Column(String(20))
    position = Column(String(100))


class AttendanceRecord(BaseModel):
    id: int
    name: str
    time: str
    confidence: float
    employee_id: str
    dob: str
    position: str
