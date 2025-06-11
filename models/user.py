from sqlalchemy import Column, String
from database import Base


class Employee(Base):
    __tablename__ = "employees"
    employee_id = Column(String(50), primary_key=True)
    name = Column(String(100))
    dob = Column(String(20))
    position = Column(String(100))



