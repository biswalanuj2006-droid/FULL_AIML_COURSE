"""Database models: users, courses, lessons, exams, assignments, projects, enrollments."""
import datetime

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text, UniqueConstraint,
                        create_engine)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _now():
    return datetime.datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(120), nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), default="student")  # student | admin
    xp = Column(Integer, default=0)
    level = Column(String(20), default="copper")
    created_at = Column(DateTime, default=_now)

    enrollments = relationship("Enrollment", back_populates="user",
                               cascade="all, delete-orphan")
    attempts = relationship("ExamAttempt", back_populates="user",
                            cascade="all, delete-orphan")
    submissions = relationship("Submission", back_populates="user",
                               cascade="all, delete-orphan")


class Course(Base):
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True)
    slug = Column(String(80), unique=True, nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, default="")
    order_index = Column(Integer, default=0)
    published = Column(Boolean, default=True)

    lessons = relationship("Lesson", back_populates="course",
                           cascade="all, delete-orphan")
    exams = relationship("Exam", back_populates="course",
                         cascade="all, delete-orphan")
    assignments = relationship("Assignment", back_populates="course",
                               cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="course",
                            cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="course",
                               cascade="all, delete-orphan")


class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    summary = Column(Text, default="")
    video = Column(String(200), default="")      # e.g. "Videos/attention_flow.mp4"
    diagram = Column(String(200), default="")    # e.g. "Images/transformers/attention_mask.png"
    order_index = Column(Integer, default=0)

    course = relationship("Course", back_populates="lessons")


class Exam(Base):
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    pass_pct = Column(Integer, default=70)

    questions = relationship("Question", back_populates="exam",
                             cascade="all, delete-orphan")
    course = relationship("Course", back_populates="exams")


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    kind = Column(String(20), default="mcq")     # mcq | visual | code
    prompt = Column(Text, nullable=False)
    diagram = Column(String(200), default="")    # image path for visual questions
    options = Column(JSON, nullable=False)       # list[str]
    answer_index = Column(Integer, nullable=False)
    explanation = Column(Text, default="")
    xp = Column(Integer, default=10)

    exam = relationship("Exam", back_populates="questions")


class ExamAttempt(Base):
    __tablename__ = "exam_attempts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    score_pct = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    answers = Column(JSON, default=list)
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="attempts")
    exam = relationship("Exam")


class Assignment(Base):
    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    brief = Column(Text, default="")
    starter_code = Column(Text, default="")
    order_index = Column(Integer, default=0)

    course = relationship("Course", back_populates="assignments")
    submissions = relationship("Submission", back_populates="assignment",
                               cascade="all, delete-orphan")


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    assignment_id = Column(Integer, ForeignKey("assignments.id"), nullable=False)
    code = Column(Text, nullable=False)
    status = Column(String(20), default="pending")   # pending | passed | feedback
    feedback = Column(Text, default="")
    created_at = Column(DateTime, default=_now)

    user = relationship("User", back_populates="submissions")
    assignment = relationship("Assignment", back_populates="submissions")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    title = Column(String(200), nullable=False)
    difficulty = Column(String(10), default="E")      # E D C B A S S+ S++
    objective = Column(Text, default="")
    tech = Column(JSON, default=list)
    order_index = Column(Integer, default=0)

    course = relationship("Course", back_populates="projects")


class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("user_id", "course_id"),)

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    progress_pct = Column(Float, default=0.0)
    completed_at = Column(DateTime, nullable=True)
    certificate_id = Column(String(40), nullable=True)

    user = relationship("User", back_populates="enrollments")
    course = relationship("Course", back_populates="enrollments")


LEVELS = ["copper", "silver", "gold", "platinum", "grandmaster"]
LEVEL_XP = {"copper": 0, "silver": 300, "gold": 800, "platinum": 1800,
            "grandmaster": 3500}


def level_for_xp(xp: int) -> str:
    current = "copper"
    for lv in LEVELS:
        if xp >= LEVEL_XP[lv]:
            current = lv
    return current


def get_engine(db_url: str):
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args)
