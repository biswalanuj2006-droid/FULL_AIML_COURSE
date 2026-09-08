"""Admin routes: stats, users, courses, question bank management."""
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException

from .main import current_user, get_db
from .models import (Course, Enrollment, Exam, ExamAttempt, Project,
                     Question, Submission, User)

router = APIRouter(prefix="/api/admin", dependencies=[Depends(current_user)])


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "admin only")
    return user


# ---------------------------------------------------------------------- stats
@router.get("/stats")
def stats(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return {
        "users": db.query(func.count(User.id)).scalar(),
        "courses": db.query(func.count(Course.id)).scalar(),
        "questions": db.query(func.count(Question.id)).scalar(),
        "attempts": db.query(func.count(ExamAttempt.id)).scalar(),
        "submissions": db.query(func.count(Submission.id)).scalar(),
        "enrollments": db.query(func.count(Enrollment.id)).scalar(),
    }


# ---------------------------------------------------------------------- users
@router.get("/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return [{"id": u.id, "email": u.email, "name": u.name, "role": u.role,
             "xp": u.xp, "level": u.level} for u in
            db.query(User).order_by(User.id).all()]


class RoleIn(BaseModel):
    role: str


@router.patch("/users/{uid}/role")
def set_role(uid: int, body: RoleIn, db: Session = Depends(get_db),
             _: User = Depends(require_admin)):
    if body.role not in ("student", "admin"):
        raise HTTPException(400, "role must be student or admin")
    u = db.get(User, uid)
    if not u:
        raise HTTPException(404, "user not found")
    u.role = body.role
    db.commit()
    return {"id": u.id, "role": u.role}


# -------------------------------------------------------------------- courses
class CourseIn(BaseModel):
    title: str
    slug: str
    description: str = ""


@router.post("/courses")
def create_course(body: CourseIn, db: Session = Depends(get_db),
                  _: User = Depends(require_admin)):
    if db.query(Course).filter(Course.slug == body.slug).first():
        raise HTTPException(409, "slug exists")
    nxt = db.query(func.coalesce(func.max(Course.order_index), 0)).scalar() + 1
    c = Course(title=body.title, slug=body.slug,
               description=body.description, order_index=nxt)
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "slug": c.slug}


@router.delete("/courses/{cid}")
def delete_course(cid: int, db: Session = Depends(get_db),
                  _: User = Depends(require_admin)):
    c = db.get(Course, cid)
    if not c:
        raise HTTPException(404, "course not found")
    db.delete(c)
    db.commit()
    return {"deleted": cid}


# ------------------------------------------------------------------ questions
class QuestionIn(BaseModel):
    exam_id: int
    kind: str = "mcq"
    prompt: str
    options: list
    answer_index: int
    explanation: str = ""
    diagram: str = ""


@router.post("/questions")
def add_question(body: QuestionIn, db: Session = Depends(get_db),
                 _: User = Depends(require_admin)):
    e = db.get(Exam, body.exam_id)
    if not e:
        raise HTTPException(404, "exam not found")
    if body.kind not in ("mcq", "visual", "code"):
        raise HTTPException(400, "kind must be mcq | visual | code")
    if not body.options or len(body.options) < 2:
        raise HTTPException(400, "at least 2 options required")
    if not 0 <= body.answer_index < len(body.options):
        raise HTTPException(400, "answer_index out of range")
    q = Question(exam_id=body.exam_id, kind=body.kind, prompt=body.prompt,
                 options=body.options, answer_index=body.answer_index,
                 explanation=body.explanation, diagram=body.diagram)
    db.add(q)
    db.commit()
    db.refresh(q)
    return {"id": q.id}


@router.delete("/questions/{qid}")
def delete_question(qid: int, db: Session = Depends(get_db),
                    _: User = Depends(require_admin)):
    q = db.get(Question, qid)
    if not q:
        raise HTTPException(404, "question not found")
    db.delete(q)
    db.commit()
    return {"deleted": qid}


# ----------------------------------------------------------------- review queue
@router.get("/submissions")
def pending_submissions(db: Session = Depends(get_db),
                        _: User = Depends(require_admin)):
    subs = (db.query(Submission).filter(Submission.status == "pending")
            .order_by(Submission.id.desc()).limit(50).all())
    return [{"id": s.id, "user_id": s.user_id,
             "assignment_id": s.assignment_id,
             "assignment": s.assignment.title if s.assignment else "?",
             "code": s.code[:1500], "status": s.status} for s in subs]


class ReviewIn(BaseModel):
    status: str
    feedback: str = ""


@router.patch("/submissions/{sid}")
def review_submission(sid: int, body: ReviewIn, db: Session = Depends(get_db),
                      _: User = Depends(require_admin)):
    s = db.get(Submission, sid)
    if not s:
        raise HTTPException(404, "submission not found")
    if body.status not in ("passed", "feedback"):
        raise HTTPException(400, "status must be passed | feedback")
    s.status = body.status
    s.feedback = body.feedback
    db.commit()
    return {"id": s.id, "status": s.status}
