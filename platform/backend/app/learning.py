"""Learning routes: courses, lessons, exams, assignments, projects, enrollments."""
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .main import add_xp, current_user, get_db
from .models import (Assignment, Course, Enrollment, Exam, ExamAttempt,
                     Lesson, Project, Question, Submission, User)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------- output
def course_out(c: Course, db: Session = None) -> dict:
    d = {"id": c.id, "slug": c.slug, "title": c.title,
         "description": c.description, "order": c.order_index,
         "lessons": len(c.lessons), "exams": len(c.exams),
         "assignments": len(c.assignments), "projects": len(c.projects)}
    return d


# ---------------------------------------------------------------------- public
@router.get("/courses")
def list_courses(db: Session = Depends(get_db)):
    courses = db.query(Course).filter(Course.published).order_by(
        Course.order_index).all()
    return [course_out(c) for c in courses]


@router.get("/courses/{slug}")
def get_course(slug: str, db: Session = Depends(get_db)):
    c = db.query(Course).filter(Course.slug == slug).first()
    if not c:
        raise HTTPException(404, "course not found")
    return {
        **course_out(c),
        "lesson_list": [{"id": l.id, "title": l.title, "summary": l.summary,
                         "video": l.video, "diagram": l.diagram,
                         "order": l.order_index}
                        for l in sorted(c.lessons, key=lambda x: x.order_index)],
        "exam_ids": [e.id for e in c.exams],
        "assignment_ids": [a.id for a in c.assignments],
        "project_list": [{"id": p.id, "title": p.title,
                          "difficulty": p.difficulty,
                          "objective": p.objective, "tech": p.tech}
                         for p in c.projects],
    }


@router.get("/exams/{exam_id}")
def get_exam(exam_id: int, db: Session = Depends(get_db)):
    e = db.get(Exam, exam_id)
    if not e:
        raise HTTPException(404, "exam not found")
    return {"id": e.id, "course_id": e.course_id, "title": e.title,
            "pass_pct": e.pass_pct,
            "questions": [{"id": q.id, "kind": q.kind, "prompt": q.prompt,
                           "diagram": q.diagram, "options": q.options,
                           "xp": q.xp}
                          for q in e.questions]}


class SubmitIn(BaseModel):
    answers: dict  # question_id -> option index


@router.post("/exams/{exam_id}/submit")
def submit_exam(exam_id: int, body: SubmitIn,
                user: User = Depends(current_user),
                db: Session = Depends(get_db)):
    e = db.get(Exam, exam_id)
    if not e:
        raise HTTPException(404, "exam not found")
    total = len(e.questions)
    if total == 0:
        raise HTTPException(400, "exam has no questions")
    correct = 0
    xp_earned = 0
    review = []
    for q in e.questions:
        given = body.answers.get(str(q.id), body.answers.get(q.id))
        ok = given is not None and given == q.answer_index
        correct += ok
        xp_earned += q.xp if ok else 0
        review.append({"id": q.id, "prompt": q.prompt, "correct": ok,
                       "answer_index": q.answer_index,
                       "explanation": q.explanation})
    score = round(100.0 * correct / total, 1)
    passed = score >= e.pass_pct
    db.add(ExamAttempt(user_id=user.id, exam_id=e.id, score_pct=score,
                       passed=passed, answers=[str(k) for k in body.answers]))
    if passed:
        add_xp(db, user, xp_earned)
    else:
        add_xp(db, user, max(1, xp_earned // 5))
    db.commit()
    return {"score_pct": score, "passed": passed, "xp_earned": xp_earned if passed else max(1, xp_earned // 5),
            "review": review}


@router.get("/assignments/{aid}")
def get_assignment(aid: int, db: Session = Depends(get_db)):
    a = db.get(Assignment, aid)
    if not a:
        raise HTTPException(404, "assignment not found")
    return {"id": a.id, "course_id": a.course_id, "title": a.title,
            "brief": a.brief, "starter_code": a.starter_code}


class CodeIn(BaseModel):
    code: str


@router.post("/assignments/{aid}/submit")
def submit_assignment(aid: int, body: CodeIn,
                      user: User = Depends(current_user),
                      db: Session = Depends(get_db)):
    a = db.get(Assignment, aid)
    if not a:
        raise HTTPException(404, "assignment not found")
    if not body.code.strip():
        raise HTTPException(400, "code is empty")
    sub = Submission(user_id=user.id, assignment_id=aid, code=body.code)
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return {"id": sub.id, "status": sub.status,
            "message": "submitted for AI review"}


@router.get("/my/progress")
def my_progress(user: User = Depends(current_user), db: Session = Depends(get_db)):
    enr = db.query(Enrollment).filter(Enrollment.user_id == user.id).all()
    attempts = db.query(ExamAttempt).filter(ExamAttempt.user_id == user.id).all()
    return {"xp": user.xp, "level": user.level,
            "enrollments": [{"course_id": e.course_id,
                             "progress_pct": e.progress_pct,
                             "completed": e.completed_at is not None,
                             "certificate_id": e.certificate_id} for e in enr],
            "attempts": [{"exam_id": a.exam_id, "score_pct": a.score_pct,
                          "passed": a.passed} for a in attempts]}


class EnrollIn(BaseModel):
    slug: str


@router.post("/enroll")
def enroll(body: EnrollIn, user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    c = db.query(Course).filter(Course.slug == body.slug).first()
    if not c:
        raise HTTPException(404, "course not found")
    e = db.query(Enrollment).filter_by(user_id=user.id, course_id=c.id).first()
    if not e:
        e = Enrollment(user_id=user.id, course_id=c.id)
        db.add(e)
        db.commit()
    return {"enrolled": True, "course": c.slug}


@router.get("/leaderboard")
def leaderboard(db: Session = Depends(get_db), user: User = Depends(current_user)):
    top = db.query(User).order_by(User.xp.desc()).limit(20).all()
    return [{"name": u.name, "xp": u.xp, "level": u.level} for u in top]
