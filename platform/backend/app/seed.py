"""Seed the database: users, courses, lessons, exams, assignments, projects."""
from .main import SessionLocal, hash_password
from .models import (Assignment, Course, Enrollment, Exam, Lesson, Project,
                     Question, User, level_for_xp)
from .seed_catalog import COURSES
from .seed_exams import EXAMS
from .seed_projects import PROJECTS

ASSIGNMENTS = {
    "python": [("Implement `cache_result` decorator", "Write a decorator that caches by args and exposes .hits.", "def cache_result(fn):\n    # your code\n    ...")],
    "numpy": [("Pairwise distances, no loops", "Given X (n,d), return the (n,n) distance matrix using broadcasting only.", "def pairwise(X):\n    ...")],
    "pandas": [("Monthly revenue table", "From orders.csv build a month x category revenue pivot with totals.", "import pandas as pd\ndef monthly(df):\n    ...")],
    "mathematics": [("Softmax with stability", "Implement numerically stable softmax and unit-test extremes.", "import numpy as np\ndef softmax(z):\n    ...")],
    "machine-learning": [("Leak-proof pipeline", "Build a sklearn Pipeline with imputer+scaler+model and cross-validate correctly.", "from sklearn.pipeline import Pipeline\n...")],
    "deep-learning": [("Training loop with early stopping", "Complete the loop: zero_grad/forward/backward/step plus val early-stop.", "for epoch in range(E):\n    ...")],
    "nlp": [("Tokenizer comparison", "Compare word vs BPE token counts on 20 sentences; report compression.", "...")],
    "transformers": [("Scaled dot-product attention", "Implement attention with masking from tensors only; verify shapes.", "def attention(Q,K,V,mask=None):\n    ...")],
    "llm-engineering": [("Sampling utilities", "Implement top-k + top-p sampling over logits with tests.", "def sample(logits, k=None, p=None):\n    ...")],
    "rag": [("Build a hybrid retriever", "Implement BM25+dense RRF retrieval over a small corpus; report hit@3.", "...")],
    "langchain-langgraph": [("Two-node graph", "Create a LangGraph with a grade->rewrite conditional loop.", "...")],
    "ai-agents": [("Tool-calling loop", "Implement a ReAct loop with a calculator tool and max-step cap.", "...")],
    "mlops": [("Track a sweep", "Log 6 runs (lr x batch) to MLflow and select the best by val AUC.", "...")],
    "ai-security": [("Injection test set", "Write 10 direct/indirect injection payloads and a checker harness.", "...")],
}


def run_seed(reset: bool = False):
    db = SessionLocal()
    try:
        if reset:
            for t in (Question, Exam, Assignment, Project, Lesson, Enrollment,
                      Course, User):
                db.query(t).delete()
            db.commit()
        if db.query(Course).count() > 0:
            return {"seeded": False, "reason": "already populated"}

        # users ------------------------------------------------------------
        admin = User(email="admin@aiml.dev", name="Admin",
                     password_hash=hash_password("admin12345"), role="admin",
                     xp=5000, level="grandmaster")
        demo = User(email="demo@aiml.dev", name="Demo Learner",
                    password_hash=hash_password("demo12345"))
        db.add_all([admin, demo])
        db.flush()

        # courses ----------------------------------------------------------
        exams_by_slug = {}
        for slug, title, desc, order, lessons in COURSES:
            c = Course(slug=slug, title=title, description=desc,
                       order_index=order)
            db.add(c)
            db.flush()
            for i, (lt, ls, video, diagram) in enumerate(lessons, 1):
                db.add(Lesson(course_id=c.id, title=lt, summary=ls,
                              video=video, diagram=diagram, order_index=i))
            # exam ---------------------------------------------------------
            e = Exam(course_id=c.id, title=f"{title} - Final Exam", pass_pct=70)
            db.add(e)
            db.flush()
            for i, (prompt, opts, ai, expl, diagram) in enumerate(
                    EXAMS.get(slug, []), 1):
                db.add(Question(exam_id=e.id, prompt=prompt, options=opts,
                                answer_index=ai, explanation=expl,
                                diagram=diagram,
                                kind="visual" if diagram else "mcq",
                                xp=10))
            exams_by_slug[slug] = e.id
            # assignment ---------------------------------------------------
            for i, (at, ab, ac) in enumerate(ASSIGNMENTS.get(slug, []), 1):
                db.add(Assignment(course_id=c.id, title=at, brief=ab,
                                  starter_code=ac, order_index=i))
        db.flush()

        # projects ---------------------------------------------------------
        course_rows = {c.slug: c.id for c in db.query(Course).all()}
        per_course = {}
        for slug, ptitle, diff, obj, tech in PROJECTS:
            idx = per_course.get(slug, 0) + 1
            per_course[slug] = idx
            db.add(Project(course_id=course_rows[slug], title=ptitle,
                           difficulty=diff, objective=obj, tech=tech,
                           order_index=idx))

        # demo enrollment ----------------------------------------------------
        rag = db.query(Course).filter_by(slug="rag").first()
        db.add(Enrollment(user_id=demo.id, course_id=rag.id, progress_pct=0.0))
        db.commit()
        return {"seeded": True, "courses": len(COURSES),
                "questions": sum(len(v) for v in EXAMS.values()),
                "projects": len(PROJECTS)}
    finally:
        db.close()


if __name__ == "__main__":
    print(run_seed())
