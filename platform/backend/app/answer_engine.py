"""ML answer engine: retrieval-based Q&A over ALL course content.

A real mini-RAG with zero external deps:
    question -> tokenize -> TF-IDF cosine retrieval over
      [lesson titles+summaries, exam-question explanations, KB entries]
    -> MMR-diversified top passages -> composed answer + sources + confidence
(LLM refinement, when an API key is present, is layered on top by chatbot.py)
"""
import math
import re
from collections import Counter

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .main import current_user, get_db
from .models import Course, Exam, Lesson, Question
from .chatbot import detect_courses, KNOWLEDGE

router = APIRouter(prefix="/api/answer")


# ------------------------------------------------------------- vectorization
_TOKEN = re.compile(r"[a-z0-9+#-]+")
_STOP = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "what",
         "why", "how", "does", "do", "for", "on", "with", "it", "this", "that",
         "be", "by", "as", "at", "can", "from", "which", "when", "if", "use",
         "using", "between", "vs", "difference", "explain", "me", "my", "i"}


def tokenize(text: str):
    return [t for t in _TOKEN.findall(text.lower())
            if t not in _STOP and len(t) > 1]


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    if len(b) < len(a):
        a, b = b, a
    return math.sqrt(sum(v * b.get(k, 0.0) for k, v in a.items()))


def _mmr_select(scored: list, k: int, lam: float = 0.72) -> list:
    """Maximal Marginal Relevance over (passage, score) pairs, score-desc input."""
    if not scored:
        return []
    chosen = [scored[0]]
    while len(chosen) < k and len(chosen) < len(scored):
        best, best_val = None, -1e9
        for cand in scored:
            if cand in chosen:
                continue
            sim = max(_cosine(cand[0]["tfidf"], c[0]["tfidf"])
                      for c in chosen)
            val = lam * cand[1] - (1 - lam) * sim
            if val > best_val:
                best, best_val = cand, val
        if best is None:
            break
        chosen.append(best)
    return chosen


class PassageIndex:
    """TF-IDF index over passages; rebuilt lazily when corpus changes."""

    def __init__(self):
        self._passages = []
        self._idf = {}
        self._sig = None

    def build(self, passages: list):
        self._passages = passages
        df = Counter()
        for p in passages:
            df.update(set(p["tokens"]))
        n = max(1, len(passages))
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}
        for p in passages:
            tf = Counter(p["tokens"])
            self._vectorize_inplace(p, tf)
        self._sig = (len(passages), sum(len(p["tokens"]) for p in passages))

    def _vectorize_inplace(self, p, tf):
        v = {}
        for t, c in tf.items():
            w = (1 + math.log(c)) * self._idf.get(t, 0.0)
            if w > 0:
                v[t] = w
        norm = math.sqrt(sum(v * v for v in v.values())) or 1.0
        p["tfidf"] = {t: w / norm for t, w in v.items()}

    def search(self, query: str, k: int = 3):
        if not self._passages:
            return []
        tf = Counter(tokenize(query))
        qv = {}
        for t, c in tf.items():
            w = (1 + math.log(c)) * self._idf.get(t, 0.0)
            if w > 0:
                qv[t] = w
        nq = math.sqrt(sum(w * w for w in qv.values())) or 1.0
        qv = {t: w / nq for t, w in qv.items()}
        hits = []
        for p in self._passages:
            s = sum(w * p["tfidf"].get(t, 0.0) for t, w in qv.items())
            if s > 0:
                hits.append((p, s))
        hits.sort(key=lambda hs: -hs[1])
        top = hits[:max(k * 3, 6)]
        diversified = _mmr_select(top, k)
        return [{"passage": p, "score": round(s, 4)} for p, s in diversified]

    def stale(self, passages: list) -> bool:
        sig = (len(passages), sum(len(p["tokens"]) for p in passages))
        return sig != self._sig


ENGINE = PassageIndex()


def _collect_passages(db: Session) -> list:
    passages = []
    # 1) lesson summaries (and titles) per course
    rows = (db.query(Lesson, Course)
            .join(Course, Lesson.course_id == Course.id)
            .filter(Course.published).all())
    for lesson, course in rows:
        text = f"{course.title}: {lesson.title}. {lesson.summary}"
        passages.append({
            "kind": "lesson", "course": course.slug,
            "course_title": course.title, "title": lesson.title,
            "text": text, "tokens": tokenize(text), "diagram": lesson.diagram,
        })
    # 2) exam question explanations (gold study material)
    rows = (db.query(Question, Course)
            .join(Exam, Question.exam_id == Exam.id)
            .join(Course, Exam.course_id == Course.id).all())
    for q, course in rows:
        if not q.explanation:
            continue
        text = f"{q.prompt} -> {q.explanation}"
        passages.append({
            "kind": "explanation", "course": course.slug,
            "course_title": course.title, "title": q.prompt[:80],
            "text": text, "tokens": tokenize(text), "diagram": q.diagram,
        })
    # 3) built-in knowledge base
    # 3) built-in knowledge base
    for key, text in KNOWLEDGE.items():
        courses_for_key = detect_courses(key)
        passages.append({
            "kind": "kb",
            "course": courses_for_key[0] if courses_for_key else "general",
            "course_title": "Knowledge base", "title": key.title(),
            "text": text, "tokens": tokenize(text), "diagram": "",
        })
    return passages


def ensure_index(db: Session):
    passages = _collect_passages(db)
    if ENGINE.stale(passages):
        ENGINE.build(passages)


def compose_answer(results: list, query: str) -> dict:
    if not results:
        return {
            "answer": "I could not find this in the course corpus yet. Try "
                      "rephrasing, or ask about: RAG, attention, overfitting, "
                      "BM25, agents, KV cache, drift...",
            "confidence": 0.0, "sources": [], "kind": "miss",
        }
    best = results[0]
    lines = [f"**{best['passage']['title']}** — {best['passage']['text']}"]
    for r in results[1:]:
        p = r["passage"]
        lines.append(f"\n**Related — {p['title']}** ({p['course_title']}): "
                     f"{p['text']}")
    confidence = min(1.0, best["score"] * (1 + 0.25 * (len(results) - 1)))
    return {
        "answer": "\n".join(lines),
        "confidence": round(confidence, 3),
        "kind": "hit",
        "sources": [{"course": p["course"], "course_title": p["course_title"],
                     "title": p["title"], "kind": p["kind"],
                     "diagram": p.get("diagram", "")}
                    for p, _ in [(r["passage"], r["score"]) for r in results]],
    }


class AskIn(BaseModel):
    question: str
    course: str = ""


@router.post("")
def ask(body: AskIn, user=Depends(current_user), db: Session = Depends(get_db)):
    if not body.question.strip():
        return {"answer": "Ask me anything about the courses.", "confidence": 0,
                "sources": [], "kind": "empty"}
    ensure_index(db)
    q = body.question
    if body.course:
        q = f"{body.course} {q}"
    results = ENGINE.search(q, k=3)
    out = compose_answer(results, body.question)
    out["suggested_courses"] = detect_courses(body.question)[:3] or \
        ([results[0]["passage"]["course"]] if results else [])
    return out


@router.get("/health")
def answer_health():
    return {"indexed": len(ENGINE._passages), "engine": "tfidf-mini-rag"}
