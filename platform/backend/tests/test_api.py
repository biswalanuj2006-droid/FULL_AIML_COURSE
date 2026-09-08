"""End-to-end API tests: auth, courses, exams, ML answer engine, chat, admin."""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client):
    email = f"pytest_{uuid.uuid4().hex[:8]}@x.com"
    r = client.post("/api/auth/register",
                    json={"email": email, "name": "Pytest", "password": "password1"})
    assert r.status_code == 200, r.text
    r = client.post("/api/auth/login", json={"email": email, "password": "password1"})
    token = r.json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_register_validation(client):
    assert client.post("/api/auth/register", json={
        "email": "bad", "name": "x", "password": "password1"}).status_code == 400
    assert client.post("/api/auth/register", json={
        "email": "ok@x.com", "name": "x", "password": "short"}).status_code == 400


def test_courses_seeded(client):
    courses = client.get("/api/courses").json()
    assert len(courses) >= 14
    slugs = {c["slug"] for c in courses}
    assert {"python", "machine-learning", "transformers", "rag"} <= slugs


def test_course_detail(client):
    d = client.get("/api/courses/rag").json()
    assert d["lesson_list"]
    assert d["exam_ids"]
    assert d["project_list"]
    assert any(l["diagram"] for l in d["lesson_list"])


def test_answer_engine_all_courses(client, auth):
    cases = {
        "why does overfitting happen": "machine-learning",
        "what is attention": "transformers",
        "how does hybrid retrieval work": "rag",
        "what is prompt injection": "ai-security",
        "explain kv cache": "llm-engineering",
        "what is drift": "mlops",
    }
    for q, expect in cases.items():
        r = client.post("/api/answer", json={"question": q}, headers=auth)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["confidence"] > 0.2, (q, body)
        srcs = {s["course"] for s in body["sources"]}
        assert expect in srcs, (q, srcs)


def test_answer_engine_miss_is_graceful(client, auth):
    r = client.post("/api/answer", json={"question": "zzzz qqqq xyz"}, headers=auth)
    body = r.json()
    assert body["kind"] in ("miss", "hit")
    assert "answer" in body


def test_exam_flow_and_xp(client, auth):
    d = client.get("/api/courses/transformers").json()
    exam_id = d["exam_ids"][0]
    exam = client.get(f"/api/exams/{exam_id}").json()
    assert exam["questions"]
    answers = {str(q["id"]): q.get("diagram") and 1 or 1 for q in exam["questions"]}
    r = client.post(f"/api/exams/{exam_id}/submit",
                    json={"answers": answers}, headers=auth)
    body = r.json()
    assert body["score_pct"] >= 0
    assert body["review"] and body["review"][0]["explanation"]
    me = client.get("/api/auth/me", headers=auth).json()
    assert me["xp"] > 0
    assert me["level"] in ("copper", "silver", "gold", "platinum", "grandmaster")


def test_assignment_submit(client, auth):
    d = client.get("/api/courses/python").json()
    aid = d["assignment_ids"][0]
    a = client.get(f"/api/assignments/{aid}").json()
    assert a["starter_code"]
    r = client.post(f"/api/assignments/{aid}/submit",
                    json={"code": "def cache_result(fn):\n    ..."}, headers=auth)
    assert r.status_code == 200


def test_enroll_and_progress(client, auth):
    assert client.post("/api/enroll", json={"slug": "rag"}, headers=auth).json()["enrolled"]
    prog = client.get("/api/my/progress", headers=auth).json()
    assert prog["enrollments"]
    assert prog["xp"] >= 0


def test_chat_and_quiz(client, auth):
    r = client.post("/api/chat", json={"message": "explain attention"}, headers=auth)
    body = r.json()
    assert body["reply"] and body["source"] in ("llm", "builtin")
    q = client.get("/api/chat/quiz/rag?n=3", headers=auth).json()
    assert 1 <= len(q["questions"]) <= 3


def test_admin_requires_role(client, auth):
    assert client.get("/api/admin/stats", headers=auth).status_code == 403
    r = client.post("/api/auth/login",
                    json={"email": "admin@aiml.dev", "password": "admin12345"})
    admin = {"Authorization": f"Bearer {r.json()['token']}"}
    stats = client.get("/api/admin/stats", headers=admin).json()
    assert stats["courses"] >= 14 and stats["questions"] >= 40
    assert client.get("/api/admin/users", headers=admin).status_code == 200


def test_leaderboard(client, auth):
    rows = client.get("/api/leaderboard", headers=auth).json()
    assert isinstance(rows, list)


def test_spa_deep_link(client):
    """The backend job builds the frontend BEFORE pytest, so the real SPA
    must be served for deep links. No skip: production behavior is verified."""
    r = client.get("/courses/rag")
    assert r.status_code == 200
    assert b'id="root"' in r.content


def test_visual_question_assets_exist(client):
    """Every visual question's diagram must be a servable asset file."""
    import os
    from app.main import REPO_ASSETS
    found = 0
    for c in client.get("/api/courses").json():
        d = client.get(f"/api/courses/{c['slug']}").json()
        for eid in d["exam_ids"]:
            for q in client.get(f"/api/exams/{eid}").json()["questions"]:
                if q["diagram"]:
                    p = os.path.join(REPO_ASSETS, q["diagram"].replace("/", os.sep))
                    assert os.path.isfile(p), p
                    found += 1
    assert found >= 15
