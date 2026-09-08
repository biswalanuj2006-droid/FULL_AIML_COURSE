# AI/ML Engineering Academy — full-stack learning platform

A complete course platform built on top of the AI/ML engineering curriculum in
this repo: **React + Vite + Tailwind CSS** frontend, **FastAPI + SQLAlchemy**
backend, with an **ML answer engine** that answers questions across ALL courses.

## Features

- 📚 **14 courses** with lessons wired to the repo's 89 diagrams and 12 videos
- 🎓 **Exam after every course** — MCQ + visual diagram questions, instant
  grading, per-question explanations, retakes
- 🤖 **AI answer engine for every course** — a mini-RAG (TF-IDF retrieval +
  MMR diversification) over lessons, explanations and a knowledge base;
  upgrades to a real LLM automatically when an API key is present
- 💬 **AI tutor chatbot** on every page — explains concepts, generates quizzes,
  routes to the right course (works offline, no key needed)
- 📝 **Assignments** with starter code + submission/review workflow
- 🚀 **50 real-world projects** tagged E → S++ difficulty
- 🏅 **Levels: Copper → Silver → Gold → Platinum → Grandmaster** (XP-driven)
- 🛠 **Admin panel** — stats, user roles, question bank, submission review
- 🏆 **Leaderboard**, enrollments, progress tracking

## Quickstart (local)

```bash
# backend
cd platform/backend
pip install -r requirements.txt
python seed_run.py
python run.py                 # http://localhost:8000

# frontend (second terminal)
cd platform/frontend
npm install
npm run dev                   # http://localhost:5173 (proxies /api)
```

Production single-service (serves SPA + API + assets on one port):

```bash
cd platform/frontend && npm run build && cd ../..
cd platform/backend && python seed_run.py && python run.py
```

## Deploy

See **[DEPLOY.md](../DEPLOY.md)** — Railway (one click), Render, Fly.io,
Heroku, Docker/Compose, Vercel/Netlify.

## Architecture

```
platform/
├── backend/               FastAPI
│   ├── app/main.py        app + auth + SPA serving (catch-all registered LAST)
│   ├── app/learning.py    courses, exams (auto-grading), assignments, XP
│   ├── app/answer_engine.py  mini-RAG answer engine (TF-IDF + MMR)
│   ├── app/chatbot.py     tutor chatbot (built-in knowledge + optional LLM)
│   ├── app/admin.py       admin panel APIs
│   ├── app/seed_*.py      course catalog, exams, 50 projects
│   └── tests/test_api.py  14 end-to-end tests
└── frontend/              React 18 + Vite + Tailwind
    └── src/
        ├── pages/         Home, Courses, CourseDetail, Exam, Assignment,
        │                  Dashboard, Leaderboard, Admin, Gallery, Login
        └── components/    Navbar, ChatWidget, AskAI
```

## Demo accounts

| role | email | password |
|---|---|---|
| admin | admin@aiml.dev | admin12345 |
| learner | demo@aiml.dev | demo12345 |
