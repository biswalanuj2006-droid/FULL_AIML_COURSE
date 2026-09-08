"""FastAPI core: app setup, auth helpers, auth routes, static serving."""
import os


def _load_env(path: str = ".env") -> None:
    """Tiny .env loader (KEY=VALUE lines) - no python-dotenv dependency."""
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_env(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
_load_env()

import bcrypt
import jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, User, get_engine, level_for_xp

DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./platform.db")
if DB_URL.startswith("postgres"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")

engine = get_engine(DB_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI(title="AI/ML Engineering Academy", version="1.0.0")

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:4173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS + ["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ASSETS = os.environ.get(
    "ASSETS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
if os.path.isdir(REPO_ASSETS):
    app.mount("/assets", StaticFiles(directory=REPO_ASSETS), name="assets")

# Serve the built SPA (single-service deploys: Railway/Render/Docker).
# NOTE: the catch-all route is registered at the BOTTOM of this file,
# after all API routers, so it can never shadow an API endpoint.
FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                 "frontend", "dist")))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------- auth helpers
from fastapi import Header  # noqa: E402


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except ValueError:
        return False


def make_token(user: User) -> str:
    return jwt.encode({"sub": str(user.id), "role": user.role},
                      JWT_SECRET, algorithm="HS256")


def current_user(authorization: str = Header(default=""),
                 db: Session = Depends(get_db)) -> User:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        payload = jwt.decode(authorization[7:], JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid token")
    user = db.get(User, int(payload["sub"]))
    if not user:
        raise HTTPException(401, "user not found")
    return user


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "admin only")
    return user


def add_xp(db: Session, user: User, amount: int) -> str:
    user.xp = max(0, user.xp + amount)
    user.level = level_for_xp(user.xp)
    return user.level


# ---------------------------------------------------------------------- models
class RegisterIn(BaseModel):
    email: str
    name: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str
    role: str
    xp: int
    level: str

    class Config:
        from_attributes = True


# ----------------------------------------------------------------------- auth
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/register", response_model=UserOut)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    if not body.email or "@" not in body.email:
        raise HTTPException(400, "valid email required")
    if len(body.password) < 8:
        raise HTTPException(400, "password must be at least 8 characters")
    if db.query(User).filter(func.lower(User.email) == body.email.lower()).first():
        raise HTTPException(409, "email already registered")
    user = User(email=body.email.strip(), name=body.name.strip() or body.email,
                password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        func.lower(User.email) == body.email.lower()).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "invalid email or password")
    return {"token": make_token(user), "user": UserOut.model_validate(user)}


@app.get("/api/auth/me", response_model=UserOut)
def me(user: User = Depends(current_user)):
    return user


# ------------------------------------------------------------------- routers
from . import chatbot, learning, admin, answer_engine  # noqa: E402

app.include_router(learning.router)
app.include_router(chatbot.router)
app.include_router(answer_engine.router)
app.include_router(admin.router)


# ------------------------------------------------------------------ SPA last
# Registered LAST so every /api route above wins; unknown GET paths fall
# through to index.html, which makes React Router deep links work.
@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    # serve real files from dist (js/css/favicon) if requested
    candidate = os.path.normpath(os.path.join(FRONTEND_DIST, full_path))
    if candidate.startswith(FRONTEND_DIST) and os.path.isfile(candidate):
        return FileResponse(candidate)
    index = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"detail": "frontend not built - run: cd platform/frontend && npm run build"}
