"""AI chatbot: tutor + examiner + course router.

Works with NO API key (built-in knowledge + course routing + quiz pool).
If any of OPENROUTER_API_KEY / GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY
is set, answers are upgraded through that LLM automatically.
"""
import os
import random

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .main import current_user, get_db
from .models import Course, Question

router = APIRouter(prefix="/api/chat")

# --------------------------------------------------------------- course router
COURSE_KEYWORDS = {
    "python": ["python", "decorator", "generator", "async", "oop", "typing"],
    "numpy": ["numpy", "ndarray", "broadcast", "vectoriz"],
    "pandas": ["pandas", "dataframe", "groupby", "merge", "csv"],
    "mathematics": ["linear algebra", "eigen", "gradient", "derivative",
                    "probability", "matrix", "svd"],
    "machine-learning": ["regression", "classification", "random forest",
                         "xgboost", "clustering", "svm", "k-means", "overfit",
                         "cross-validation", "svm"],
    "deep-learning": ["neural", "backprop", "activation", "cnn", "dropout",
                      "batch norm", "pytorch", "training loop", "autograd"],
    "nlp": ["tokeniz", "word2vec", "ner", "sentiment", "tf-idf"],
    "transformers": ["attention", "transformer", "self-attention", "rope",
                     "kv cache", "gqa", "mha"],
    "llm-engineering": ["llm", "prompt", "temperature", "top-p", "hallucin",
                        "context window", "quantiz", "lora", "fine-tun"],
    "rag": ["rag", "retrieval", "chunk", "embedding", "vector db", "bm25",
            "rerank", "hybrid search", "citation"],
    "langchain-langgraph": ["langchain", "langgraph", "lcel", "runnable",
                            "agent state", "human-in-the-loop"],
    "ai-agents": ["agent", "react", "tool call", "multi-agent", "planner",
                  "memory"],
    "mlops": ["mlops", "mlflow", "drift", "deployment", "docker", "registry",
              "monitoring"],
    "ai-security": ["prompt injection", "jailbreak", "poisoning", "security",
                    "pii", "tenant isolation"],
}

KNOWLEDGE = {
    "rag": "RAG = retrieval-augmented generation: chunk documents, embed them, "
           "retrieve top-k for the query (hybrid BM25 + dense beats either alone), "
           "rerank, then generate a grounded answer with citations. Evaluate "
           "retrieval (hit@k, MRR) SEPARATELY from generation (faithfulness).",
    "attention": "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V. Each query "
                 "compares against all keys; softmax turns scores into weights; "
                 "the weighted sum of values is the output. Causal masking hides "
                 "future positions; GQA shares K/V heads to shrink the KV cache.",
    "overfitting": "Overfitting = train error keeps dropping while validation "
                   "error rises. Fix with more data, augmentation, regularization "
                   "(L2, dropout), early stopping, or a smaller model. Diagnose "
                   "by plotting both curves against capacity/epochs.",
    "gradient descent": "GD iterates w <- w - lr * grad(L). Momentum adds "
                        "velocity (v = b*v - lr*g) to smooth the path; Adam adds "
                        "per-parameter adaptive rates. If loss explodes: lower "
                        "lr; if it plateaus: schedule or warmup.",
    "temperature": "Temperature divides logits before softmax: P = softmax(z/T). "
                   "T->0 is near-deterministic (good for RAG/exams); T>1 flattens "
                   "the distribution (creative, riskier). Temperature controls "
                   "GENERATION, never retrieval.",
    "bm25": "BM25 scores lexical matches: IDF-weighted term frequency with "
            "saturation (k1 controls saturation) and document-length "
            "normalization (b). Still strong when query shares corpus "
            "vocabulary - hybrid with dense for robustness.",
    "langgraph": "LangGraph models workflows as state machines: nodes are "
                 "functions that read/write a shared State, edges route "
                 "transitions, checkpointer persists every super-step, and "
                 "interrupt() gives human-in-the-loop approval.",
    "transformer": "A transformer block = (multi-head attention -> add&norm -> "
                   "FFN -> add&norm) x N. Decoder-only stacks add causal masking. "
                   "Position info comes from RoPE/ALiBi rather than absolute "
                   "encodings in modern LLMs.",
}

GREETINGS = {"hi", "hello", "hey", "yo", "namaste", "hola"}

PROVIDERS = [
    ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1/chat/completions",
     "openrouter/auto", "Authorization"),
    ("GROQ_API_KEY", "https://api.groq.com/openai/v1/chat/completions",
     "llama-3.3-70b-versatile", "Authorization"),
    ("GOOGLE_API_KEY",
     "https://generativelanguage.googleapis.com/v1beta/models/"
     "gemini-2.0-flash:generateContent", None, "x-goog-api-key"),
    ("OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions",
     "gpt-4o-mini", "Authorization"),
]

SYSTEM_PROMPT = (
    "You are the AI tutor for an AI/ML engineering academy. Explain precisely "
    "with small examples. When the student asks for an exam, generate MCQs as "
    "JSON: [{prompt, options:[4], answer_index, explanation}]. Be encouraging "
    "but technically rigorous. Never invent APIs.")


def detect_courses(text: str) -> list:
    t = text.lower()
    hits = []
    for slug, kws in COURSE_KEYWORDS.items():
        if any(k in t for k in kws):
            hits.append(slug)
    return hits


def local_answer(message: str, courses: list) -> str:
    t = message.lower().strip()
    if t in GREETINGS:
        return ("Hey! I'm your AI tutor. Ask me to explain any concept "
                "(RAG, attention, overfitting...), say 'quiz me on <topic>' "
                "for exam questions, or ask 'which course covers X'.")
    if "which course" in t or ("course" in t and "?" in t):
        if courses:
            return (f"'{courses[0]}' is the best match - open /courses/{courses[0]} "
                    "for lessons, exam, assignments and projects.")
        return ("Tell me the topic (e.g. 'which course covers attention?') and "
                "I'll route you.")
    for key, text_ in KNOWLEDGE.items():
        if key in t or key.replace(" ", "") in t:
            extra = ""
            if courses:
                extra = f"\n\nStudy it in: /courses/{courses[0]}"
            return text_ + extra
    if courses:
        return (f"That topic is covered in the '{courses[0]}' course - "
                f"open /courses/{courses[0]} for lessons, its exam, assignments "
                "and projects. Ask me 'explain <concept>' for a quick answer.")
    return ("I can explain AI/ML concepts (RAG, transformers, gradient descent, "
            "BM25, agents...), route you to the right course, or quiz you: "
            "say 'quiz me on transformers'.")


async def llm_answer(messages: list) -> str | None:
    for env_key, url, model, header in PROVIDERS:
        key = os.environ.get(env_key)
        if not key:
            continue
        try:
            if "generativelanguage" in url:  # Gemini shape
                contents = [{"role": ("user" if m["role"] != "assistant"
                                      else "model"), "parts": [{"text": m["content"]}]}
                            for m in messages]
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(url, json={"contents": contents,
                                                     "systemInstruction": {
                                                         "parts": [{"text": SYSTEM_PROMPT}]}},
                                          headers={header: key})
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                headers = {header: f"Bearer {key}"}
                if env_key == "OPENROUTER_API_KEY":
                    headers["HTTP-Referer"] = "https://aiml-academy.local"
                body = {"model": model,
                        "messages": [{"role": "system",
                                      "content": SYSTEM_PROMPT}] + messages}
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(url, json=body, headers=headers)
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
    return None


class ChatIn(BaseModel):
    message: str
    history: list = []


@router.post("")
async def chat(body: ChatIn, user=Depends(current_user),
               db: Session = Depends(get_db)):
    if not body.message.strip():
        raise HTTPException(400, "empty message")
    courses = detect_courses(body.message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in body.history[-8:]:
        messages.append({"role": h.get("role", "user"),
                         "content": str(h.get("content", ""))[:2000]})
    messages.append({"role": "user", "content": body.message[:2000]})
    reply = await llm_answer(messages)
    source = "llm"
    if not reply:
        reply = local_answer(body.message, courses)
        source = "builtin"
    return {"reply": reply, "source": source,
            "suggested_courses": courses[:3]}


@router.get("/quiz/{topic}")
def quiz(topic: str, n: int = 5, user=Depends(current_user),
         db: Session = Depends(get_db)):
    """Pull exam questions matching a topic -> instant practice set."""
    t = topic.lower()[:40]
    qset = db.query(Question).all()
    scored = []
    for q in qset:
        blob = (q.prompt + " " + " ".join(q.options or [])).lower()
        overlap = sum(1 for w in t.split() if w in blob) + (2 if t in blob else 0)
        if overlap:
            scored.append((overlap, q))
    random.shuffle(scored)
    scored.sort(key=lambda p: -p[0])
    picked = [q for _, q in scored[:max(1, min(n, 20))]]
    if not picked:
        allq = db.query(Question).all()
        random.shuffle(allq)
        picked = allq[:max(1, min(n, 20))]
    return {"topic": topic, "questions": [
        {"id": q.id, "prompt": q.prompt, "options": q.options,
         "kind": q.kind, "diagram": q.diagram} for q in picked]}
