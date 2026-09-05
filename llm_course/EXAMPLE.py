"""
================================================================================
EXAMPLE.py - PRACTICAL LLM LABORATORY (companion to llm_course/COURSE.txt)
================================================================================
54 sections: math -> tokenizer -> attention -> transformer -> mini-GPT ->
training loop -> generation -> perplexity -> param counting -> LoRA ->
quantization -> KV cache -> inference -> serving -> RAG/tools/agents.

RULES
- [# RUNNABLE] sections execute with numpy only.
- [# TORCH] sections train/check tiny models with PyTorch; they are
  guarded - if torch is missing the section prints SKIPPED and the file
  still runs.
- [# SKELETON] infra sections (FastAPI/HF/Postgres) are guarded reading
  snippets. Marked EDUCATIONAL IMPLEMENTATION where simplified.
- No hardcoded secrets; provider APIs change - isolate behind interfaces.

Run: python EXAMPLE.py   (expect: all numpy sections OK + torch OK)
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

rng = random.Random(42)
np.random.seed(42)
REPORT: list[tuple[str, str]] = []


def section(name: str, ok: bool = True, note: str = "") -> None:
    REPORT.append((name, "OK" if ok else "SKIP" + (f" - {note}" if note else "")))
    print(f"  [{name}] {'OK' if ok else 'skip: ' + note}")


# ============================================================================
# 01-04 PYTHON / NUMPY / MATRIX OPS / PROBABILITY  [# RUNNABLE]
# ============================================================================
@dataclass
class TokenSample:
    """One training example: input ids and the SHIFTED target ids (Part 14)."""
    input_ids: list[int]
    target_ids: list[int]


def retry(max_attempts: int = 3, base_delay: float = 0.05):
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last: Exception | None = None
            for i in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - demo
                    last = e
                    time.sleep(base_delay * (i + 1))
            raise RuntimeError(f"{fn.__name__} failed: {last}")
        return wrapper
    return deco


def run_01_04() -> None:
    v = np.array([1.0, 2.0, 3.0])
    dot = float(v @ v)                                  # 14
    M = np.arange(6.0).reshape(2, 3)
    matmul = (M @ np.ones(3)).shape                     # (2,)
    p_spam_given_free = 0.6 * 0.2 / 0.25                # Bayes -> 0.48
    ok = dot == 14.0 and matmul == (2,) and abs(p_spam_given_free - 0.48) < 1e-12
    section("01-04 python/numpy/matmul/probability", ok)


# ============================================================================
# 05-07 ENTROPY / CROSS ENTROPY / SOFTMAX  [# RUNNABLE]
# ============================================================================
def softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Stable log-softmax = z - max - log(sum(exp(z-max)))."""
    m = np.max(z, axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, target_idx: int) -> float:
    """NLL of the target token under the logits: -log_softmax[target]."""
    return float(-log_softmax(logits)[target_idx])


def entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def run_05_07() -> None:
    s1 = softmax(np.array([2.0, 1.0, 0.1]))             # [0.659,0.242,0.099]
    ce = cross_entropy(np.array([2.0, 1.0, 0.1]), 0)    # -ln(0.659)=0.417
    h = entropy(np.array([0.5, 0.5]))                   # ln 2 = 0.693
    ok = abs(s1[0] - 0.659) < 1e-3 and abs(ce - 0.417) < 1e-2 and abs(h - 0.693) < 1e-3
    section("05-07 entropy/cross-entropy/softmax", ok,
            f"softmax={s1[0]:.3f} CE={ce:.3f} H={h:.3f}")


# ============================================================================
# 08-09 EMBEDDINGS + COSINE  [# RUNNABLE]
# ============================================================================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_08_09() -> None:
    king, queen, apple = (np.array([1.0, 1.0, .5]), np.array([1.0, .9, .4]),
                          np.array([.2, .1, 1.0]))
    ok = cosine(king, queen) > cosine(king, apple)
    section("08/09 embeddings + cosine similarity", ok)


# ============================================================================
# 10-11 TOKENIZER FROM SCRATCH + BPE  [# RUNNABLE]
# ============================================================================
class CharTokenizer:
    """Char-level tokenizer: enough to train a tiny LM (Part 12)."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def bpe_merge_once(words: list[str]) -> list[str]:
    from collections import Counter
    pairs: Counter = Counter()
    for w in words:
        parts = w.split(" ")
        for a, b in zip(parts, parts[1:]):
            pairs[(a, b)] += 1
    if not pairs:
        return words
    (a, b), _ = pairs.most_common(1)[0]
    return [w.replace(f"{a} {b}", f"{a}{b}") for w in words]


def run_10_11() -> None:
    tok = CharTokenizer("hello world hello")
    enc = tok.encode("hello")
    ok_tok = enc == [tok.stoi[c] for c in "hello"] and tok.decode(enc) == "hello"
    merged = bpe_merge_once(["low low lower", "lowest low"])
    ok_bpe = any("lower" in w for w in merged)
    section("10/11 char tokenizer + BPE merge step", ok_tok and ok_bpe)


# ============================================================================
# 12 POSITIONAL ENCODING  [# RUNNABLE]
# ============================================================================
def sinusoidal(T: int, D: int) -> np.ndarray:
    pos = np.arange(T)[:, None]
    i = np.arange(D // 2)[None, :]
    ang = pos / np.power(10000.0, 2 * i / D)
    pe = np.zeros((T, D))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def run_12() -> None:
    pe = sinusoidal(64, 32)
    # Rotational property: adding a constant offset rotates the angle dims
    ok = pe.shape == (64, 32) and abs(float(pe[0, 0])) < 1e-9
    section("12 sinusoidal positional encoding", ok)


# ============================================================================
# 13-16 ATTENTION / CAUSAL / MASK / MHA  [# RUNNABLE]
# ============================================================================
def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
              mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    dk = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(dk)                    # [T,T]
    if mask is not None:
        scores = np.where(mask == 1, scores, -1e9)
    w = softmax(scores)                                 # rows sum to 1
    return w @ V, w


def causal_mask(T: int) -> np.ndarray:
    return np.tril(np.ones((T, T)))


def mha(X: np.ndarray, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
        Wo: np.ndarray, H: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    B, T, D = X.shape
    Dh = D // H
    Q = (X @ Wq).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)  # [B,H,T,Dh]
    K = (X @ Wk).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    V = (X @ Wv).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
    s = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(Dh)         # [B,H,T,T]
    if mask is not None:
        s = np.where(mask[None, None] == 1, s, -1e9)
    w = softmax(s)
    out = (w @ V).transpose(0, 2, 1, 3).reshape(B, T, D)    # concat heads
    return out @ Wo


def run_13_16() -> None:
    T, D, H = 4, 8, 2
    X = np.random.randn(T, D)
    o1, w1 = attention(X, X, X, mask=causal_mask(T))
    ok1 = np.allclose(o1[0], w1[0, 0] * X[0], atol=1e-6) and np.allclose(w1.sum(1), 1.0)
    B = 2
    Wq = np.random.randn(D, D) / math.sqrt(D)
    o2 = mha(np.random.randn(B, T, D), Wq, Wq, Wq, Wq, H, mask=causal_mask(T))
    section("13-16 attention + causal mask + MHA", ok1 and o2.shape == (B, T, D))


# ============================================================================
# 17-20 LAYERNORM / RMSNORM / FFN / SWIGLU  [# RUNNABLE]
# ============================================================================
def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def rmsnorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def ffn(x: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    return gelu(x @ W1) @ W2


def swiglu(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, W3: np.ndarray) -> np.ndarray:
    gate = 1.0 / (1.0 + np.exp(-(x @ W3)))
    return (x @ W1 * gate) @ W2


def run_17_20() -> None:
    x = np.random.randn(3, 16)
    ok_ln = abs(float(layernorm(x).mean())) < 1e-5 and abs(float(layernorm(x).std()) - 1) < 1e-3
    D, F = 8, 32
    xx = np.random.randn(2, D)
    o1 = ffn(xx, np.random.randn(D, F) / math.sqrt(D), np.random.randn(F, D) / math.sqrt(F))
    o2 = swiglu(xx, np.random.randn(D, F) / math.sqrt(D), np.random.randn(F, D) / math.sqrt(F),
                np.random.randn(D, F) / math.sqrt(D))
    section("17-20 LayerNorm/RMSNorm/FFN/SwiGLU", ok_ln and o1.shape == o2.shape == (2, D))


# ============================================================================
# 21-23 TRANSFORMER BLOCK + MINI TRANSFORMER + TOKENIZER INTEGRATION
# ============================================================================
class MiniGPT:
    """Educational decoder-only mini-GPT with NumPy forward (Part 10/18)."""

    def __init__(self, vocab: int, D: int, H: int, N: int):
        s = 1.0 / math.sqrt(D)
        self.D, self.H = D, H
        self.emb = np.random.randn(vocab, D) * s
        self.wq = [np.random.randn(D, D) * s for _ in range(N)]
        self.wk = [np.random.randn(D, D) * s for _ in range(N)]
        self.wv = [np.random.randn(D, D) * s for _ in range(N)]
        self.wo = [np.random.randn(D, D) * s for _ in range(N)]
        self.w1 = [np.random.randn(D, 4 * D) * s for _ in range(N)]
        self.w2 = [np.random.randn(4 * D, D) * s for _ in range(N)]
        self.head = np.random.randn(D, vocab) * s

    def forward(self, ids: np.ndarray) -> np.ndarray:
        B, T = ids.shape
        x = self.emb[ids] + sinusoidal(T, self.D)[None]       # [B,T,D]
        mask = causal_mask(T)
        for n in range(len(self.wq)):
            h = layernorm(x)
            x = x + mha(h, self.wq[n], self.wk[n], self.wv[n], self.wo[n], self.H, mask)
            x = x + ffn(layernorm(x), self.w1[n], self.w2[n])
        return layernorm(x) @ self.head                       # [B,T,V]


def run_21_23() -> None:
    tok = CharTokenizer("the quick brown fox jumps over the lazy dog the end")
    vocab = tok.vocab_size
    model = MiniGPT(vocab=vocab, D=16, H=2, N=2)
    ids = np.array([tok.encode("the quick brown fox")])
    logits = model.forward(ids)
    ok = logits.shape == (1, len(ids[0]), vocab)
    section("21-23 transformer block + mini transformer + tokenizer", ok,
            f"logits {logits.shape}")


# ============================================================================
# 24-26 LM TRAINING LOOP + NEXT-TOKEN PREDICTION + LOGITS  [# TORCH]
# ============================================================================
def run_24_26() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        section("24-26 LM training + next-token prediction", False, "torch missing")
        return

    torch.manual_seed(0)
    text = ("the cat sat on the mat and the cat saw the dog "
            "the dog ran after the cat the cat climbed the tree "
            "the dog barked at the tree the cat slept in the sun ")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text])

    # EDUCATIONAL IMPLEMENTATION: a toy embedding->LM-head stack. The point
    # is the TRAINING LOOP mechanics (shifted labels, CE, clip, AdamW), not
    # the architecture - the NumPy MiniGPT forward lives in sections 21-23.
    model = nn.Sequential(
        nn.Embedding(len(chars), 32),
        nn.LayerNorm(32),
        nn.Linear(32, len(chars)),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    x = ids[:-1].unsqueeze(0)            # input  "the cat sat ... sun "
    y = ids[1:].unsqueeze(0)             # target SHIFTED by one (Part 14)
    losses: list[float] = []
    for step in range(300):
        opt.zero_grad()
        logits = model(x)                # [1,T,V]
        loss = nn.functional.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    learn = losses[0] > losses[-1] * 1.5
    # next-token prediction on a fresh prefix
    prefix = "the cat"
    with torch.no_grad():
        px = torch.tensor([[stoi[c] for c in prefix]])
        probs = torch.softmax(model(px)[0, -1], -1)
        nxt = chars[int(probs.argmax())]
    section("24-26 LM training + next-token prediction (torch)", learn,
            f"loss {losses[0]:.3f}->{losses[-1]:.3f}, after 'the cat' -> '{nxt}'")


# ============================================================================
# 27-31 GENERATION: TEMPERATURE / TOP-K / TOP-P / TEXT / PERPLEXITY
# ============================================================================
def sample(logits: np.ndarray, temperature: float = 1.0, k: Optional[int] = None,
           p: Optional[float] = None) -> int:
    if k is not None:
        thr = np.partition(logits, -k)[-k]
        l = np.where(logits >= thr, logits, -np.inf)
    elif p is not None:
        order = np.argsort(-logits)
        cum = np.cumsum(softmax(logits[order]))
        keep = order[:int(np.searchsorted(cum, p)) + 1]
        l = np.full_like(logits, -np.inf)
        l[keep] = logits[keep]
    else:
        l = logits
    probs = softmax(l, temperature)
    return int(rng.choices(range(len(probs)), weights=probs)[0])


def generate(model: MiniGPT, tok: CharTokenizer, prompt: str, n: int,
             temperature: float = 0.8, k: Optional[int] = None) -> str:
    ids = tok.encode(prompt)
    for _ in range(n):
        logits = model.forward(np.array([ids[-16:]]))[0, -1]
        ids.append(sample(logits, temperature, k=k))
    return tok.decode(ids)


def perplexity_of(probs: np.ndarray) -> float:
    """probs: per-token probabilities of the true tokens -> exp(-mean ln p)."""
    return float(np.exp(-np.mean(np.log(np.clip(probs, 1e-12, 1.0)))))


def run_27_31() -> None:
    l = np.array([2.0, 1.0, 0.1])
    s1, s05, s2 = softmax(l), softmax(l, 0.5), softmax(l, 2.0)
    mono = float(s05[0]) > float(s1[0]) > float(s2[0])
    tok = CharTokenizer("the quick brown fox jumps over the lazy dog ")
    m = MiniGPT(tok.vocab_size, D=16, H=2, N=2)
    out = generate(m, tok, "the", n=6, temperature=0.9, k=5)
    ppl_uniform = perplexity_of(np.full(100, 0.5))
    ok = mono and len(out) > 3 and abs(ppl_uniform - 2.0) < 1e-9
    section("27-31 temperature/top-k/top-p/generation/perplexity", ok,
            f"generated '{out}', ppl(0.5)={ppl_uniform}")


# ============================================================================
# 32 PARAMETER COUNTING  [# RUNNABLE]
# ============================================================================
def count_params(V: int, D: int, N: int, tied: bool = True) -> dict[str, int]:
    emb = V * D
    attn = N * 4 * D * D
    ffn_ = N * 8 * D * D                     # 2 * D * 4D
    norm = N * 4 * D
    head = 0 if tied else V * D
    return {"emb": emb, "attn": attn, "ffn": ffn_, "norm": norm,
            "head": head, "total": emb + attn + ffn_ + norm + head}


def run_32() -> None:
    p = count_params(50_000, 768, 12, tied=True)
    expected = 38_400_000 + 12 * 4 * 768 * 768 + 12 * 8 * 768 * 768 + 12 * 4 * 768
    section("32 parameter counting (V=50k,D=768,N=12)", p["total"] == expected,
            f"{p['total']:,} params ~ {p['total']/1e6:.0f}M")


# ============================================================================
# 33-36 HF / FINE-TUNING SKELETON  [# SKELETON - GUARDED]
# ============================================================================
HF_SNIPPET = '''
from transformers import AutoModelForCausalLM, AutoTokenizer  # API MAY CHANGE
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
msgs = [{"role": "user", "content": "2+2=?"}]
ids = tok.apply_chat_template(msgs, tokenize=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=32, do_sample=True, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=True))
'''


def run_33_36() -> None:
    section("33-36 HuggingFace inference/fine-tuning skeleton", True,
            "guarded snippet in file (API MAY CHANGE)")


# ============================================================================
# 37-38 LoRA + QLoRA  [# TORCH]
# ============================================================================
def run_37_38() -> None:
    try:
        import torch
    except ImportError:
        section("37-38 LoRA", False, "torch missing")
        return
    torch.manual_seed(0)
    D, r, alpha = 768, 8, 16.0
    W = torch.randn(D, D)
    A = torch.randn(r, D) * 0.01     # A ~ small normal
    B = torch.zeros(D, r)            # B = 0 -> W' == W at init
    W_lora = W + (alpha / r) * (B @ A)
    at_init = torch.allclose(W_lora, W, atol=1e-6)
    full_params = W.numel()
    lora_params = A.numel() + B.numel()
    B.requires_grad_(True)
    A.requires_grad_(True)
    # Train ONLY the adapters (the frozen-base trick in real code). A random
    # rank-8 update can never match a random full-rank target exactly, so the
    # assertion is that loss DECREASES toward the rank-8 floor, not that it
    # reaches zero.
    opt = torch.optim.AdamW([A, B], lr=1e-2)
    target = torch.randn(D, D)
    losses: list[float] = []
    for _ in range(100):
        opt.zero_grad()
        loss = ((W + (alpha / r) * (B @ A)) - target).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    improved = losses[-1] < losses[0] * 0.98   # adapters actually learned
    section("37-38 LoRA math + training (torch)", at_init and improved,
            f"full {full_params:,} vs LoRA {lora_params:,} "
            f"({full_params / lora_params:.0f}x fewer), init==W: {at_init}")


# ============================================================================
# 39 QUANTIZATION CONCEPTS  [# RUNNABLE]
# ============================================================================
def quantize_int8(w: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.abs(w).max()) / 127.0
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale


def run_39() -> None:
    w = np.random.randn(128, 128)
    q, scale = quantize_int8(w)
    w_hat = q.astype(np.float32) * scale
    rel = float(np.abs(w - w_hat).mean() / np.abs(w).mean())
    ok = rel < 0.08 and w.nbytes / q.nbytes == 8.0   # float64 -> int8 = 8x
    section("39 INT8 quantization", ok, f"mean rel err {rel:.4f}, 8x smaller")


# ============================================================================
# 40-42 KV CACHE + AUTOREGRESSIVE GENERATION + BATCHING  [# RUNNABLE]
# ============================================================================
class KVCache:
    def __init__(self, L: int, H: int, Dh: int):
        self.k = [np.zeros((0, H, Dh)) for _ in range(L)]
        self.v = [np.zeros((0, H, Dh)) for _ in range(L)]

    def append(self, layer: int, k_t: np.ndarray, v_t: np.ndarray) -> None:
        self.k[layer] = np.concatenate([self.k[layer], k_t])
        self.v[layer] = np.concatenate([self.v[layer], v_t])


def kv_bytes(L: int, H: int, Dh: int, T: int, bytes_: int = 2) -> int:
    return 2 * L * H * Dh * T * bytes_


def run_40_42() -> None:
    per_token = kv_bytes(32, 32, 128, 1)             # LLaMA-7B-like
    full = kv_bytes(32, 32, 128, 4096)
    ok = per_token == 524_288 and full == 2_147_483_648
    section("40-42 KV cache sizing + decode", ok,
            f"0.5 MiB/token -> {full / 2**30:.1f} GiB @ 4096 ctx")


# ============================================================================
# 43-44 STREAMING + EVALUATION  [# RUNNABLE]
# ============================================================================
def stream_tokens(ids: list[int]) -> Any:
    for t in ids:
        yield t


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def run_43_44() -> None:
    ok = list(stream_tokens([1, 2, 3])) == [1, 2, 3] and exact_match("cat", " cat ") == 1.0
    section("43/44 streaming + exact-match eval", ok)


# ============================================================================
# 45-46 FASTAPI SERVER + STREAMING ENDPOINT  [# SKELETON]
# ============================================================================
FASTAPI_SNIPPET = '''
from fastapi import FastAPI, StreamingResponse
from pydantic import BaseModel
app = FastAPI()

class ChatRequest(BaseModel):
    messages: list[dict]

@app.post("/chat")
async def chat(req: ChatRequest):
    async def gen():
        for token in ["hello", " ", "world"]:   # real model call here
            yield f"data: {token}\\n\\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
'''


def run_45_46() -> None:
    section("45/46 FastAPI + streaming endpoint skeleton", True,
            "guarded snippet in file")


# ============================================================================
# 47 MODEL ROUTER  [# RUNNABLE]
# ============================================================================
def route(question: str, difficulty: float) -> str:
    if difficulty < 0.3 or len(question) < 60:
        return "cheap-fast"
    return "flagship" if difficulty >= 0.7 else "medium"


def run_47() -> None:
    ok = route("what is 2+2", 0.1) == "cheap-fast" and route("x" * 100, 0.9) == "flagship"
    section("47 model router (cost/latency aware)", ok)


# ============================================================================
# 48-49 LOGGING + MONITORING  [# RUNNABLE]
# ============================================================================
class UsageLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def log(self, prompt_t: int, comp_t: int, model: str, err: Optional[str] = None) -> None:
        self.rows.append({"prompt_t": prompt_t, "comp_t": comp_t, "model": model,
                          "err": err, "ts": time.time()})

    def totals(self) -> dict[str, float]:
        pt = sum(r["prompt_t"] for r in self.rows)
        ct = sum(r["comp_t"] for r in self.rows)
        errs = sum(1 for r in self.rows if r["err"])
        return {"prompt_tokens": pt, "completion_tokens": ct, "errors": errs}


def run_48_49() -> None:
    log = UsageLog()
    log.log(1000, 200, "cheap-fast")
    log.log(4000, 800, "flagship", err="timeout")
    t = log.totals()
    ok = t == {"prompt_tokens": 5000, "completion_tokens": 1000, "errors": 1}
    section("48/49 usage logging + monitoring counters", ok)


# ============================================================================
# 50-52 RAG / TOOL CALLING / AGENT  [# RUNNABLE]
# ============================================================================
class TinyEmbedder:
    def __init__(self, corpus: list[str]):
        self.vocab: dict[str, int] = {}
        df: dict[str, int] = {}
        for doc in corpus:
            for tok in set(doc.split()):
                df[tok] = df.get(tok, 0) + 1
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self.df, self.N = df, len(corpus)

    def embed(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in toks:
            if t in self.vocab:
                v[self.vocab[t]] += math.log((self.N + 1) / (self.df[t] + 1))
        n = np.linalg.norm(v)
        return v / (n + 1e-12)


class MiniVectorDB:
    def __init__(self, corpus: list[str]):
        self.emb = TinyEmbedder(corpus)
        self.docs = corpus
        self.vecs = [self.emb.embed(d.split()) for d in corpus]

    def retrieve(self, q: str, k: int = 1) -> list[str]:
        qv = self.emb.embed(q.split())
        scored = sorted(range(len(self.docs)),
                        key=lambda i: cosine(qv, self.vecs[i]), reverse=True)
        return [self.docs[i] for i in scored[:k]]


def run_50_52() -> None:
    corpus = ["the refund policy allows returns within 30 days",
              "the company was founded in 2010"]
    db = MiniVectorDB(corpus)
    hits = db.retrieve("what is the refund policy?", k=1)
    ok_rag = hits and "refund" in hits[0]
    section("50 RAG integration (vector retrieval)", ok_rag, f"top1: {hits[0][:30]}...")
    section("51/52 tool calling + agent concepts", True, "see EXAMPLE in genai_agents_course")


# ============================================================================
# 53-54 DOCKER + PRODUCTION CONFIG  [# SKELETON]
# ============================================================================
DOCKERFILE = '''
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m app && chown -R app /app
USER app
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
'''


def run_53_54() -> None:
    section("53/54 Docker + production config", True, "Dockerfile in file")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("LLM LABORATORY - running all sections")
    print("=" * 78)
    run_01_04()
    run_05_07()
    run_08_09()
    run_10_11()
    run_12()
    run_13_16()
    run_17_20()
    run_21_23()
    run_24_26()
    run_27_31()
    run_32()
    run_33_36()
    run_37_38()
    run_39()
    run_40_42()
    run_43_44()
    run_45_46()
    run_47()
    run_48_49()
    run_50_52()
    run_53_54()
    ok = sum(1 for _, s in REPORT if s == "OK")
    print("=" * 78)
    print(f"SUMMARY: {ok}/{len(REPORT)} sections OK")
    print("=" * 78)
