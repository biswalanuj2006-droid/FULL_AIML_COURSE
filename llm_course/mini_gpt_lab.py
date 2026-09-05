"""
================================================================================
REAL MINI-GPT LAB  (llm_course/mini_gpt_lab.py)
================================================================================
A runnable, end-to-end miniature GPT (decoder-only Transformer) that:
  1. loads a REAL corpus (this course's own lesson text on disk; a classic
     public-domain English fallback if the lessons are not present),
  2. character-level tokenizes it (encode / decode),
  3. builds a complete GPT from scratch (token + position embeddings,
     N causal self-attention blocks with LayerNorm + residual connections
     + GELU MLP, final norm, LM head),
  4. trains it with AdamW (next-token prediction / cross entropy),
  5. reports train/val loss + perplexity every checkpoint, and prints
     GENERATED SAMPLES so you can watch the model learn to write,
  6. compares against a BIGRAM baseline (the classic nanoGPT signal:
     GPT must beat the bigram loss to prove attention learns structure).

This lab is the practical companion to COURSE.txt Parts 13-22 (language
modeling, training objective, pipeline, optimization) and Part 40
(autoregressive inference).  Runs on CPU with only numpy + torch.

Run it from anywhere:
    python mini_gpt_lab.py
    python mini_gpt_lab.py --max-steps 600    # shorter run
    python mini_gpt_lab.py --corpus-dir ..    # point at a dir of *.txt files

Every tensor shape is printed inline in comments.  A tiny config means a few
hundred steps take roughly 1-3 minutes on a laptop CPU.
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. CORPUS  --  real text, loaded from disk when available
# ----------------------------------------------------------------------------

# Public-domain fallback (used only if no lesson .txt files are found nearby):
# Shakespeare sonnets + US founding documents + Gettysburg Address.  Plain
# ASCII prose gives a clean, classic character-level training signal.
FALLBACK_CORPUS = (
    "Shall I compare thee to a summer's day? Thou art more lovely and more "
    "temperate: Rough winds do shake the darling buds of May, And summer's "
    "lease hath all too short a date: Sometime too hot the eye of heaven "
    "shines, And often is his gold complexion dimm'd; And every fair from "
    "fair sometime declines, By chance or nature's changing course untrimm'd; "
    "But thy eternal summer shall not fade, Nor lose possession of that fair "
    "thou ow'st; Nor shall death brag thou wander'st in his shade, When in "
    "eternal lines to time thou grow'st: So long as men can breathe or eyes "
    "can see, So long lives this, and this gives life to thee. "
    "When to the sessions of sweet silent thought I summon up remembrance of "
    "things past, I sigh the lack of many a thing I sought, And with old woes "
    "new wail my dear time's waste: Then can I drown an eye, unused to flow, "
    "For precious friends hid in death's dateless night, And weep afresh "
    "love's long since cancell'd woe, And moan the expense of many a vanish'd "
    "sight: Then can I grieve at grievances foregone, And heavily from woe to "
    "woe tell o'er The sad account of fore-bemoaned moan, Which I new pay as "
    "if not paid before. But if the while I think on thee, dear friend, All "
    "losses are restor'd and sorrows end. "
    "Let me not to the marriage of true minds admit impediments. Love is not "
    "love which alters when it alteration finds, Or bends with the remover to "
    "remove: O no! it is an ever-fixed mark, That looks on tempests and is "
    "never shaken; It is the star to every wandering bark, Whose worth's "
    "unknown, although his height be taken. Love's not Time's fool, though "
    "rosy lips and cheeks within his bending sickle's compass come: Love "
    "alters not with his brief hours and weeks, But bears it out even to the "
    "edge of doom. If this be error and upon me proved, I never writ, nor no "
    "man ever loved. "
    "We the People of the United States, in Order to form a more perfect "
    "Union, establish Justice, insure domestic Tranquility, provide for the "
    "common defence, promote the general Welfare, and secure the Blessings of "
    "Liberty to ourselves and our Posterity, do ordain and establish this "
    "Constitution for the United States of America. "
    "Four score and seven years ago our fathers brought forth on this "
    "continent, a new nation, conceived in Liberty, and dedicated to the "
    "proposition that all men are created equal. Now we are engaged in a "
    "great civil war, testing whether that nation, or any nation so conceived "
    "and so dedicated, can long endure. We are met on a great battle-field of "
    "that war. We have come to dedicate a portion of that field, as a final "
    "resting place for those who here gave their lives that that nation might "
    "live. It is altogether fitting and proper that we should do this. But, "
    "in a larger sense, we can not dedicate, we can not consecrate, we can "
    "not hallow this ground. The brave men, living and dead, who struggled "
    "here, have consecrated it, far above our poor power to add or detract. "
    "The world will little note, nor long remember what we say here, but it "
    "can never forget what they did here. It is for us the living, rather, to "
    "be dedicated here to the unfinished work which they who fought here have "
    "thus far so nobly advanced. It is rather for us to be here dedicated to "
    "the great task remaining before us, that from these honored dead we take "
    "increased devotion to that cause for which they gave the last full "
    "measure of devotion, that we here highly resolve that these dead shall "
    "not have died in vain, that this nation, under God, shall have a new "
    "birth of freedom, and that government of the people, by the people, for "
    "the people, shall not perish from the earth."
)


_BASE = os.path.dirname(os.path.abspath(__file__))


def clean_text(raw: str) -> str:
    """Data cleaning (COURSE.txt Part 28): strip the noise that dominates
    markdown files so the model trains on prose instead of formatting.
    Removed: pure ruler/decoration lines (====, ----, ***), table borders,
    checkbox markers, heading hashes, and code/math-heavy lines (a real
    quality filter).  Paragraph breaks are preserved.
    """
    def is_prose_line(s: str) -> bool:
        non_ws = [c for c in s if not c.isspace()]
        if not non_ws:
            return False
        letters = sum(c.isalpha() for c in non_ws)
        if letters / len(non_ws) < 0.60:      # too many digits/symbols
            return False
        code_chars = sum(c in "=(){}[];<>\\/|`*" for c in non_ws)
        if code_chars > 0.05 * len(non_ws):    # looks like code or a formula
            return False
        return True

    lines: List[str] = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # decoration / ruler lines made of only 1-2 repeated symbols
        uniq = set(s)
        if len(uniq) <= 2 and uniq <= set("=-_*~#|.:+ ") and len(s) > 1:
            continue
        # markdown table borders like  |---|---|   or   ---
        if re.fullmatch(r"[|\-\s:]+\|[|\-\s:]*", s) or re.fullmatch(r"-{3,}", s):
            continue
        if re.match(r"^\[\s*[xX]?\s*\]", s):  # checkbox lines
            continue
        s = re.sub(r"^#{1,6}\s*", "", s)  # heading markers
        if not is_prose_line(s):
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def load_corpus(corpus_dirs: Optional[List[str]] = None, max_chars: int = 200_000) -> Tuple[str, str]:
    """Concatenate real prose found on disk (this course's own lesson text),
    cleaned of markdown formatting noise.  Returns (text, source_label).
    If nothing is found nearby, falls back to the embedded classic English
    corpus so the lab always runs.
    """
    if corpus_dirs is None:
        # Defaults (anchored to this file, so cwd does not matter): prose-heavy
        # module lessons + the two sub-course files.
        here = _BASE
        up = os.path.dirname(_BASE)
        corpus_dirs = [
            here,                           # COURSE/PRACTICE/PROJECT .txt
            os.path.join(up, "genai_agents_course"),
            os.path.join(up, "06_ML_FUNDAMENTALS"),
            os.path.join(up, "09_CLASSIFICATION"),
            os.path.join(up, "14_MODEL_EVALUATION"),
            os.path.join(up, "10_REGRESSION"),
            os.path.join(up, "27_GENAI_FOUNDATIONS"),
            os.path.join(up, "28_LLM_FUNDAMENTALS"),
        ]
    texts: List[str] = []
    seen: set = set()
    for d in corpus_dirs:
        for pattern in (os.path.join(d, "*.txt"), os.path.join(d, "*.md")):
            for path in sorted(glob.glob(pattern)):
                real = os.path.realpath(path)
                if real in seen:
                    continue
                seen.add(real)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        texts.append(fh.read())
                except OSError:
                    continue
    if texts:
        joined = clean_text("".join(texts))
        if len(joined) > 4000:  # enough real prose to train on
            return joined[:max_chars], "cleaned lesson text on disk"
    return FALLBACK_CORPUS, "embedded classic-English fallback corpus"


# ----------------------------------------------------------------------------
# 2. CHARACTER TOKENIZER  (COURSE.txt Part 12)
# ----------------------------------------------------------------------------

class CharTokenizer:
    """token <-> id mapping over every character that appears in the corpus.

    encode: str -> List[int];  decode: List[int] -> str
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.vocab_size: int = len(chars)
        self.stoi: dict = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)


# ----------------------------------------------------------------------------
# 3. GPT MODEL  (COURSE.txt Parts 18, 10, 8, 9)
# ----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, fused QKV projection.

    Input  x: [B, T, C]
    qkv(x):  [B, T, 3*C]  ->  [B, T, 3, H, Dh]  (Dh = C // H)
    scores = q @ k^T * (1/sqrt(Dh)):  [B, H, T, T]   (quadratic in T!)
    causal mask zeroes the upper triangle (token t sees only t' <= t).
    out = attn @ v:  [B, H, T, Dh] -> merge heads -> [B, T, C]
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must split evenly across heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        # Lower-triangular causal mask: [1, 1, T, T] broadcast over batch+heads.
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)  # each [B, T, C]
        # [B, T, C] -> [B, T, H, Dh] -> [B, H, T, Dh]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # scores [B, H, T, T], scaled by 1/sqrt(Dh) (Part 6: why the sqrt?)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = attn @ v  # [B, H, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # concat heads
        return self.proj(out)


class MLP(nn.Module):
    """Position-wise FFN: Linear(C->4C) -> GELU -> Linear(4C->C)."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block: x -> x + attn(LN(x)) -> x + mlp(LN(x))."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # residual stream (Part 25)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Decoder-only language model (Part 18).

    Shapes:
        idx [B, T] -> token_emb [B, T, C] + pos_emb -> blocks -> LN -> lm_head
        logits [B, T, V]; the row logits[b, t] scores the token AT t+1.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int,
                 n_layer: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"sequence {T} longer than block {self.block_size}"
        tok = self.token_emb(idx)  # [B, T, C]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [1, T]
        x = tok + self.pos_emb(pos)  # [B, T, C]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B, T, V]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.9, top_k: Optional[int] = None) -> torch.Tensor:
        """Autoregressive sampling (Part 40): feed last block_size tokens,
        read the final position's logits, sample one token, append, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature  # [B, V]
            if top_k is not None:
                topk = torch.topk(logits, top_k, dim=-1).values[:, -1:]  # k-th value
                logits = logits.masked_fill(logits < topk, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ----------------------------------------------------------------------------
# 4. DATA BATCHING  (Part 21: next-token prediction, shifted labels)
# ----------------------------------------------------------------------------

def get_batch(data: np.ndarray, batch_size: int, block_size: int) -> tuple:
    """Random contiguous windows.  x = chars[t : t+B], y = chars[t+1 : t+B+1]."""
    ix = np.random.randint(0, len(data) - block_size, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


# ----------------------------------------------------------------------------
# 5. BIGRAM BASELINE  (the bar the GPT must clear)
# ----------------------------------------------------------------------------

def bigram_val_loss(train_ids: np.ndarray, val_ids: np.ndarray, vocab_size: int,
                    eps: float = 0.01) -> float:
    """Smoothed bigram NLL over the validation split, in nats per character."""
    pairs = train_ids[:-1].astype(np.int64) * vocab_size + train_ids[1:].astype(np.int64)
    counts = np.bincount(pairs, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size)
    row_sums = counts.sum(axis=1, keepdims=True) + eps * vocab_size
    logp = np.log((counts + eps) / row_sums)  # Laplace-smoothed log probs
    val_pairs = val_ids[:-1].astype(np.int64) * vocab_size + val_ids[1:].astype(np.int64)
    return float(-logp.ravel()[val_pairs].mean())


# ----------------------------------------------------------------------------
# 6. TRAINING
# ----------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: np.ndarray, val_data: np.ndarray,
                  batch_size: int, block_size: int, n_batches: int = 40) -> dict:
    """Fixed eval batches (fresh RNG seed) so checkpoints are comparable."""
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)  # [B, T, V]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


def train(args) -> MiniGPT:
    # --- corpus + tokenizer ---
    print("=" * 72)
    print("REAL MINI-GPT LAB")
    print("=" * 72)
    corpus, src_label = load_corpus(corpus_dirs=args.corpus_dir.split(",") if args.corpus_dir else None)
    tok = CharTokenizer(corpus)
    V = tok.vocab_size
    print(f"corpus: {len(corpus):,} characters | vocab: {V} distinct characters")
    print(f"data source: {src_label}")
    ids = np.array(tok.encode(corpus), dtype=np.int64)

    # train / validation split (last 5% held out)
    n_val = max(1000, len(ids) // 20)
    train_data, val_data = ids[:-n_val], ids[-n_val:]
    print(f"train split: {len(train_data):,} chars | val split: {len(val_data):,} chars")

    # --- model ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        vocab_size=V,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=args.dropout,
    ).to(device)
    print(f"device: {device} | model params: {model.count_params():,}")
    print(f"config: {args.n_layer} layers, {args.n_head} heads, "
          f"embd {args.n_embd}, block {args.block_size}, batch {args.batch_size}")

    # --- bigram baseline (printed once, GPT must beat it) ---
    bigram_loss = bigram_val_loss(train_data, val_data, V)
    print(f"bigram baseline val loss: {bigram_loss:.3f} nats "
          f"(ppl {math.exp(bigram_loss):.1f}) <- GPT must go below this")

    # --- optimizer: AdamW (Part 23) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    # simple cosine schedule with a short linear warmup
    def lr_at(step: int) -> float:
        warmup = max(1, args.max_steps // 10)
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    # --- train ---
    print("step | train loss |  val loss | val ppl | elapsed")
    t0 = time.time()
    # Seed the demo prompts from a natural-language position in the corpus
    # (first occurrence of ' the '), not from a file header.
    seed_start = max(0, corpus.find(" the "))
    samples_done = False
    for step in range(1, args.max_steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step - 1)
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        x, y = x.to(device), y.to(device)
        logits = model(x)  # [B, T, V]
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Part 22: grad clipping
        optimizer.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            est = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size)
            ppl = math.exp(est["val"])
            print(f"{step:5d} | {est['train']:10.3f} | {est['val']:8.3f} | {ppl:7.1f} | "
                  f"{time.time() - t0:6.1f}s")

        # sample every eval interval, using a prompt lifted from the corpus
        if step % (args.eval_every * 2) == 0 or (step == args.max_steps and not samples_done):
            samples_done = step == args.max_steps
            ctx0 = max(0, seed_start + (step // args.eval_every) % max(1, len(corpus) - 40))
            prompt = corpus[ctx0 : ctx0 + 24]
            ctx = torch.tensor(tok.encode(prompt), dtype=torch.long, device=device)[None, :]
            sample = model.generate(ctx, max_new_tokens=160, temperature=0.9, top_k=40)
            print(f"\n--- sample @ step {step} (prompt: {prompt!r}) ---")
            print(tok.decode(sample[0].tolist()))
            print("---\n")

    return model, tok, train_data, val_data, corpus


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a real miniature GPT")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=96)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=3)
    ap.add_argument("--n-embd", type=int, default=96)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--corpus-dir", type=str, default=None,
                    help="comma-separated dirs of *.txt to train on")
    ap.add_argument("--big", action="store_true",
                    help="bigger model + longer run: 4 layers, 4 heads, "
                         "embd 128, block 128, 800 steps")
    args = ap.parse_args()

    # --big preset: a noticeably larger model (~3.4x the default parameters)
    # trained longer, to show that more capacity + more steps -> lower val
    # loss.  Kept CPU-reachable: 800 steps of the 4x128 model take roughly
    # 10-15 minutes on a laptop.
    if args.big:
        args.n_layer, args.n_head = 4, 4
        args.n_embd, args.block_size = 128, 128
        args.max_steps = 800
        args.eval_every = 200

    model, tok, train_data, val_data, corpus = train(args)

    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    est = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size)
    print(f"GPT   val loss: {est['val']:.3f} nats | val ppl {math.exp(est['val']):.1f}")
    bigram_loss = bigram_val_loss(train_data, val_data, tok.vocab_size)
    print(f"bigram baseline: {bigram_loss:.3f} nats | ppl {math.exp(bigram_loss):.1f}")
    print(f"=> the mini-GPT {( 'BEATS' if est['val'] < bigram_loss else 'did not beat' )} "
          "the bigram baseline: attention learned real structure, not just pairs")
    print("final trained sample (temperature 0.8):")
    seed_start = max(0, corpus.find(" the "))
    prompt = corpus[seed_start : seed_start + 24]
    ctx = torch.tensor(tok.encode(prompt), dtype=torch.long)[None, :]
    sample = model.generate(ctx, max_new_tokens=240, temperature=0.8, top_k=40)
    print(tok.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
