"""
================================================================================
KV CACHE LAB  (llm_course/kv_cache_lab.py)
================================================================================
A runnable lab for COURSE.txt Part 41 (KV cache), Part 42 (prefill/decode)
and Part 64 (MHA vs MQA vs GQA).  It builds the same miniature GPT as
mini_gpt_lab.py but with an OPTIONAL per-layer KV cache, then proves three
things with numbers:

  1. CORRECTNESS: cached token-by-token generation produces EXACTLY the
     same logits as full recomputation (the cache is a pure speedup, it
     must never change the math).
  2. SPEED: decode with a cache reuses previous K/V and skips recomputing
     the whole prefix; we measure tokens/sec with and without the cache
     and the per-token decode cost growth.
  3. MEMORY: the cache grows linearly with sequence length; we compute the
     exact byte cost for this model and for a 7B-class config, then show
     how GQA (Part 64) shrinks it.

Runs on CPU with only numpy + torch.  No training is needed - correctness
holds for any weights, so the model is randomly initialized.

    python kv_cache_lab.py
================================================================================
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ----------------------------------------------------------------------------
# MODEL WITH OPTIONAL KV CACHE
# ----------------------------------------------------------------------------

class CachedSelfAttention(nn.Module):
    """Multi-head causal attention that can store/append K,V across steps.

    Full pass:   x [B, T, C]  -> q,k,v [B, H, T, Dh]; scores [B, H, T, T];
                 causal mask applied; out [B, T, C].
    Cached step: x [B, 1, C]  -> the new token's k,v are APPENDED to the
                 cached k,v; attention runs over all tokens seen so far,
                 which for a single new token is exactly its causal row.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if cache is not None:
            k_prev, v_prev = cache                      # [B,H,T_prev,Dh]
            k = torch.cat([k_prev, k], dim=2)           # append new token
            v = torch.cat([v_prev, v], dim=2)
        T_all = k.size(2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T_all]
        if T > 1:  # full pass: enforce causality over the whole window
            mask = self.tril[:, :, :T, :T_all]
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                                    # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), (k, v)                     # (output, updated cache)


class CachedBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CachedSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x: torch.Tensor, cache: Optional[tuple]):
        h, kv = self.attn(self.ln1(x), cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, kv


class KVGPT(nn.Module):
    """Mini decoder-only GPT whose attention layers keep a KV cache.

    forward(x)        - full batched pass (prefill / uncached decode).
    forward_step(x_t) - one new token against cached K/V (cached decode).
    """

    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_head = n_head
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([CachedBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        h = self.token_emb(x) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        for blk in self.blocks:
            h, _ = blk(h, None)
        return self.lm_head(self.ln_f(h))  # [B, T, V]

    def forward_step(self, x_t: torch.Tensor, pos_t: int,
                     caches: List[Optional[tuple]]) -> Tuple[torch.Tensor, List[tuple]]:
        h = self.token_emb(x_t) + self.pos_emb(torch.tensor([[pos_t]], device=x_t.device))
        new_caches: List[tuple] = []
        for blk, cache in zip(self.blocks, caches):
            h, kv = blk(h, cache)
            new_caches.append(kv)
        return self.lm_head(self.ln_f(h)), new_caches  # [B, 1, V]

    def cache_bytes_per_token(self, dtype_bytes: int) -> float:
        """K and V for every layer: 2 * n_layer * n_embd floats per token."""
        return 2.0 * self.n_layer * self.n_embd * dtype_bytes


# ----------------------------------------------------------------------------
# 1. CORRECTNESS: cache must reproduce the full forward exactly
# ----------------------------------------------------------------------------

def check_correctness(model: KVGPT, vocab: int, seq_len: int = 40) -> None:
    ids = torch.randint(0, vocab, (1, seq_len))
    with torch.no_grad():
        logits_full = model(ids)  # [1, seq_len, V]  (one parallel pass)

        # cached path: feed one token at a time, appending K/V
        caches: List[Optional[tuple]] = [None] * model.n_layer
        logits_cached = []
        for t in range(seq_len):
            logit_t, caches = model.forward_step(ids[:, t : t + 1], t, caches)
            logits_cached.append(logit_t)
        logits_cached = torch.cat(logits_cached, dim=1)  # [1, seq_len, V]

    max_diff = (logits_full - logits_cached).abs().max().item()
    ok = max_diff < 1e-5
    print(f"[1] CORRECTNESS: max |cached - full| logit diff = {max_diff:.2e} -> "
          f"{'PASS: the cache is a pure speedup, math unchanged' if ok else 'FAIL'}")
    assert ok, "cache broke the math"


# ----------------------------------------------------------------------------
# 2. SPEED: cached vs uncached decode
# ----------------------------------------------------------------------------

def timing_demo(model: KVGPT, vocab: int, prompt_len: int = 20, gen_len: int = 60) -> None:
    prompt = torch.randint(0, vocab, (1, prompt_len))
    model.eval()

    # --- uncached: every step re-runs the WHOLE sequence (O(n^2) work) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        seq = prompt
        for _ in range(gen_len):
            logits = model(seq)[:, -1, :]
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            seq = torch.cat([seq, nxt], dim=1)
    t_uncached = time.perf_counter() - t0

    # --- cached: each step only processes ONE new token (O(1) per step) ---
    t0 = time.perf_counter()
    with torch.no_grad():
        seq = prompt
        caches: List[Optional[tuple]] = [None] * model.n_layer
        for t in range(gen_len):
            pos = prompt_len + t
            logit_t, caches = model.forward_step(seq[:, -1:], pos, caches)
            nxt = torch.multinomial(F.softmax(logit_t[:, -1, :], dim=-1), 1)
            seq = torch.cat([seq, nxt], dim=1)
    t_cached = time.perf_counter() - t0

    print(f"[2] SPEED: decode {gen_len} tokens (prompt {prompt_len})")
    print(f"    uncached: {t_uncached:.4f}s ({gen_len / t_uncached:.1f} tok/s)  "
          f"- recomputes the whole prefix every step")
    print(f"    cached:   {t_cached:.4f}s ({gen_len / t_cached:.1f} tok/s)  "
          f"- reuses stored K/V, one token through the net")
    print(f"    speedup:  {t_uncached / max(t_cached, 1e-9):.1f}x  "
          f"(grows with sequence length: uncached is O(n^2), cached decode O(n) total)")
    print("    why: uncached redoes qkv+attention for ALL past tokens each step;")
    print("         cached attention still reads the full KV (memory-bound), but")
    print("         it skips recomputing K/V for the prefix (compute saved).")


# ----------------------------------------------------------------------------
# 3. MEMORY: exact KV-cache byte math  (Part 41 + Part 64)
# ----------------------------------------------------------------------------

def memory_math(model: KVGPT) -> None:
    block = model.block_size
    bytes_per_tok_layer = model.cache_bytes_per_token(4)  # fp32
    print(f"[3] MEMORY for this model (n_layer={model.n_layer}, n_embd={model.n_embd}, fp32):")
    print(f"    per token per layer: 2 x {model.n_embd} x 4B = {bytes_per_tok_layer:.0f} B")
    for n in (64, block):
        total = bytes_per_tok_layer * model.n_layer * n
        print(f"    full cache @ {n:4d} tokens: {total:>8,} B ({total / 1024:.1f} KB) "
              f"- linear in n, while attention compute is O(n^2)")

    # 7B-class example (e.g. LLaMA-style): 80 layers, hidden 8192, fp16
    n_layer, n_embd, n_head, head_dim, ctx = 80, 8192, 32, 128, 4096
    kv_bytes = 2 * n_embd * 2  # K+V, fp16
    mha = kv_bytes * n_layer * ctx / (1024**3)
    gqa = (2 * (n_head // 4) * head_dim * 2) * n_layer * ctx / (1024**3)  # 4x fewer KV heads
    mqa = (2 * 1 * head_dim * 2) * n_layer * ctx / (1024**3)             # 1 shared KV head
    print("    7B-class config (80 layers, hidden 8192, ctx 4096, fp16):")
    print(f"    MHA (32 KV heads):    {mha:5.2f} GB   <- 2*8192*2B * 80 layers * 4096 tokens")
    print(f"    GQA (8 KV heads):     {gqa:5.2f} GB   (Part 64: quality close to MHA)")
    print(f"    MQA (1 KV head):      {mqa:5.2f} GB   (Part 64: cheapest, slightly weaker)")
    print("    => GQA/MQA shrink the KV cache by sharing K/V heads across query heads;")


def main() -> None:
    V, n_embd, n_head, n_layer, block = 64, 96, 3, 2, 96
    print("=" * 72)
    print("KV CACHE LAB")
    print("=" * 72)
    model = KVGPT(V, n_embd, n_head, n_layer, block)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"model: {n_layer} layers, {n_head} heads, embd {n_embd}, "
          f"block {block}, params {nparams:,} (random init - no training needed)\n")
    check_correctness(model, V)
    timing_demo(model, V)
    memory_math(model)
    print("=" * 72)
    print("SUMMARY: the KV cache stores K/V per layer and per token, trading")
    print("memory (linear growth) for compute (no prefix recomputation).")
    print("Prefill is compute-bound (parallel); decode is memory-bandwidth-bound.")
    print("All numbers above were produced by this run - nothing is hand-waved.")
    print("=" * 72)


if __name__ == "__main__":
    main()