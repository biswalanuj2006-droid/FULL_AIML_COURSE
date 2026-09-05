"""
================================================================================
SPECULATIVE DECODING LAB  (llm_course/speculative_decoding_lab.py)
================================================================================
A runnable lab for COURSE.txt Part 43 (inference optimization: speculative
decoding) and Part 63 (research topics).  It proves the two claims that make
speculative decoding the single most impactful free inference speedup:

  1. CORRECTNESS: greedy speculative decoding produces EXACTLY the same
     tokens as plain greedy decoding of the target model.  Speculative
     decoding is an identity transform on the distribution - it only
     changes the NUMBER of target forward passes, never the answers.
  2. EFFICIENCY: a cheap DRAFT model proposes gamma tokens, the expensive
     TARGET model verifies all gamma in ONE forward pass, and rejected
     tokens are resampled from the true target distribution.  We count
     target forwards and show the wall-clock reasoning.

The cast:
  - TARGET  = a real miniature GPT (same architecture as mini_gpt_lab.py),
             trained briefly on the embedded classic-English corpus.
  - DRAFT   = a bigram model (Markov order 1) fit to the same training
             text - hundreds of times cheaper per forward, but weaker.

What actually happens per round (gamma = 4 in the diagram):

  ctx -> draft proposes:        a b c d          (gamma greedy/sampled tokens)
  ctx + [a b c d] -> target:    one forward pass -> logits for a,b,c,d
  verify:                       target_argmax == draft token?  accept / reject
  if a,b accepted, c rejected:  keep a b, resample c from (q - p)+, restart
  if all accepted:              keep all 4, restart - 4 tokens for 1 forward

Memory-bound decode is WHY this works: one target forward over gamma tokens
costs barely more than one forward over 1 token (Part 42: decode is
bandwidth-bound, not compute-bound), so gamma tokens "for the price of one".

    python speculative_decoding_lab.py
================================================================================
"""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Sibling lab reuse: the GPT architecture, tokenizer, batching and the
# embedded classic-English fallback corpus all live in mini_gpt_lab.py
# (running a script puts its own directory on sys.path, so this import works
# from anywhere).  No training state is shared - we train a fresh target here.
from mini_gpt_lab import CharTokenizer, MiniGPT, get_batch, FALLBACK_CORPUS

torch.manual_seed(1337)
np.random.seed(1337)

CORPUS = FALLBACK_CORPUS
TRAIN_STEPS = 200          # brief training -> target beats the bigram draft
TRAIN_BLOCK = 64           # training window size (batch sampling)
MODEL_BLOCK = 160          # model block_size: large enough that ALL decode
                           # steps below fit the FULL sequence, so the target
                           # conditions on the same context in the plain and
                           # the speculative path (exactly what a KV cache
                           # provides in production - no windowing artifacts)
GAMMA = 4                  # draft tokens proposed per verification round


# ----------------------------------------------------------------------------
# 1. SETUP: corpus, tokenizer, TARGET (trained GPT), DRAFT (bigram)
# ----------------------------------------------------------------------------

def build_models() -> Tuple[MiniGPT, CharTokenizer, np.ndarray, np.ndarray, np.ndarray]:
    tok = CharTokenizer(CORPUS)
    V = tok.vocab_size
    ids = np.array(tok.encode(CORPUS), dtype=np.int64)
    n_val = max(500, len(ids) // 10)
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]
    print(f"corpus: {len(CORPUS):,} chars | vocab {V} | "
          f"train {len(train_ids):,} / val {len(val_ids):,}")

    # --- TARGET: the expensive model we want to decode from ---
    model = MiniGPT(vocab_size=V, n_embd=96, n_head=3, n_layer=2,
                    block_size=MODEL_BLOCK, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    t0 = time.time()
    for step in range(1, TRAIN_STEPS + 1):
        x, y = get_batch(train_ids, 32, TRAIN_BLOCK)
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    # quick val loss (fixed 20 batches)
    with torch.no_grad():
        losses = []
        for _ in range(20):
            x, y = get_batch(val_ids, 32, TRAIN_BLOCK)
            losses.append(F.cross_entropy(model(x).view(-1, V), y.view(-1)).item())
    target_val = float(np.mean(losses))

    # --- DRAFT: smoothed bigram log-probability matrix P[tok_t | tok_{t-1}] ---
    eps = 0.01
    pairs = train_ids[:-1].astype(np.int64) * V + train_ids[1:].astype(np.int64)
    counts = np.bincount(pairs, minlength=V * V).reshape(V, V)
    row_sums = counts.sum(axis=1, keepdims=True) + eps * V
    logp = np.log((counts + eps) / row_sums)          # [V, V], Laplace-smoothed
    draft_val = float(-logp.ravel()[val_ids[:-1] * V + val_ids[1:]].mean())

    print(f"target val loss: {target_val:.3f} (ppl {math.exp(target_val):.1f}) "
          f"| bigram draft val loss: {draft_val:.3f} (ppl {math.exp(draft_val):.1f})")
    print(f"target trained {TRAIN_STEPS} steps in {time.time() - t0:.1f}s - "
          "the draft is ~1000x cheaper per forward (no attention)")
    return model, tok, train_ids, val_ids, logp


# ----------------------------------------------------------------------------
# 2. PLAIN GREEDY DECODING  (baseline: one target forward per token)
# ----------------------------------------------------------------------------

@torch.no_grad()
def greedy_target_decode(model: MiniGPT, ctx: torch.Tensor, n_new: int) -> torch.Tensor:
    """Baseline: every token costs one full target forward.

    The FULL sequence is fed each step (no windowing): MODEL_BLOCK is sized
    so the whole generation fits, which is exactly what a KV-cache-backed
    server does - the speculative path must reproduce this conditioning."""
    seq = ctx.clone()
    assert seq.size(1) + n_new <= MODEL_BLOCK
    for _ in range(n_new):
        logits = model(seq)[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, nxt], dim=1)
    return seq


# ----------------------------------------------------------------------------
# 3. GREEDY SPECULATIVE DECODING  (must match 2. token-for-token)
# ----------------------------------------------------------------------------

@torch.no_grad()
def greedy_spec_decode(model: MiniGPT, logp: np.ndarray, ctx: torch.Tensor,
                       n_new: int, gamma: int) -> Tuple[torch.Tensor, int]:
    """Draft gamma greedy tokens, verify with ONE target forward, correct at
    the first disagreement, restart.  Returns (sequence, target forwards)."""
    seq = ctx.clone()
    assert seq.size(1) + n_new + gamma <= MODEL_BLOCK
    forwards = 0
    while seq.size(1) - ctx.size(1) < n_new:
        # --- draft: gamma greedy tokens from the bigram ---
        last = int(seq[0, -1].item())
        drafts: List[int] = []
        for _ in range(gamma):
            nxt = int(logp[last].argmax())
            drafts.append(nxt)
            last = nxt
        # --- verify: ONE target forward over the full context + drafts ---
        forwards += 1
        probe = torch.cat([seq, torch.tensor([drafts], dtype=torch.long)], dim=1)
        logits = model(probe)[0]  # [len, V]; row i predicts token i+1
        start = seq.size(1) - 1   # first draft token predicted by last ctx row
        n_accept = 0
        for i, tok in enumerate(drafts):
            pred = int(logits[start + i].argmax().item())
            if pred != tok:       # first disagreement: correct it and restart
                seq = torch.cat([seq, torch.tensor([drafts[:i] + [pred]])], dim=1)
                break
            n_accept += 1
        else:
            seq = torch.cat([seq, torch.tensor([drafts])], dim=1)
    return seq, forwards


# ----------------------------------------------------------------------------
# 4. STOCHASTIC SPECULATIVE SAMPLING  (rejection sampling, Part 43 exact rule)
# ----------------------------------------------------------------------------

@torch.no_grad()
def stochastic_spec_sample(model: MiniGPT, logp: np.ndarray, ctx: torch.Tensor,
                           n_new: int, gamma: int, seed: int = 7
                           ) -> Tuple[torch.Tensor, int, float]:
    """Sample gamma draft tokens from the bigram p, then apply the exact
    target-q rejection rule per token:
        accept x  if q(x) >= p(x)          (q dominates p -> always)
        accept x  with prob q(x)/p(x)      otherwise
        on reject: resample x from (q - p)+ / Z  (the corrected distribution)
    Returns (sequence, target forwards, acceptance rate)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.RandomState(seed)
    p = np.exp(logp)                          # [V, V] draft probabilities
    seq = ctx.clone()
    assert seq.size(1) + n_new + gamma <= MODEL_BLOCK
    forwards = 0
    accepted = total = 0
    while seq.size(1) - ctx.size(1) < n_new:
        last = int(seq[0, -1].item())
        # draft the gamma tokens SEQUENTIALLY from the bigram (each draft
        # token conditions on the previous one - p must be the actual
        # proposal distribution for the acceptance rule to be exact)
        draft: List[int] = []
        d = last
        for _ in range(gamma):
            d = int(torch.multinomial(torch.from_numpy(p[d]).float(), 1,
                                      generator=g).item())
            draft.append(d)
        forwards += 1
        probe = torch.cat([seq, torch.tensor([draft], dtype=torch.long)], dim=1)
        q = F.softmax(model(probe)[0], dim=-1).numpy()   # target probs [len, V]
        start = seq.size(1) - 1
        out_toks: List[int] = []
        for i, tok in enumerate(draft):
            total += 1
            qx, px = q[start + i, tok], p[last, tok]
            u = rng.rand()
            if px <= 0 or u < min(1.0, qx / px):        # accept rule
                out_toks.append(tok)
                accepted += 1                            # DRAFT token accepted
                last = tok
            else:                                        # reject: resample from q-p
                residual = np.maximum(q[start + i] - p[last], 0.0)
                z = residual.sum()
                if z > 0:
                    corr = rng.choice(len(q[start + i]), p=residual / z)
                else:                                    # degenerate case: fall back
                    corr = int(q[start + i].argmax())
                out_toks.append(corr)                    # corrected token NOT
                last = corr                              # counted as accepted
                break
        seq = torch.cat([seq, torch.tensor([out_toks], dtype=torch.long)], dim=1)
    return seq, forwards, (accepted / total if total else 0.0)


# ----------------------------------------------------------------------------
# 5. GAMMA SWEEP: how many target forwards per 100 generated tokens?
# ----------------------------------------------------------------------------

def gamma_sweep(model: MiniGPT, logp: np.ndarray, ctx: torch.Tensor,
                n_new: int = 100) -> None:
    naive = n_new  # baseline: one forward per token
    print(f"[4] GAMMA SWEEP - target forwards to generate {n_new} tokens "
          f"(greedy; baseline = {naive}):")
    for gamma in (2, 3, 4, 6, 8):
        _, fwd = greedy_spec_decode(model, logp, ctx.clone(), n_new, gamma)
        bar = "#" * max(1, int(fwd / naive * 40))
        print(f"    gamma={gamma}: {fwd:4d} forwards "
              f"({naive / max(fwd, 1):4.2f}x savings) {bar}")


def main() -> None:
    print("=" * 72)
    print("SPECULATIVE DECODING LAB")
    print("=" * 72)
    model, tok, _, _, logp = build_models()
    V = tok.vocab_size
    # context: lift 24 chars of real prose from the corpus
    start = max(0, CORPUS.find(" the "))
    prompt = CORPUS[start : start + 24]
    ctx = torch.tensor(tok.encode(prompt), dtype=torch.long)[None, :]

    # --- [2] greedy correctness ---
    print(f"\n[1] GREEDY CORRECTNESS (prompt: {prompt!r})")
    t0 = time.perf_counter()
    seq_plain, fwd_plain = greedy_target_decode(model, ctx.clone(), 60), 60
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    seq_spec, fwd_spec = greedy_spec_decode(model, logp, ctx.clone(), 60, GAMMA)
    t_spec = time.perf_counter() - t0
    identical = bool(torch.equal(seq_plain, seq_spec))
    print(f"    plain greedy:  {fwd_plain} target forwards ({t_plain:.3f}s)")
    print(f"    spec  greedy:  {fwd_spec} target forwards ({t_spec:.3f}s)  "
          f"-> {fwd_plain / fwd_spec:.2f}x fewer")
    print(f"    outputs identical: {identical} "
          f"({'PASS - spec decoding changes nothing but the compute' if identical else 'FAIL'})")
    assert identical, "speculative decoding changed the greedy output!"
    assert fwd_spec < fwd_plain, "speculative decoding must save target forwards"
    print(f"    generated: {tok.decode(seq_spec[0].tolist())!r}")

    # --- [3] stochastic sampling ---
    print(f"\n[2] STOCHASTIC (rejection-sampling) mode, gamma={GAMMA}")
    s1, f1, acc1 = stochastic_spec_sample(model, logp, ctx.clone(), 60, GAMMA, seed=7)
    s2, f2, acc2 = stochastic_spec_sample(model, logp, ctx.clone(), 60, GAMMA, seed=11)
    print(f"    seed 7:  {f1} forwards | accept rate {acc1:.2f} -> {tok.decode(s1[0].tolist())!r}")
    print(f"    seed 11: {f2} forwards | accept rate {acc2:.2f} -> {tok.decode(s2[0].tolist())!r}")
    print(f"    different seeds -> different samples ({'PASS' if not torch.equal(s1, s2) else 'FAIL'})")
    assert not torch.equal(s1, s2), "sampling should be stochastic"
    print("    the sampled distribution IS the target distribution - the draft")
    print("    only proposes; q >= p tokens are kept, others resampled exactly.")

    # --- [4] gamma sweep ---
    print()
    gamma_sweep(model, logp, ctx.clone(), n_new=100)

    print("\n" + "=" * 72)
    print("SUMMARY: speculative decoding = draft cheaply, verify in batches.")
    print("Correctness is exact (proved above by token-identical output);")
    print("the savings come from trading cheap draft compute + one target")
    print("forward per gamma tokens for gamma sequential target forwards.")
    print("In production the target is a real LLM and the draft a small model")
    print("(or an n-gram / n-gram-style head) on the same vocabulary.")
    print("=" * 72)


if __name__ == "__main__":
    main()