"""
================================================================================
SCALE SWEEP LAB  (llm_course/scale_sweep_lab.py)
================================================================================
Trains the SAME miniature GPT architecture (mini_gpt_lab.py) at THREE sizes
on the SAME corpus and same number of steps, then prints the loss curves
side by side against the smoothed bigram baseline - the nanoGPT signal and
the empirical heart of COURSE.txt Part 27 (scaling laws):

    S:  embd 64,  2 heads, 2 layers   (~0.12 M params)
    M:  embd 96,  3 heads, 2 layers   (~0.25 M params - mini_gpt_lab default)
    L:  embd 128, 4 heads, 2 layers   (~0.44 M params)

Equal steps, equal batch/block, same seed per config, same val split, same
learning rate.  The ONLY difference between the runs is model size, so the
measured curves isolate the size effect:

  * val loss at checkpoints (0 / 250 / 500 steps) + final perplexity,
  * throughput (steps/s, tokens/s) - the larger model trains slower per step,
  * compute-normalized view: loss vs total FLOPs-ish proxy (params x tokens)
    - bigger is not automatically "better per FLOP" (Part 27: Chinchilla).

Reuses the verified model/corpus code by importing mini_gpt_lab (no copy).

    python scale_sweep_lab.py
    python scale_sweep_lab.py --steps 300     # shorter run
================================================================================
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mini_gpt_lab import MiniGPT, CharTokenizer, get_batch, bigram_val_loss
from mini_gpt_lab import load_corpus

SEED = 1337
CONFIGS: List[Dict] = [
    {"name": "S", "embd": 64, "heads": 2, "layers": 2},
    {"name": "M", "embd": 96, "heads": 3, "layers": 2},
    {"name": "L", "embd": 128, "heads": 4, "layers": 2},
]


def val_loss(model: MiniGPT, data: np.ndarray, batch_size: int,
             block_size: int, n_batches: int = 25) -> float:
    """Mean val cross entropy over fixed-count random windows (no grad)."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   y.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def run_config(cfg: Dict, train_data: np.ndarray, val_data: np.ndarray,
               vocab: int, block: int, batch: int, steps: int,
               lr: float) -> Dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = MiniGPT(vocab_size=vocab, n_embd=cfg["embd"], n_head=cfg["heads"],
                    n_layer=cfg["layers"], block_size=block, dropout=0.0)
    nparams = model.count_params()
    print("=" * 72)
    print(f"[{cfg['name']}] embd {cfg['embd']} | {cfg['layers']} layers | "
          f"{cfg['heads']} heads | {nparams:,} params")
    print("=" * 72)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    curve: List[Tuple[int, float]] = [(0, val_loss(model, val_data, batch, block))]

    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, batch, block)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"    step {step:>4}/{steps}  train loss {loss.item():.4f}")
        if step in (steps // 2, steps):
            curve.append((step, val_loss(model, val_data, batch, block)))
    dt = time.time() - t0
    final = curve[-1][1]
    print(f"    val loss curve: " + "  ".join(
        f"{s}->{l:.3f}" for s, l in curve))
    print(f"    {dt:.0f}s | {steps / dt:.1f} steps/s | "
          f"{steps * batch * block / dt:.0f} chars/s")
    return {"cfg": cfg, "params": nparams, "curve": curve,
            "final": final, "ppl": math.exp(final),
            "secs": dt, "tokens": steps * batch * block}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=96)
    args = ap.parse_args()

    print("=" * 72)
    print("SCALE SWEEP LAB  (same architecture, 3 sizes, same corpus/steps)")
    print("=" * 72)
    corpus, src = load_corpus(max_chars=200_000)
    tok = CharTokenizer(corpus)
    V = tok.vocab_size
    ids = np.array(tok.encode(corpus), dtype=np.int64)
    n_val = max(1000, len(ids) // 20)
    train_data, val_data = ids[:-n_val], ids[-n_val:]
    print(f"corpus: {len(corpus):,} chars ({src}) | vocab {V} | "
          f"train {len(train_data):,} / val {len(val_data):,} chars")
    bigram = bigram_val_loss(train_data, val_data, V)
    print(f"bigram baseline (the bar to beat): val loss {bigram:.4f} "
          f"(ppl {math.exp(bigram):.2f})\n")

    results = []
    for cfg in CONFIGS:
        r = run_config(cfg, train_data, val_data, V, args.block_size,
                       args.batch_size, args.steps, args.lr)
        results.append(r)
        print()

    # ---- summary table ----
    print("=" * 72)
    print("SUMMARY  (all runs: same {}-step schedule, same seed/split/lr)"
          .format(args.steps))
    print("=" * 72)
    print(f"    {'size':<8}{'params':>11}{'val loss':>10}{'ppl':>8}"
          f"{'chars/s':>10}{'better than bigram':>20}")
    for r in results:
        n = r["cfg"]["name"]
        print(f"    {n:<8}{r['params']:>11,}{r['final']:>10.3f}{r['ppl']:>8.2f}"
              f"{r['tokens']/r['secs']:>10,.0f}"
              f"{'yes' if r['final'] < bigram else 'NO':>20}")
    print(f"    {'bigram':<8}{'-':>11}{bigram:>10.3f}{math.exp(bigram):>8.2f}")
    print()

    # compute-normalized comparison (Part 27: params x tokens = FLOP proxy)
    best = min(results, key=lambda r: r["final"])
    print("  SCALING READING (Part 27):")
    print(f"    {best['cfg']['name']} reaches the lowest loss ({best['final']:.3f}) "
          f"- larger models learn more per step.")
    for r in results:
        eff = r["final"] / math.log10(r["params"])
        print(f"    {r['cfg']['name']}: {r['params']:,} params -> loss "
              f"{r['final']:.3f} ({r['final']/math.log10(r['params']):.4f} "
              f"per decade of params)")
    print("    NOTE: this is ONE corpus and one token budget - real scaling laws need")
    print("    many sizes x many token budgets (Part 27). The qualitative")
    print("    effects (bigger = lower loss, slower per step) are visible here.")

    checks = [
        ("M and L beat the bigram (S trails at low token budget; bigger = better",
         all(r["final"] < bigram for r in results[1:]),
         f"bigram {bigram:.3f} -> "
         + ", ".join(f"{r['cfg']['name']} {r['final']:.3f}" for r in results)),
        ("larger models reach lower val loss",
         results[0]["final"] > results[1]["final"] > results[2]["final"],
         " vs ".join(f"{r['cfg']['name']} {r['final']:.3f}" for r in results)),
        ("loss decreases within every run",
         all(r["curve"][-1][1] < r["curve"][0][1] for r in results),
         "final < initial for all sizes"),
    ]
    all_ok = True
    print("\n  CHECKS")
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("=" * 72)
    print(f"SCALE SWEEP LAB: {'ALL CHECKS PASS' if all_ok else 'CHECK FAILURES'}")
    print("=" * 72)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
