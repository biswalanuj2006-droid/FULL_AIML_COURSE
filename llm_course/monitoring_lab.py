"""
================================================================================
LLM MONITORING LAB  (llm_course/monitoring_lab.py)
================================================================================
A runnable observability laboratory for COURSE.txt Part 57 (LLM
observability) and Part 58 (LLM cost).  A real LLM is out of scope (no
API keys, no network), so the "model" is a tiny deterministic fake that
produces a completion, a token count, and a latency per call -- exactly
the three signals a serving layer measures.  EVERYTHING else in the lab
is real production machinery:

  1. SPAN RECORDING: every request logs latency, prompt/completion tokens,
     model, cost in USD, cache hit, error flag (the Part 57 track list).
  2. METRICS: request rate, error rate, p50/p95 latency, tokens/second,
     aggregate cost, cache hit rate.
  3. TOKEN COSTING: per-model $/1M-token tables (input + output) and exact
     per-request cost arithmetic.
  4. CACHING: a response cache with a hit rate that measurably cuts both
     latency and cost.
  5. RETRIES: two injected upstream failures are retried once, so the
     client-visible error rate stays 0 while the raw error count is real.
  6. DRIFT DETECTION: a reference distribution of query topics (built
     online from the first N requests) vs. a sliding live window, scored
     with Jensen-Shannon divergence -- the concept-drift radar of
     Part 57/Module-46 style monitoring.
  7. ALERTS + DASHBOARD: threshold rules (p95 latency, error rate, cache
     hit rate, drift) fire as WARN/CRIT lines; a snapshot table is printed.

Runs with the standard library only (no torch, no network), < 10s:
    python monitoring_lab.py
================================================================================
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

random.seed(2026)

# ----------------------------------------------------------------------------
# 1. COST MODEL  (Part 58: token pricing, input vs output)
# ----------------------------------------------------------------------------
# price per 1M tokens, (input, output), in USD -- representative API prices.
PRICING: Dict[str, Tuple[float, float]] = {
    "small":  (0.15, 0.60),
    "medium": (0.50, 1.50),
    "large":  (2.50, 10.00),
}

TOPICS = ["docs", "code", "support", "finance", "math"]  # query-topic bins


def hash_topic(query: str) -> str:
    """Deterministic topic bin so traffic is reproducible."""
    h = int(hashlib.md5(query.encode("utf-8")).hexdigest()[:8], 16)
    return TOPICS[h % len(TOPICS)]


# ----------------------------------------------------------------------------
# 2. DATA MODEL
# ----------------------------------------------------------------------------

@dataclass
class Span:
    ts: float
    model: str
    topic: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool = False
    error: bool = False
    retried: bool = False
    status: int = 200


class Metrics:
    """Rolling aggregations over recorded spans (Part 57 metrics list)."""

    def __init__(self) -> None:
        self.spans: List[Span] = []
        self.errors: List[Span] = []

    def add(self, s: Span) -> None:
        self.spans.append(s)
        if s.error:
            self.errors.append(s)

    def pct(self, ms: List[float], q: float) -> float:
        if not ms:
            return 0.0
        s = sorted(ms)
        return s[min(len(s) - 1, int(q * len(s)))]

    def latency(self, q: float) -> float:
        return self.pct([s.latency_ms for s in self.spans if not s.error], q)

    def error_rate(self) -> float:
        return len(self.errors) / len(self.spans) if self.spans else 0.0

    def cache_hit_rate(self) -> float:
        if not self.spans:
            return 0.0
        return sum(1 for s in self.spans if s.cache_hit) / len(self.spans)

    def totals(self) -> Dict[str, float]:
        prom = sum(s.prompt_tokens for s in self.spans)
        comp = sum(s.completion_tokens for s in self.spans)
        ms = [s.latency_ms for s in self.spans if not s.error]
        dur = sum(ms) / 1000.0
        return {
            "requests": len(self.spans),
            "tokens": prom + comp,
            "tok_per_s": (prom + comp) / dur if dur > 0 else 0.0,
            "cost_usd": sum(s.cost_usd for s in self.spans),
            "prompt_tokens": prom,
            "completion_tokens": comp,
        }


# ----------------------------------------------------------------------------
# 3. FAKE MODEL  (latency + deterministic completion + token count)
# ----------------------------------------------------------------------------

class FakeModel:
    """Plays the role of the inference service: measurable, deterministic."""

    def __init__(self, model: str = "medium") -> None:
        self.model = model
        self.fail_left = {"inject-a": 1, "inject-b": 1}   # each fails once

    def generate(self, prompt: str) -> Tuple[str, float]:
        """Returns (completion, latency_ms).  Raises once per injected key."""
        for key in list(self.fail_left):
            if key in prompt and self.fail_left[key] > 0:
                self.fail_left[key] -= 1
                raise ConnectionError(f"upstream timeout for {key}")
        topic = hash_topic(prompt)
        words = prompt.split()
        completion = (f"ok[{topic}] {' '.join(words[-6:])} "
                      f"{random.randint(1, 100)}")
        # tail-latency realism: a few calls are slow (queue/GC effects)
        latency = 4.0 + 3.0 * random.random()
        if "slow" in prompt:
            latency += 60.0
        return completion, latency


def est_tokens(text: str) -> int:
    """Rough heuristic: ~4 chars per token (Part 12: tokenization affects cost)."""
    return max(1, len(text) // 4)


def cost_of(model: str, ptok: int, ctok: int) -> float:
    p_in, p_out = PRICING[model]
    return (ptok * p_in + ctok * p_out) / 1_000_000


class ResponseCache:
    """Prompt->completion cache (LRU-ish by recency list)."""

    def __init__(self, cap: int = 64) -> None:
        self.cap = cap
        self.store: Dict[str, Tuple[str, float]] = {}
        self.order: Deque[str] = deque()

    def get(self, prompt: str) -> Optional[Tuple[str, float]]:
        hit = self.store.get(prompt)
        if hit is not None:                       # refresh recency
            self.order.remove(prompt)
            self.order.append(prompt)
        return hit

    def put(self, prompt: str, completion: str, latency: float) -> None:
        if prompt in self.store:
            return
        self.store[prompt] = (completion, latency)
        self.order.append(prompt)
        while len(self.order) > self.cap:
            old = self.order.popleft()
            self.store.pop(old, None)


# ----------------------------------------------------------------------------
# 4. DRIFT DETECTOR  (Jensen-Shannon divergence over topic histograms)
# ----------------------------------------------------------------------------

def _norm(c: Counter) -> Dict[str, float]:
    total = sum(c.values()) or 1
    return {t: c[t] / total for t in TOPICS}


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Symmetric JS divergence between two discrete distributions."""
    m = {t: 0.5 * (p.get(t, 0.0) + q.get(t, 0.0)) for t in TOPICS}
    d = 0.0
    for t in TOPICS:
        for dist, name in ((p, "p"), (q, "q")):
            v = dist.get(t, 0.0)
            d += 0.5 * (v * (math_log(v) - math_log(m[t])) if v > 0 else 0.0)
    return d


def math_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


class DriftDetector:
    """Reference distribution built online; live window scored each request."""

    def __init__(self, warmup: int = 30, window: int = 15) -> None:
        self.warmup = warmup
        self.window = window
        self.ref: Counter = Counter()
        self.live: Deque[str] = deque()
        self.history: List[Tuple[int, float]] = []

    def observe(self, topic: str, i: int) -> Optional[float]:
        if i < self.warmup:
            self.ref[topic] += 1
            return None
        self.live.append(topic)
        if len(self.live) > self.window:
            self.live.popleft()
        js = js_divergence(_norm(self.ref), _norm(Counter(self.live)))
        self.history.append((i, js))
        return js


# ----------------------------------------------------------------------------
# 5. SERVING LAYER  (the thing monitoring watches)
# ----------------------------------------------------------------------------

class ServingLayer:
    def __init__(self, model: str = "medium") -> None:
        self.model_name = model
        self.model = FakeModel(model)
        self.cache = ResponseCache()
        self.metrics = Metrics()
        self.drift = DriftDetector()
        self.request_count = 0
        self.alerts: List[Tuple[str, str, str]] = []   # (severity, rule, msg)

    def _emit_alerts(self, span: Span, drift_js: Optional[float]) -> None:
        n = len(self.metrics.spans)
        if n >= 5:                                   # enough data to judge
            if self.metrics.latency(0.95) > 45.0 and span.latency_ms > 45.0:
                self.alerts.append(("WARN", "p95-latency",
                                    f"p95 {self.metrics.latency(0.95):.0f}ms > 45ms"))
        err_rate = self.metrics.error_rate()
        if err_rate > 0.15:
            self.alerts.append(("CRIT", "error-rate",
                                f"{err_rate:.0%} of requests failed"))
        if drift_js is not None and drift_js > 0.18:
            self.alerts.append(("WARN", "drift",
                                f"JS={drift_js:.2f}: live topic mix differs "
                                f"from reference"))

    def chat(self, prompt: str, retries: int = 1) -> Span:
        self.request_count += 1
        i = self.request_count
        t0 = time.perf_counter()
        topic = hash_topic(prompt)
        hit = self.cache.get(prompt)
        span = Span(ts=t0, model=self.model_name, topic=topic,
                    prompt_tokens=est_tokens(prompt),
                    completion_tokens=0, latency_ms=0.0,
                    cost_usd=0.0, cache_hit=hit is not None)
        if hit is not None:
            completion, _ = hit                      # cache: no model call
            span.completion_tokens = est_tokens(completion)
            span.cost_usd = cost_of(self.model_name, 0, 0)  # ~0 token cost
            span.latency_ms = (time.perf_counter() - t0) * 1000
            self.metrics.add(span)
            return span
        attempt = 0
        while True:
            try:
                completion, model_ms = self.model.generate(prompt)
                self.cache.put(prompt, completion, model_ms)
                span.completion_tokens = est_tokens(completion)
                span.cost_usd = cost_of(self.model_name,
                                        span.prompt_tokens, span.completion_tokens)
                span.latency_ms = (time.perf_counter() - t0) * 1000 + model_ms
                span.status = 200
                if attempt > 0:
                    span.retried = True
                self.metrics.add(span)
                self._emit_alerts(span, self.drift.observe(topic, i))
                return span
            except ConnectionError:
                attempt += 1
                if attempt > retries:
                    span.error = True
                    span.latency_ms = (time.perf_counter() - t0) * 1000
                    span.status = 503
                    self.metrics.add(span)
                    return span
                time.sleep(0.002 * attempt)          # tiny backoff


# ----------------------------------------------------------------------------
# 6. TRAFFIC SCRIPT  (phases: normal -> cache reuse -> drift -> failures)
# ----------------------------------------------------------------------------

def make_queries(n: int, topics: List[str]) -> List[str]:
    out = []
    words = {
        "docs": ["how do i", "where is", "what does", "explain", "documentation"],
        "code": ["write code", "debug", "refactor", "function", "api"],
        "support": ["help", "error", "broken", "issue", "does not work"],
        "finance": ["budget", "invoice", "revenue", "tax", "quarterly report"],
        "math": ["prove", "derive", "theorem", "lemma", "gradient", "integral"],
    }
    for _ in range(n):
        t = topics[_ % len(topics)]
        body = random.choice(words[t])
        extra = random.randint(0, 999)
        out.append(f"{body} {t} item {extra}")
    return out


def main() -> None:
    t0 = time.time()
    print("=" * 72)
    print("LLM MONITORING LAB  (Part 57 observability + Part 58 cost)")
    print("=" * 72)
    print("pricing ($/1M tokens, input/output): "
          + ", ".join(f"{m} {p[0]}/{p[1]}" for m, p in PRICING.items()))
    svc = ServingLayer("medium")
    print("model: medium  | traffic phases: normal(25) -> cache-replay(6)\n"
          "      -> drift(18 math-heavy) -> injected upstream failures\n")

    # Phase 1: normal traffic (mixed topics) -- builds the reference dist.
    phase1 = make_queries(25, ["docs", "docs", "code", "support", "code"])
    for q in phase1:
        svc.chat(q)

    # Phase 2: exact repeats -> cache hits (latency + cost drop).
    for q in phase1[:6]:
        svc.chat(q)

    # Phase 3: concept drift -- suddenly most queries are math/finance.
    phase3 = make_queries(18, ["math", "math", "finance", "math"])

    # Phase 4: two queries that fail upstream once, then succeed on retry.
    phase3[0] = "inject-a calculate the budget slowly"       # slow + fails once
    phase3[1] = "inject-b prove the lemma about gradients"   # fails once

    for q in phase3:
        svc.chat(q)

    # ---- dashboard snapshot ----
    m = svc.metrics
    tot = m.totals()
    print("=" * 72)
    print("DASHBOARD SNAPSHOT")
    print("=" * 72)
    print(f"  requests          : {tot['requests']}")
    print(f"  tokens            : {tot['tokens']:,} "
          f"(prompt {tot['prompt_tokens']:,} / completion {tot['completion_tokens']:,})")
    print(f"  throughput        : {tot['tok_per_s']:.0f} tok/s")
    print(f"  latency p50 / p95 : {m.latency(0.50):6.1f} ms / {m.latency(0.95):6.1f} ms")
    print(f"  error rate        : {m.error_rate():.1%}  "
          f"(client-visible after retry: "
          f"{sum(1 for s in m.spans if s.status != 200) / len(m.spans):.1%})")
    print(f"  cache hit rate    : {m.cache_hit_rate():.1%}")
    print(f"  cost              : ${tot['cost_usd']:.4f}")

    # cost breakdown: what caching saved
    cost_all = sum(s.cost_usd for s in m.spans)
    cost_nocache = sum(s.cost_usd for s in m.spans if not s.cache_hit)
    replays = [s for s in m.spans if s.cache_hit]
    saved = sum(cost_of("medium", s.prompt_tokens, s.completion_tokens)
                for s in replays)
    print(f"  cache savings     : ${saved:.4f} avoided "
          f"({len(replays)} replayed prompts)")

    # drift chart (Part 57: watch the distribution shift over time)
    print("\n  DRIFT (JS divergence, live 15-req window vs reference):")
    hist = svc.drift.history
    if hist:
        scale = 40.0 / (max(h[1] for h in hist) or 1.0)
        for i, js in hist[::2]:                        # thin out for the plot
            bar = "#" * int(js * scale)
            print(f"    req {i:>3}  JS={js:.2f} |{bar}")
        max_js = max(h[1] for h in hist)
        print(f"    peak JS = {max_js:.2f} "
              f"({'DRIFT DETECTED (phase 3 changed the topic mix)' if max_js > 0.18 else 'stable'})")

    # alerts
    print("\n  ALERTS RAISED:")
    if svc.alerts:
        for sev, rule, msg in svc.alerts[-6:]:
            print(f"    [{sev:>4}] {rule:<12} {msg}")
    else:
        print("    (none)")

    # retry evidence
    retried = [s for s in m.spans if s.retried]
    failed = [s for s in m.spans if s.error]
    print("\n  RETRIES (Part 43/44: transient failures are retried, not surfaced):")
    print(f"    upstream failures injected : 2")
    print(f"    retried + succeeded        : {len(retried)}")
    print(f"    still failed after retry   : {len(failed)}")

    # ---- checks ----
    checks = [
        ("cache replays hit the cache",
         sum(1 for s in m.spans if s.cache_hit) >= 6,
         f"{sum(1 for s in m.spans if s.cache_hit)} hits"),
        ("cache cuts cost", saved > 0,
         f"${saved:.4f} avoided"),
        ("failures recovered by retry",
         len(failed) == 0 and len(retried) == 2,
         f"{len(retried)} retried, {len(failed)} lost"),
        ("drift detector fires during phase 3",
         max_js > 0.18 if hist else False,
         f"peak JS={max_js:.2f}" if hist else "no data"),
        ("p95 latency metric present",
         m.latency(0.95) > m.latency(0.50),
         f"p50 {m.latency(0.50):.0f}ms < p95 {m.latency(0.95):.0f}ms"),
    ]
    all_ok = True
    print("\n  CHECKS")
    for name, ok, detail in checks:
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")
    print("=" * 72)
    print(f"MONITORING LAB: {'ALL CHECKS PASS' if all_ok else 'CHECK FAILURES'} "
          f"({time.time() - t0:.0f}s)")
    print("=" * 72)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
