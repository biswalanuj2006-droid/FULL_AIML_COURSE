# ============================================================
# PYTHON STANDARD LIBRARY FOR AI/ML ENGINEERS
# The modules below appear in almost every real ML project.
# Run: python 01_stdlib_for_ai.py  (no third-party deps)
# ============================================================
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

# ------------------------------------------------------------
# 1. pathlib + os — file and directory handling
# ------------------------------------------------------------
p = Path(".") / "example.json"
print("Path:", p.resolve())
print("Exists:", p.exists())

# ------------------------------------------------------------
# 2. json + csv — serialization
# ------------------------------------------------------------
records = {"name": "model-a", "metrics": {"acc": 0.91, "auc": 0.94}, "tags": ["cv", "v1"]}
p.write_text(json.dumps(records, indent=2))
loaded = json.loads(p.read_text())
print("JSON round-trip ok:", loaded == records)
p.unlink()

# ------------------------------------------------------------
# 3. re — text preprocessing in NLP
# ------------------------------------------------------------
text = "  Model  loss:0.1234, acc:0.9821  "
clean = re.sub(r"\s+", " ", text.strip())
numbers = re.findall(r"[-+]?\d*\.\d+", clean)
print("Cleaned:", clean)
print("Numbers:", numbers)

# ------------------------------------------------------------
# 4. collections — counters, defaults, queues
# ------------------------------------------------------------
labels = ["cat", "dog", "cat", "bird", "cat", "dog"]
counts = Counter(labels)
print("Label counts:", dict(counts))

d = defaultdict(list)
d["train"].append("x1")            # no KeyError for missing keys
print("defaultdict:", dict(d))

q = deque(maxlen=3)                # O(1) both ends; fixed-size buffer
for i in range(10):
    q.append(i)                    # sliding window of last 3 values
print("Sliding-window buffer:", list(q))

# ------------------------------------------------------------
# 5. dataclasses + typing — clean config objects
# ------------------------------------------------------------
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "mlp"


cfg = TrainingConfig(epochs=20)
print("Config:", cfg)
print("Config as dict:", asdict(cfg))      # handy for MLflow logging

# ------------------------------------------------------------
# 6. logging — structured, level-based output (prefer over print)
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train")
log.info("Training started with lr=%s", cfg.learning_rate)
log.warning("Early stopping patience not set — using default")

# ------------------------------------------------------------
# 7. math / random / time — numerics, seeding, timing
# ------------------------------------------------------------
random.seed(42)                     # reproducibility
print("Seeded shuffle:", sorted(random.sample(range(10), 5)))
print("Sigmoid(0):", 1 / (1 + math.exp(-0)))

t0 = time.perf_counter()
_ = sum(i * i for i in range(1_000_000))
print(f"Timed computation: {(time.perf_counter() - t0) * 1000:.1f} ms")

# ------------------------------------------------------------
# When to reach for each in an ML project:
#  pathlib        -> everywhere files are read/written
#  json           -> configs, API payloads, experiment logs
#  re             -> text cleaning before tokenizers
#  collections    -> Counting labels, vocab building, sliding buffers
#  dataclasses    -> config objects; asdict() for experiment tracking
#  logging        -> training logs, API logs (never print in prod)
#  argparse       -> CLI entry points (see 02_argparse_cli.py)
# ============================================================
