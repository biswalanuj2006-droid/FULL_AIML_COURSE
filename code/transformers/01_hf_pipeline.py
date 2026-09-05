# ============================================================
# HUGGING FACE TRANSFORMERS — the three patterns you use daily
#   1. pipeline()  — quickest path to inference
#   2. tokenizer + model — explicit control
#   3. embeddings — features for search / fine-tuning
#
# Run: python 01_hf_pipeline.py
# Requires: pip install transformers sentence-transformers
#           (first run downloads model weights, ~100-400 MB)
# ============================================================
import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
except ImportError:
    raise SystemExit(
        "transformers not installed. Run: pip install transformers sentence-transformers"
    )

MODEL = "distilbert-base-uncased"   # small, fast, good for learning

# ------------------------------------------------------------
# 1. pipeline() — one line of inference
# ------------------------------------------------------------
classifier = pipeline("sentiment-analysis", model=MODEL)
result = classifier("Transformers make NLP much easier than it used to be.")
print("Sentiment:", result)

# ------------------------------------------------------------
# 2. tokenizer + model (explicit path — needed for fine-tuning)
# ------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)

texts = [
    "The service was outstanding.",
    "I have waited three weeks for a reply.",
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print("\nTokenized input shapes:", {k: tuple(v.shape) for k, v in inputs.items()})
print("Token ids:", inputs["input_ids"].tolist())

with torch.no_grad():
    outputs = model(**inputs)
    cls_vec = outputs.last_hidden_state[:, 0]      # [CLS] = sentence vector
print("CLS embedding shape:", tuple(cls_vec.shape))

# ------------------------------------------------------------
# 3. Semantic similarity with embeddings (cosine)
# ------------------------------------------------------------
a = cls_vec[0].numpy()
b = cls_vec[1].numpy()
cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print(f"\nCosine similarity between the two sentences: {cos_sim:.3f}")
print("(a low-ish score here is expected: opposite sentiments)")

# ------------------------------------------------------------
# Next steps (see module 26_TRANSFORMERS and 28_LLM_FUNDAMENTALS):
#  - Swap the model for a task-tuned one:
#      pipeline(\"text-classification\", model=\"cardiffnlp/twitter-roberta-base-sentiment-latest\")
#  - For fine-tuning use AutoModelForSequenceClassification
#    (see code/dl/ training loop pattern) or the HF Trainer.
#  - For better sentence vectors use sentence-transformers
#    (see code/rag/01_rag_embeddings.py).
# ============================================================
