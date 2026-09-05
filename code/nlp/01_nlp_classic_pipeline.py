# ============================================================
# CLASSIC NLP PIPELINE — TF-IDF + LOGISTIC REGRESSION
# The pre-transformer baseline for any text classification task.
# Uses only scikit-learn; the corpus is generated inline so the
# script needs no downloads.
# Run: python 01_nlp_classic_pipeline.py
# ============================================================
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# ------------------------------------------------------------
# 1. Corpus generation: 3 classes x 120 docs, each doc = 1-2
#    class-specific phrases. Each class has its own vocabulary,
#    mirroring how real review corpora behave.
# ------------------------------------------------------------
POS = [
    "the product works perfectly", "absolutely love this product",
    "great quality for the price", "very satisfied with my purchase",
    "highly recommend to everyone", "excellent value and build quality",
    "best purchase this year", "amazing customer experience",
    "works flawlessly every day", "worth every penny",
]
NEG = [
    "terrible product it broke quickly", "do not buy waste of money",
    "poor quality and bad support", "not worth the price",
    "disappointed it stopped working", "horrible experience overall",
    "defective item received", "worst gadget I have owned",
    "frustrating and unreliable", "fell apart after a week",
]
NEUTRAL = [
    "the item arrived on time", "standard product nothing special",
    "package was well sealed", "delivery took a few days",
    "average quality overall", "matches the description",
    "shipping had no issues", "functions as expected",
    "ordered and received it", "product arrived as shown",
]

# Shared context phrases add vocabulary overlap across classes, so the
# task is not trivially separable (as real corpora are not).
FILLER = [
    "customer service handled my issue", "ordered from the online store",
    "the seller shipped it promptly", "read many reviews before buying",
    "asked for a refund through the app",
]

random.seed(0)
docs, labels = [], []
for label, bank in [("pos", POS), ("neg", NEG), ("neutral", NEUTRAL)]:
    for _ in range(120):
        n_phrases = random.randint(1, 2)
        parts = [random.choice(bank) for _ in range(n_phrases)]
        if random.random() < 0.25:                 # add a shared phrase
            parts.append(random.choice(FILLER))
        docs.append(" ".join(parts))
        labels.append(label)

# Inject 5% label noise — real human-labeled corpora always contain
# mislabels, and 100% accuracy would signal an unrealistically clean task.
classes = ["pos", "neg", "neutral"]
for i in range(len(labels)):
    if random.random() < 0.05:
        labels[i] = random.choice([c for c in classes if c != labels[i]])

# ------------------------------------------------------------
# 2. Pipeline: TF-IDF (lowercase, stopwords, unigrams+bigrams)
#    -> Logistic Regression
# ------------------------------------------------------------
pipe = make_pipeline(
    TfidfVectorizer(lowercase=True, stop_words="english",
                    ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000),
)

Xtr, Xte, ytr, yte = train_test_split(
    docs, labels, test_size=0.25, random_state=0, stratify=labels)
print(f"train={len(Xtr)} test={len(Xte)}")

pipe.fit(Xtr, ytr)
print(f"Accuracy: {pipe.score(Xte, yte):.3f}  "
      f"(not 1.0 on purpose: 5% of labels were flipped as label noise)")
print(classification_report(yte, pipe.predict(Xte), digits=2))

# ------------------------------------------------------------
# 3. Predictions on new text
# ------------------------------------------------------------
for text in ["works perfectly and great quality", "broke immediately, terrible",
             "item arrived after two days, package fine"]:
    print(f"'{text}' -> {pipe.predict([text])[0]}")

# ------------------------------------------------------------
# 4. Inspect what the model learned
# ------------------------------------------------------------
vec = pipe.named_steps["tfidfvectorizer"]
clf = pipe.named_steps["logisticregression"]
pos_idx = clf.classes_.tolist().index("pos")
weights = sorted(zip(vec.get_feature_names_out(), clf.coef_[pos_idx]),
                 key=lambda p: p[1], reverse=True)
print("Top positive indicators:", [w for w, _ in weights[:8]])

# ------------------------------------------------------------
# When to reach for this vs transformers:
#  - Small data, latency/cost constraints, or a debugging baseline:
#    TF-IDF + linear model is strong, fast, and transparent.
#  - Large data, synonyms/semantics matter, SOTA required:
#    fine-tune a transformer (see code/transformers/).
#  ALWAYS run this baseline first — it sets the difficulty ceiling
#  your deep model must beat.
# ============================================================
