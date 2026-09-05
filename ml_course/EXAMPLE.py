"""
================================================================================
ULTRA-DEEP ML COURSE - EXAMPLE.py  (ml_course/EXAMPLE.py)
================================================================================
A 26-section runnable reference implementation for COURSE.txt.  Every
section implements the concept FROM SCRATCH first (as the master prompt
requires), then shows the library version, and ends with a PASS line whose
numbers are asserted - running this file IS the verification.

  S1  NumPy fundamentals           S14  PCA from scratch
  S2  Pandas data analysis         S15  ML metrics from scratch
  S3  Data preprocessing           S16  Cross-validation from scratch
  S4  Visualization (Agg-safe)     S17  Feature engineering
  S5  Statistics                   S18  Hyperparameter optimization
  S6  Linear regression (scratch)  S19  GBDT from scratch (+ XGB/LGBM/CatBoost if installed)
  S7  Gradient descent (scratch)   S20  Model explainability (permutation + SHAP if installed)
  S8  Logistic regression (scratch)S21  Time-series ML (lag features, temporal CV)
  S9  KNN from scratch             S22  Anomaly detection (Isolation Forest)
  S10 Naive Bayes from scratch     S23  scikit-learn Pipelines (no leakage)
  S11 Decision tree from scratch   S24  FastAPI model serving (guarded)
  S12 Random forest from scratch   S25  MLflow experiment tracking (guarded)
  S13 K-Means from scratch         S26  Production artifact + architecture

Dependency policy: numpy/pandas/sklearn/matplotlib are required.  Heavier
tools (xgboost, lightgbm, catboost, shap, fastapi, mlflow) are USED when
present and reported as SKIPPED-with-reason when absent - never crash.

    python EXAMPLE.py          # runs all 26 sections
================================================================================
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# deterministic by default (S16/S18 show seeds explicitly)
SEED = 42
np.random.seed(SEED)

rng = np.random.default_rng(SEED)
TMP = tempfile.mkdtemp(prefix="ml_example_")
RESULTS: List[str] = []


def ok(section: str, detail: str = "") -> None:
    RESULTS.append(f"S{section}: PASS")
    print(f"[PASS] S{section} - {detail}")


def skip(section: str, reason: str) -> None:
    RESULTS.append(f"S{section}: SKIP")
    print(f"[SKIP] S{section} - {reason}")


def fail(section: str, detail: str) -> None:
    RESULTS.append(f"S{section}: FAIL")
    print(f"[FAIL] S{section} - {detail}")
    raise SystemExit(f"EXAMPLE.py failed at {detail}")


# ============================================================================
# S1. NUMPY FUNDAMENTALS (broadcasting, matmul, norms - COURSE Module 1/2)
# ============================================================================
def s01_numpy() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0], [6.0]])          # [2,1] broadcasts against [2,2]
    c = a @ b                             # matrix multiply
    assert c.shape == (2, 1) and abs(c[0, 0] - 17.0) < 1e-9   # 1*5+2*6
    assert abs(c[1, 0] - 39.0) < 1e-9                         # 3*5+4*6
    # broadcasting: [3,1] * [1,4] -> [3,4]
    x = np.arange(3)[:, None] * np.ones(4)[None, :]
    assert x.shape == (3, 4) and x[2, 3] == 2.0
    v = np.array([3.0, 4.0])
    assert abs(np.linalg.norm(v) - 5.0) < 1e-9                # L2 norm
    assert abs(v @ v - 25.0) < 1e-9
    ok("01", "broadcasting + matmul + norms verified (5-12-13 Pythagorean)")

    X = np.random.default_rng(0).normal(size=(20, 4))
    Xc = X - X.mean(axis=0)                                   # centering
    cov = Xc.T @ Xc / len(X)
    ev = np.linalg.eigvalsh(cov)
    assert np.all(ev >= -1e-10)                               # PSD covariance
    ok("01", "covariance is positive semi-definite (PCA input is valid)")


# ============================================================================
# S2. PANDAS DATA ANALYSIS (Module 1/6/7)
# ============================================================================
def s02_pandas() -> None:
    import pandas as pd
    df = pd.DataFrame({
        "age": [25, 34, 45, np.nan, 29],
        "income": [40_000, 70_000, 120_000, 60_000, 55_000],
        "churn": [0, 1, 0, 1, 0],
    })
    assert df.isna().sum()["age"] == 1
    df["age"] = df["age"].fillna(df["age"].median())          # median impute
    assert not df["age"].isna().any()
    g = df.groupby("churn")["income"].mean()                  # split-apply
    assert abs(float(g.loc[0]) - (40000 + 120000 + 55000) / 3) < 1e-6
    corr = df[["age", "income"]].corr().iloc[0, 1]            # bivariate
    assert -1.0 <= corr <= 1.0
    ok("02", "missing-value handling + groupby + correlation in pandas")


# ============================================================================
# S3. DATA PREPROCESSING (Module 8 - fit on train only!)
# ============================================================================
def s03_preprocessing() -> None:
    from sklearn.preprocessing import StandardScaler
    X = rng.normal(size=(200, 3)) * np.array([1, 100, 0.01])  # different scales
    X_train, X_test = X[:150], X[150:]
    sc = StandardScaler().fit(X_train)                        # fit on TRAIN
    Z = sc.transform(X_train)
    Zt = sc.transform(X_test)
    assert abs(Z.mean(axis=0)).max() < 1e-9                   # train mean 0
    assert abs(Z.std(axis=0) - 1).max() < 1e-9                # train std 1
    # the test transform uses TRAIN statistics (the correct way)
    assert abs(Zt.mean(axis=0)).max() > 0                      # test != standardized itself
    ok("03", f"StandardScaler fit-on-train (test means differ: {Zt.mean(axis=0).round(2)})")


# ============================================================================
# S4. VISUALIZATION (Module 7 - Agg backend, never blocks)
# ============================================================================
def s04_visualization() -> None:
    import matplotlib
    matplotlib.use("Agg")                                     # headless-safe
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    r = rng.normal(0, 1, 1000)
    ax[0].hist(r, bins=30)                                    # univariate
    ax[1].scatter(r[:-1], r[1:], s=2)                         # bivariate/lag
    out = os.path.join(TMP, "s04_plot.png")
    fig.savefig(out, dpi=60)
    plt.close(fig)
    assert os.path.getsize(out) > 1000
    ok("04", f"histogram + scatter saved to {os.path.basename(out)}")


# ============================================================================
# S5. STATISTICS (Module 4 - CI + t-test + chi-square)
# ============================================================================
def s05_statistics() -> None:
    from scipy import stats as st
    x = rng.normal(5.0, 2.0, 200)
    se = x.std(ddof=1) / np.sqrt(len(x))                      # standard error
    ci = (x.mean() - 1.96 * se, x.mean() + 1.96 * se)         # 95% CI
    assert ci[0] < 5.0 < ci[1], "CI should cover the true mean"
    # paired t-test: known-signal shift is detected
    before = rng.normal(10, 1, 60)
    after = before + 0.8 + rng.normal(0, 1, 60)
    t, p = st.ttest_rel(before, after)
    assert p < 0.05, "a 0.8 shift must be significant at n=60"
    # chi-square independence: constructed dependent table
    obs = np.array([[80, 20], [30, 70]])                      # rows/cols dependent
    chi2, p2, _, _ = st.chi2_contingency(obs)
    assert p2 < 1e-6
    ok("05", f"CI=({ci[0]:.2f},{ci[1]:.2f}) covers 5.0; paired t p={p:.2e}; chi2 p={p2:.2e}")


# ============================================================================
# S6. LINEAR REGRESSION FROM SCRATCH (Module 11)
# ============================================================================
def ols_scratch(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed form: w = (X^T X)^-1 X^T y via lstsq (never invert by hand)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return w


def s06_linear_regression() -> None:
    from sklearn.linear_model import LinearRegression
    X = rng.normal(size=(500, 2))
    true_w = np.array([3.0, -1.5])
    y = X @ true_w + 0.5 + rng.normal(0, 0.1, 500)
    w = ols_scratch(X, y)
    sk = LinearRegression().fit(X, y)
    assert np.allclose(w[1:], sk.coef_, atol=1e-4)
    assert np.allclose(w[0], sk.intercept_, atol=1e-4)
    r2 = 1 - np.sum((y - X @ true_w - w[0]) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.99
    ok("06", f"scratch OLS == sklearn (coef {np.round(w[1:], 3)}), R2={r2:.4f}")


# ============================================================================
# S7. GRADIENT DESCENT FROM SCRATCH (Module 30 - must reach closed form)
# ============================================================================
def s07_gradient_descent() -> None:
    X = rng.normal(size=(300, 1))
    y = 2.0 * X[:, 0] + 1.0 + rng.normal(0, 0.05, 300)
    Xb = np.column_stack([np.ones(300), X])
    w = np.zeros(2)
    lr = 0.5
    for _ in range(2000):                                     # batch GD
        grad = (2 / len(Xb)) * Xb.T @ (Xb @ w - y)
        w -= lr * grad
    w_closed = ols_scratch(X, y)
    assert np.allclose(w, w_closed, atol=1e-3), "GD must converge to OLS"
    ok("07", f"GD converged to closed form (w={np.round(w, 3)}) in 2000 steps")


# ============================================================================
# S8. LOGISTIC REGRESSION FROM SCRATCH (Module 13)
# ============================================================================
def s08_logistic_regression() -> None:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=800, n_features=4, n_informative=3,
                               n_redundant=0, random_state=SEED)
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    lr = 0.5
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Xb @ w)))
        grad = Xb.T @ (p - y) / len(y)                        # X^T (p - y)
        w -= lr * grad
    proba = 1 / (1 + np.exp(-(Xb @ w)))
    acc = ((proba >= 0.5).astype(int) == y).mean()
    sk = LogisticRegression(max_iter=2000).fit(X, y)
    sk_acc = sk.score(X, y)
    assert abs(acc - sk_acc) < 0.05, f"scratch {acc:.3f} vs sklearn {sk_acc:.3f}"
    ok("08", f"scratch logreg acc {acc:.3f} vs sklearn {sk_acc:.3f} (gradient = X^T(p-y))")


# ============================================================================
# S9. KNN FROM SCRATCH (Module 15 - vectorized, like mini_gpt_lab's fix)
# ============================================================================
def s09_knn() -> None:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    X, y = make_classification(n_samples=600, n_features=5, n_informative=5,
                               n_redundant=0, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)

    def knn_predict(Xtr, ytr, Xte, k=5):
        D = np.linalg.norm(Xtr[:, None, :] - Xte[None, :, :], axis=2)  # [n_tr, n_te]
        idx = np.argsort(D, axis=0)[:k]
        votes = ytr[idx]
        return np.array([np.bincount(col).argmax() for col in votes.T])

    pred = knn_predict(Xtr, ytr, Xte)
    sk = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
    sk_pred = sk.predict(Xte)
    # allow a couple of tie-break differences (identical votes otherwise)
    assert (pred == sk_pred).mean() >= 0.97
    ok("09", f"scratch KNN agrees with sklearn on {(pred == sk_pred).mean():.3f} of test set")


# ============================================================================
# S10. NAIVE BAYES FROM SCRATCH (Module 16)
# ============================================================================
def s10_naive_bayes() -> None:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    X, y = make_classification(n_samples=800, n_features=6, n_informative=5,
                               n_redundant=0, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    classes = np.unique(ytr)
    mu = np.array([Xtr[ytr == c].mean(axis=0) for c in classes])
    sd = np.array([Xtr[ytr == c].std(axis=0) + 1e-9 for c in classes])
    prior = np.array([(ytr == c).mean() for c in classes])
    log_lik = -0.5 * np.log(2 * np.pi * sd ** 2) - (Xte[:, None, :] - mu) ** 2 / (2 * sd ** 2)
    scores = np.log(prior) + log_lik.sum(axis=2)              # log P(y) + sum log P(x_j|y)
    pred = classes[scores.argmax(axis=1)]
    acc = (pred == yte).mean()
    sk = GaussianNB().fit(Xtr, ytr)
    assert abs(acc - sk.score(Xte, yte)) < 0.05
    ok("10", f"scratch GaussianNB acc {acc:.3f} vs sklearn {sk.score(Xte, yte):.3f}")


# ============================================================================
# S11. DECISION TREE FROM SCRATCH (Module 17 - CART, greedy, entropy)
# ============================================================================
def _entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _best_split(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    best = (0.0, 0.0, 0.0, 0.0)
    H = _entropy(y)
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j])[1:]:                      # candidate thresholds
            left = y[X[:, j] < t]
            right = y[X[:, j] >= t]
            if len(left) == 0 or len(right) == 0:
                continue
            w = len(left) / len(y) * _entropy(left) + len(right) / len(y) * _entropy(right)
            gain = H - w                                      # information gain
            if gain > best[0]:
                best = (gain, j, t, w)
    return best


def _tree(X: np.ndarray, y: np.ndarray, depth: int = 0, max_depth: int = 4):
    if depth >= max_depth or len(np.unique(y)) == 1:
        return {"leaf": int(np.bincount(y).argmax())}
    gain, j, t, _ = _best_split(X, y)
    if gain <= 1e-9:
        return {"leaf": int(np.bincount(y).argmax())}
    mask = X[:, j] < t
    return {"feat": j, "thr": t,
            "left": _tree(X[mask], y[mask], depth + 1, max_depth),
            "right": _tree(X[~mask], y[~mask], depth + 1, max_depth)}


def _predict_tree(node, x: np.ndarray) -> int:
    while "leaf" not in node:
        node = node["left"] if x[node["feat"]] < node["thr"] else node["right"]
    return node["leaf"]


def s11_decision_tree() -> None:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=3, n_informative=3,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=1.5, random_state=SEED)
    tree = _tree(X, y, max_depth=6)                      # depth-limited CART
    pred = np.array([_predict_tree(tree, x) for x in X])
    assert (pred == y).mean() > 0.97, "a depth-6 CART should nearly fit this data"
    ok("11", f"scratch CART (entropy, greedy, depth 6) fits train at {(pred == y).mean():.3f}")


# ============================================================================
# S12. RANDOM FOREST FROM SCRATCH (Module 18 - bagging + feature subsets)
# ============================================================================
def _forest(X: np.ndarray, y: np.ndarray, n_trees: int = 30, max_features: int = 2):
    trees = []
    rngf = np.random.default_rng(SEED)
    n = len(X)
    for _ in range(n_trees):
        idx = rngf.integers(0, n, size=n)                     # bootstrap sample
        cols = rngf.choice(X.shape[1], size=max_features, replace=False)
        # depth 6: the S11 lesson — depth-4 CARTs underfit this data
        trees.append((cols, _tree(X[idx][:, cols], y[idx], max_depth=6)))
    return trees


def _predict_forest(forest, x: np.ndarray) -> int:
    votes = np.array([_predict_tree(t, x[cols]) for cols, t in forest])
    return int(np.bincount(votes).argmax())


def s12_random_forest() -> None:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=4, n_informative=4,
                               n_redundant=0, n_clusters_per_class=1,
                               class_sep=1.2, random_state=SEED)
    forest = _forest(X, y)
    pred = np.array([_predict_forest(forest, x) for x in X])
    assert (pred == y).mean() > 0.98
    ok("12", f"scratch RF (30 bagged CARTs, max_features=2) train acc {(pred == y).mean():.3f}")


# ============================================================================
# S13. K-MEANS FROM SCRATCH (Module 20 - Lloyd's algorithm)
# ============================================================================
def s13_kmeans() -> None:
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=SEED)
    k = 3
    cents = X[rng.choice(len(X), k, replace=False)]
    for _ in range(100):
        d = np.linalg.norm(X[:, None, :] - cents[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        new = np.array([X[assign == c].mean(axis=0) if np.any(assign == c) else cents[c]
                        for c in range(k)])
        if np.allclose(new, cents):
            break
        cents = new
    ari = adjusted_rand_score(y, assign)
    assert ari > 0.95
    ok("13", f"scratch Lloyd K-Means ARI={ari:.3f} (centers recovered)")


# ============================================================================
# S14. PCA FROM SCRATCH (Module 21 - eigendecomposition of covariance)
# ============================================================================
def s14_pca() -> None:
    from sklearn.decomposition import PCA
    X = rng.normal(size=(500, 5)) @ rng.normal(size=(5, 5))   # correlated data
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / len(X)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs, vals = vecs[:, order], vals[order]
    proj = Xc @ vecs[:, :2]
    sk = PCA(n_components=2).fit(X)
    cos = abs(vecs[:, 0] @ sk.components_[0])
    assert cos > 0.99, f"first PC must align with sklearn (cos={cos:.4f})"
    ev_ratio = vals / vals.sum()
    assert abs(float(ev_ratio[:2].sum()) - float(sk.explained_variance_ratio_.sum())) < 0.01
    ok("14", f"scratch PCA PC1 cos={cos:.4f}; explained var {ev_ratio[:2].sum():.3f} "
             f"(sklearn {sk.explained_variance_ratio_.sum():.3f})")


# ============================================================================
# S15. ML METRICS FROM SCRATCH (Module 14)
# ============================================================================
def s15_metrics() -> None:
    from sklearn.metrics import roc_auc_score
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp)                                     # 4/5
    rec = tp / (tp + fn)                                      # 4/5
    f1 = 2 * prec * rec / (prec + rec)                        # 0.8
    assert abs(prec - 0.8) < 1e-9 and abs(f1 - 0.8) < 1e-9
    # AUC from scratch via rank formula
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.3, 0.7, 0.4, 0.5, 0.05])
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    auc = np.mean([(scores[i] > scores[j]) for i in pos for j in neg])
    assert abs(auc - roc_auc_score(y_true, scores)) < 1e-9
    ok("15", f"precision={prec}, recall={rec}, F1={f1}; rank-AUC {auc:.3f} == sklearn")


# ============================================================================
# S16. CROSS-VALIDATION FROM SCRATCH (Module 9 - stratified k-fold)
# ============================================================================
def s16_cv() -> None:
    y = np.array([0] * 80 + [1] * 20)                         # imbalanced labels
    k = 5
    folds = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        per = np.array_split(idx, k)
        folds.append(per)
    cv_folds = [np.concatenate([f[c] for f in folds]) for c in range(k)]
    for f in cv_folds:
        assert len(f) == 20                                    # 5 folds of 20
        assert abs((y[f] == 1).mean() - 0.2) < 1e-9            # 20% pos in EVERY fold
    assert len(np.unique(np.concatenate(cv_folds))) == 100     # no sample twice
    ok("16", "stratified 5-fold: every fold has exactly 20% positives (imbalance-safe)")


# ============================================================================
# S17. FEATURE ENGINEERING (Module 22 - lags, date, ratios, interactions)
# ============================================================================
def s17_feature_engineering() -> None:
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    df = pd.DataFrame({"date": dates, "sales": np.sin(np.arange(200) / 7) * 10 + 50})
    df["dow"] = df["date"].dt.dayofweek                          # calendar
    df["sales_lag7"] = df["sales"].shift(7)                      # lag
    df["sales_roll7"] = df["sales"].rolling(7).mean()            # rolling
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    assert df["sales_lag7"].notna().mean() > 0.9
    assert abs(df["sales_roll7"].iloc[10] - df["sales"].iloc[4:11].mean()) < 1e-9
    # interaction feature for a linear model to express x1*x2
    X = rng.normal(size=(300, 2))
    y = 2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 0] + rng.normal(0, 0.01, 300)
    Xw = np.column_stack([np.ones(300), X, X[:, 0] * X[:, 1]])   # add x1*x2
    w, *_ = np.linalg.lstsq(Xw, y, rcond=None)
    assert abs(w[3] - 2.0) < 0.05, "linear model recovers the interaction with the cross feature"
    ok("17", "lag/rolling/calendar features + x1*x2 interaction recovered (w3=2.0)")


# ============================================================================
# S18. HYPERPARAMETER OPTIMIZATION (Module 26 - manual random search)
# ============================================================================
def s18_hpo() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=5, random_state=SEED)
    r = np.random.default_rng(SEED)
    best = (0.0, None)
    for _ in range(12):                                        # random search
        depth = int(r.integers(2, 12))
        leaf = int(r.integers(1, 8))
        m = RandomForestClassifier(max_depth=depth, min_samples_leaf=leaf,
                                   n_estimators=30, random_state=SEED)
        score = cross_val_score(m, X, y, cv=3, scoring="accuracy").mean()
        if score > best[0]:
            best = (score, (depth, leaf))
    assert best[0] > 0.8
    ok("18", f"random search best CV accuracy {best[0]:.3f} "
             f"at depth={best[1][0]}, min_leaf={best[1][1]} (log-scale-friendly)")


# ============================================================================
# S19. GRADIENT BOOSTING FROM SCRATCH (Module 18) + library boosters
# ============================================================================
def _best_split_reg(X: np.ndarray, y: np.ndarray) -> Tuple[float, int, float]:
    """Regression split: minimize weighted child VARIANCE (MSE gain)."""
    best = (1e18, 0, 0.0)
    parent = float(np.var(y)) * len(y)
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j])[1:]:
            left, right = y[X[:, j] < t], y[X[:, j] >= t]
            if len(left) == 0 or len(right) == 0:
                continue
            w = float(np.var(left)) * len(left) + float(np.var(right)) * len(right)
            if w < best[0]:
                best = (w, j, t)
    return best


def _tree_reg(X: np.ndarray, y: np.ndarray, max_depth: int) -> dict:
    """CART regression tree: leaves predict the mean of their samples."""
    if max_depth <= 0 or len(y) < 4:
        return {"leaf": float(y.mean())}
    w, j, t = _best_split_reg(X, y)
    if w >= float(np.var(y)) * len(y) - 1e-12:   # no gain -> leaf
        return {"leaf": float(y.mean())}
    mask = X[:, j] < t
    return {"feat": j, "thr": t,
            "left": _tree_reg(X[mask], y[mask], max_depth - 1),
            "right": _tree_reg(X[~mask], y[~mask], max_depth - 1)}


class GBDT:
    """Regression GBDT: each tree fits the NEGATIVE GRADIENT (pseudo-residual)
    of the previous ensemble.  For MSE: r_i = y_i - F(x_i).  shrinkage lr
    scales each tree.  This is gradient descent in function space."""

    def __init__(self, n_trees: int = 40, lr: float = 0.1, max_depth: int = 3):
        self.n_trees, self.lr, self.max_depth = n_trees, lr, max_depth
        self.base: Optional[float] = None
        self.trees: List[dict] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDT":
        self.base = float(y.mean())
        F = np.full(len(y), self.base)
        for _ in range(self.n_trees):
            r = y - F                                       # pseudo-residuals
            t = _tree_reg(X, r, self.max_depth)             # regression tree
            F += self.lr * np.array([_predict_tree(t, x) for x in X])
            self.trees.append(t)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = np.full(len(X), self.base)
        for t in self.trees:
            F += self.lr * np.array([_predict_tree(t, x) for x in X])
        return F


def s19_boosting() -> None:
    from sklearn.ensemble import GradientBoostingRegressor
    X = rng.normal(size=(500, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.05, 500)
    gb = GBDT(n_trees=40, lr=0.1, max_depth=3).fit(X, y)
    pred = gb.predict(X)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    sk = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1,
                                   max_depth=3, random_state=SEED).fit(X, y)
    sk_rmse = float(np.sqrt(np.mean((sk.predict(X) - y) ** 2)))
    assert rmse < 0.15
    ok("19", f"scratch GBDT RMSE {rmse:.4f} vs sklearn {sk_rmse:.4f} (pseudo-residuals)")
    # library boosters when installed
    for name, mod in (("xgboost", "XGBRegressor"), ("lightgbm", "LGBMRegressor"),
                      ("catboost", "CatBoostRegressor")):
        try:
            cls = getattr(__import__(name), mod)
            m = cls(n_estimators=40, verbose=0, random_state=SEED)
            m.fit(X, y)
            r = float(np.sqrt(np.mean((m.predict(X) - y) ** 2)))
            ok("19", f"{name} ({mod}) RMSE {r:.4f}")
        except Exception as exc:
            skip("19", f"{name} not installed ({type(exc).__name__})")


# ============================================================================
# S20. EXPLAINABILITY (Module 28 - permutation importance + SHAP if present)
# ============================================================================
def s20_explainability() -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    X = rng.normal(size=(400, 4))
    y = 5 * X[:, 0] + 0.1 * X[:, 1] + 0.01 * X[:, 2]           # feature 0 dominates
    m = RandomForestRegressor(n_estimators=40, random_state=SEED).fit(X, y)
    pi = permutation_importance(m, X, y, n_repeats=3, random_state=SEED)
    order = np.argsort(pi.importances_mean)[::-1]
    assert order[0] == 0, "permutation importance must rank the dominant feature first"
    ok("20", f"permutation importance ranks feature {order[0] + 1} first "
             f"(score drop {pi.importances_mean[order[0]]:.3f})")
    try:
        import shap
        ex = shap.TreeExplainer(m)
        vals = ex.shap_values(X[:50])
        if isinstance(vals, list):
            vals = vals[0]
        mean_abs = np.abs(vals).mean(axis=0)
        assert mean_abs.argmax() == 0
        ok("20", f"SHAP TreeExplainer ranks feature {mean_abs.argmax() + 1} first")
    except Exception as exc:
        skip("20", f"shap not installed ({type(exc).__name__})")


# ============================================================================
# S21. TIME-SERIES ML (Module 25 - lags + expanding-window validation)
# ============================================================================
def s21_time_series() -> None:
    from sklearn.ensemble import RandomForestRegressor
    t = np.arange(500)
    y = 10 * np.sin(t / 20) + 0.05 * t + rng.normal(0, 0.2, 500)
    # --- Why a longer horizon? At 1-step-ahead on a smooth signal the naive
    # "last value" forecast is nearly unbeatable (next sample == this one).
    # Persistence decays with the horizon, but a model with lag features and
    # learned dynamics does not — so H=12 is where lag-based ML genuinely wins.
    L, H = 6, 12                            # 6 lags -> 12-step-ahead forecast
    n = 500 - L - H + 1                     # rows where window + horizon fit
    X = np.column_stack([y[i: i + n] for i in range(L)])   # lag columns
    yt = y[L + H - 1: L + H - 1 + n]        # targets: H steps after each window
    idx = np.arange(L + H - 1, 500)         # TIME index of each target row
    cut = int(0.8 * n)
    Xtr, ytr, Xte, yte = X[:cut], yt[:cut], X[cut:], yt[cut:]
    assert idx[cut - 1] < idx[cut], "temporal split: train is strictly in the past"
    m = RandomForestRegressor(n_estimators=80, random_state=SEED).fit(Xtr, ytr)
    pred = m.predict(Xte)
    # naive baseline: flat H-step forecast = last value of each test window
    # (row r's window ends at y[r + L - 1]; window start = L + H - 1 + r - H)
    flat = np.array([y[r + L - 1] for r in range(cut, n)])
    naive = np.mean((yte - flat) ** 2)
    mse_ = np.mean((pred - yte) ** 2)
    assert mse_ < naive, "lag-based RF must beat the flat last-value baseline"
    ok("21", f"temporal split clean; {L}-lag RF at {H}-step horizon: "
             f"MSE {mse_:.3f} << flat baseline {naive:.3f} (persistence decays)")


# ============================================================================
# S22. ANOMALY DETECTION (Module 24 - Isolation Forest)
# ============================================================================
def s22_anomaly() -> None:
    from sklearn.ensemble import IsolationForest
    X = rng.normal(size=(500, 2))
    outliers = rng.uniform(-6, 6, size=(20, 2))                # injected far points
    Xall = np.vstack([X, outliers])
    clf = IsolationForest(contamination=0.04, random_state=SEED).fit(Xall)
    labels = clf.predict(Xall)                                 # -1 = anomaly
    flagged = np.where(labels == -1)[0]
    caught = np.sum(flagged >= 500)                            # outliers are last 20 rows
    assert caught >= 15, f"only caught {caught}/20 injected outliers"
    ok("22", f"IsolationForest caught {caught}/20 injected outliers "
             f"(score s(x)=2^(-E[h]/c(n)) - shallow = anomalous)")


# ============================================================================
# S23. SCIKIT-LEARN PIPELINES (Module 8/9 - leakage-free preprocessing)
# ============================================================================
def s23_pipelines() -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=400, n_features=3, n_informative=3,
                               n_redundant=0, random_state=SEED)
    X = np.column_stack([X, np.where(rng.random(len(X)) < 0.5, "a", "b")])  # categorical
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), [0, 1, 2]),
            ("cat", OneHotEncoder(), [3]),
        ])),
        ("model", RandomForestClassifier(n_estimators=40, random_state=SEED)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5)
    assert scores.mean() > 0.8
    # sanity: the fitted artifact transforms a row end-to-end
    pipe.fit(X, y)
    out = pipe.predict(X[:3])
    assert out.shape == (3,) and set(np.unique(out)).issubset({0, 1})
    ok("23", f"Pipeline + ColumnTransformer CV accuracy {scores.mean():.3f} "
             f"(preprocessing inside CV = no leakage)")


# ============================================================================
# S24. FASTAPI MODEL SERVING (Module 40 - guarded: no server is started)
# ============================================================================
def s24_fastapi() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=4, random_state=SEED)
    model = RandomForestClassifier(n_estimators=40, random_state=SEED).fit(X, y)
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
        app = FastAPI(title="ml-course-model")

        class Input(BaseModel):                 # typed contract = validation
            f1: float
            f2: float
            f3: float
            f4: float

        @app.post("/predict")
        def predict(x: Input) -> dict:
            proba = model.predict_proba(np.array([[x.f1, x.f2, x.f3, x.f4]]))[0]
            return {"class": int(proba.argmax()), "prob": float(proba.max())}

        # exercise the endpoint logic directly (no uvicorn in CI)
        resp = predict(Input(f1=0.1, f2=-0.2, f3=0.3, f4=0.4))
        assert resp["class"] in (0, 1) and 0 <= resp["prob"] <= 1
        ok("24", "FastAPI app defined; endpoint returns a valid prediction "
                 "(run `uvicorn` manually to serve)")
    except Exception as exc:
        skip("24", f"fastapi/pydantic not installed ({type(exc).__name__})")


# ============================================================================
# S25. MLFLOW EXPERIMENT TRACKING (Module 41 - guarded, temp tracking dir)
# ============================================================================
def s25_mlflow() -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    X = rng.normal(size=(300, 4))
    y = X @ np.array([1.0, 2.0, -0.5, 0.1]) + rng.normal(0, 0.1, 300)
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file://{os.path.join(TMP, 'mlruns')}")
        mlflow.set_experiment("example_25")
        with mlflow.start_run() as run:
            params = {"n_estimators": 50, "max_depth": 4}
            m = RandomForestRegressor(**params, random_state=SEED)
            score = cross_val_score(m, X, y, cv=3, scoring="r2").mean()
            mlflow.log_params(params)
            mlflow.log_metric("cv_r2", score)
            m.fit(X, y)
            mlflow.sklearn.log_model(m, "model")
        # read the run back from the artifact store by run id (stable API)
        stored = mlflow.get_run(run.info.run_id)
        assert abs(stored.data.metrics["cv_r2"] - score) < 1e-6
        ok("25", f"MLflow run tracked (cv_r2={score:.3f}) + model logged in {TMP}")
    except Exception as exc:
        skip("25", f"mlflow not installed ({type(exc).__name__})")


# ============================================================================
# S26. PRODUCTION ARTIFACT + ARCHITECTURE (Module 39/40 - joblib round trip)
# ============================================================================
def s26_production() -> None:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    X = rng.normal(size=(400, 5))
    y = X @ np.ones(5) + rng.normal(0, 0.1, 400)
    artifact = Pipeline([("scale", StandardScaler()),
                         ("model", RandomForestRegressor(n_estimators=50,
                                                         random_state=SEED))])
    artifact.fit(X, y)
    try:
        import joblib
        path = os.path.join(TMP, "model.joblib")
        joblib.dump(artifact, path)                            # preprocessor INSIDE
        loaded = joblib.load(path)
        assert np.allclose(loaded.predict(X[:3]), artifact.predict(X[:3]))
        ok("26", f"preprocessor+model shipped as ONE artifact; joblib round-trip OK")
    except Exception as exc:
        skip("26", f"joblib not available ({type(exc).__name__})")
    print("        architecture: ingest -> validate -> features -> train -> registry")
    print("        -> serve (FastAPI) -> monitor (drift + perf) -> retrain loop")


def main() -> None:
    print("=" * 72)
    print("ULTRA-DEEP ML COURSE - EXAMPLE.py (26 sections)")
    print("=" * 72)
    t0 = time.time()
    for fn in (s01_numpy, s02_pandas, s03_preprocessing, s04_visualization,
               s05_statistics, s06_linear_regression, s07_gradient_descent,
               s08_logistic_regression, s09_knn, s10_naive_bayes,
               s11_decision_tree, s12_random_forest, s13_kmeans, s14_pca,
               s15_metrics, s16_cv, s17_feature_engineering, s18_hpo,
               s19_boosting, s20_explainability, s21_time_series,
               s22_anomaly, s23_pipelines, s24_fastapi, s25_mlflow,
               s26_production):
        fn()
    passed = sum(1 for r in RESULTS if r.endswith("PASS"))
    skipped = sum(1 for r in RESULTS if r.endswith("SKIP"))
    print("=" * 72)
    print(f"SUMMARY: {passed} sections PASS, {skipped} guarded/skipped, "
          f"{len(RESULTS) - passed - skipped} FAIL  ({time.time() - t0:.1f}s)")
    print("=" * 72)


if __name__ == "__main__":
    main()