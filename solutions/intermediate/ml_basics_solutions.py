# ============================================================
# SOLUTIONS — exercises/intermediate/ml_basics.py
# Run: python ml_basics_solutions.py
# Requires: numpy, scikit-learn
# ============================================================
import numpy as np
from sklearn.datasets import make_classification, make_regression

# ------------------------------------------------------------
# EXERCISE 1: MSE Loss
# ------------------------------------------------------------
def mse_loss(y_true, y_pred):
    """Mean Squared Error = mean((y_true - y_pred)^2)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


# ------------------------------------------------------------
# EXERCISE 2: Binary Cross-Entropy Loss
# ------------------------------------------------------------
def cross_entropy_loss(y_true, y_pred, eps=1e-12):
    """BCE = -mean(y*log(p) + (1-y)*log(1-p)).
    eps avoids log(0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


# ------------------------------------------------------------
# EXERCISE 3: Accuracy
# ------------------------------------------------------------
def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


# ------------------------------------------------------------
# EXERCISE 4: Confusion Matrix (binary, sklearn ordering:
# rows = actual [neg, pos], cols = predicted [neg, pos])
# ------------------------------------------------------------
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


# ------------------------------------------------------------
# EXERCISE 5: Precision, Recall, F1
# ------------------------------------------------------------
def precision(y_true, y_pred):
    """TP / (TP + FP) — of predicted positives, how many are right."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp = cm[1, 1], cm[0, 1]
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """TP / (TP + FN) — of actual positives, how many found."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fn = cm[1, 1], cm[1, 0]
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ------------------------------------------------------------
# EXERCISE 6: StandardScaler  z = (x - mean) / std
# ------------------------------------------------------------
class StandardScaler:
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid div-by-zero on constant columns
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ------------------------------------------------------------
# EXERCISE 7: Train-Test Split
# ------------------------------------------------------------
def train_test_split(X, y, test_size=0.2, random_state=None):
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = len(X)
    n_test = int(round(n * test_size))
    idx = rng.permutation(n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ------------------------------------------------------------
# EXERCISE 8: K-Fold Cross Validation
# ------------------------------------------------------------
def k_fold_cv(X, y, model_class, k=5, random_state=42):
    X, y = np.asarray(X), np.asarray(y)
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    fold_size = len(X) // k
    scores = []
    for i in range(k):
        val_idx = np.arange(i * fold_size, (i + 1) * fold_size)
        # Handle remainder rows by appending to the last fold
        if i == k - 1:
            val_idx = np.arange(i * fold_size, len(X))
        mask = np.ones(len(X), dtype=bool)
        mask[val_idx] = False
        model = model_class()
        model.fit(X[mask], y[mask])
        scores.append(model.score(X[val_idx], y[val_idx]))
    return np.array(scores)


# ------------------------------------------------------------
# EXERCISE 9: Variance Threshold (feature selection)
# ------------------------------------------------------------
class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.variances_ = X.var(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        keep = self.variances_ > self.threshold
        return X[:, keep]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ------------------------------------------------------------
# EXERCISE 10: Complete ML Pipeline
# ------------------------------------------------------------
def ml_pipeline(X, y):
    """Standardize -> split -> train -> evaluate."""
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy(y_test, y_pred),
        "precision": precision(y_test, y_pred),
        "recall": recall(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }


# ============================================================
# TESTS — expected values match the exercise comments
# ============================================================
if __name__ == "__main__":
    # Ex 1
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
    mse = mse_loss(y_true, y_pred)
    print(f"MSE: {mse:.3f}")                      # 0.038
    assert abs(mse - 0.038) < 1e-9

    # Ex 2
    yt = np.array([1, 0, 1, 1, 0])
    yp = np.array([0.9, 0.1, 0.8, 0.7, 0.3])
    bce = cross_entropy_loss(yt, yp)
    print(f"BCE: {bce:.4f}")                      # 0.2294
    assert abs(bce - 0.2294) < 1e-3

    # Ex 3
    acc = accuracy(np.array([1, 0, 1, 1, 0]), np.array([1, 0, 0, 1, 1]))
    print(f"Accuracy: {acc}")                     # 0.6
    assert acc == 0.6

    # Ex 4
    y_true4 = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred4 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
    cm = confusion_matrix(y_true4, y_pred4)
    print(f"Confusion Matrix:\n{cm}")             # [[4 1] [1 4]]
    assert cm.tolist() == [[4, 1], [1, 4]]

    # Ex 5
    p, r, f1 = precision(y_true4, y_pred4), recall(y_true4, y_pred4), f1_score(y_true4, y_pred4)
    print(f"Precision: {p:.2f}")                  # 0.80
    print(f"Recall: {r:.2f}")                     # 0.80
    print(f"F1: {f1:.2f}")                        # 0.80
    assert abs(p - 0.8) < 1e-9 and abs(r - 0.8) < 1e-9

    # Ex 6
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Original mean: {X.mean(axis=0)}, Std: {X.std(axis=0)}")
    print(f"Scaled mean: {X_scaled.mean(axis=0)}, Std: {X_scaled.std(axis=0)}")
    assert np.allclose(X_scaled.mean(axis=0), 0)
    assert np.allclose(X_scaled.std(axis=0), 1)

    # Ex 7
    Xd, yd = make_classification(n_samples=100, n_features=5, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.2, random_state=0)
    print(f"Train size: {len(Xtr)}, Test size: {len(Xte)}")
    assert len(Xtr) == 80 and len(Xte) == 20

    # Ex 8
    from sklearn.linear_model import LogisticRegression
    scores = k_fold_cv(Xd, yd, LogisticRegression, k=5)
    print(f"CV Scores: {np.round(scores, 4)}")
    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    assert len(scores) == 5

    # Ex 9
    Xv = np.array([[0, 2, 0.5], [1, 2, 0.5], [2, 2, 0.5], [3, 2, 0.5]])
    selector = VarianceThreshold(threshold=0.1)
    X_selected = selector.fit_transform(Xv)
    print(f"Original shape: {Xv.shape}, Selected shape: {X_selected.shape}")
    assert X_selected.shape == (4, 1)

    # Ex 10
    Xm, ym = make_classification(n_samples=200, n_features=10, random_state=42)
    results = ml_pipeline(Xm, ym)
    print(f"Results: {results}")

    print("\nALL ML SOLUTION TESTS PASSED [OK]")
