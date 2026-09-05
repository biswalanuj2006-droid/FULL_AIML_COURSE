# ============================================================
# MATPLOTLIB BASICS FOR ML
# The four charts you draw in every ML project:
# loss curve, scatter + regression, confusion-matrix heatmap, ROC.
# Run: python 01_matplotlib_basics.py
# Requires: numpy, matplotlib, scikit-learn
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 0. Consistent style
# ------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

# ------------------------------------------------------------
# 1. Loss curve
# ------------------------------------------------------------
epochs = np.arange(1, 51)
train_loss = 0.9 * np.exp(-epochs / 12) + 0.02
val_loss = 1.1 * np.exp(-epochs / 14) + 0.05

plt.figure(figsize=(5, 3.5))
plt.plot(epochs, train_loss, label="train")
plt.plot(epochs, val_loss, label="validation")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Training vs validation loss")
plt.legend(); plt.tight_layout(); plt.savefig("vis_01_loss_curve.png"); plt.close()

# ------------------------------------------------------------
# 2. Regression line
# ------------------------------------------------------------
rng = np.random.default_rng(0)
x = rng.uniform(-3, 3, 120)
y = 2.0 * x - 1.0 + rng.normal(0, 0.8, 120)
coef = np.polyfit(x, y, 1)
xs = np.linspace(-3, 3, 100)

plt.figure(figsize=(5, 3.5))
plt.scatter(x, y, s=12, alpha=0.6, label="data")
plt.plot(xs, np.polyval(coef, xs), color="crimson", label=f"fit: y={coef[0]:.2f}x{coef[1]:+.2f}")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Linear regression fit")
plt.legend(); plt.tight_layout(); plt.savefig("vis_02_regression.png"); plt.close()

# ------------------------------------------------------------
# 3. Confusion matrix heatmap
# ------------------------------------------------------------
X, y = make_classification(n_samples=500, n_features=10, weights=[0.7, 0.3],
                           random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
cm = confusion_matrix(yte, model.predict(Xte))

plt.figure(figsize=(4, 3.5))
plt.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], ["neg", "pos"])
plt.yticks([0, 1], ["neg", "pos"])
plt.xlabel("predicted"); plt.ylabel("actual")
plt.title("Confusion matrix"); plt.colorbar()
plt.tight_layout(); plt.savefig("vis_03_confusion.png"); plt.close()

# ------------------------------------------------------------
# 4. ROC curve
# ------------------------------------------------------------
proba = model.predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure(figsize=(5, 3.5))
plt.plot(fpr, tpr, label=f"LogReg (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve"); plt.legend()
plt.tight_layout(); plt.savefig("vis_04_roc.png"); plt.close()

print("Saved: vis_01_loss_curve.png, vis_02_regression.png, "
      "vis_03_confusion.png, vis_04_roc.png")

# ------------------------------------------------------------
# Golden rules:
#  1. Always label axes + title; add legend when >1 series.
#  2. Use figsize + dpi so figures are readable when embedded.
#  3. Tight_layout before save; close() in loops to save memory.
#  4. Move generated PNGs into diagrams/graphs/ when finished.
# ============================================================
