# ============================================================
# MATHEMATICS FOR ML — CANONICAL VISUALIZATIONS
# Renders the five plots every ML student must be able to draw
# by hand conceptually, then check with matplotlib.
#
# Run: python math_visualizations.py
# Requires: numpy, matplotlib
# Output: 5 PNGs in the current directory; move them into
#         05_MATHEMATICS/visualizations/ (or diagrams/) after.
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})

rng = np.random.default_rng(0)

# ------------------------------------------------------------
# 1. Sigmoid + derivative
# ------------------------------------------------------------
z = np.linspace(-8, 8, 400)
sig = 1 / (1 + np.exp(-z))

plt.figure(figsize=(5, 3.5))
plt.plot(z, sig, label="sigma(z) = 1/(1+e^-z)")
plt.plot(z, sig * (1 - sig), label="sigma'(z)", ls="--")
plt.axhline(0.5, color="gray", lw=0.6, ls=":")
plt.xlabel("z"); plt.ylabel("value")
plt.title("Sigmoid activation and its derivative")
plt.legend(); plt.tight_layout(); plt.savefig("math_1_sigmoid.png"); plt.close()

# ------------------------------------------------------------
# 2. MSE loss surface + gradient steps toward the minimum
# ------------------------------------------------------------
w0 = np.linspace(-2, 4, 300)
w1 = np.linspace(-2, 4, 300)
W0, W1 = np.meshgrid(w0, w1)
# synthetic convex loss: L = 1.2*(W0-1)^2 + 2.0*(W1-0.5)^2 + interaction-free
L = 1.2 * (W0 - 1) ** 2 + 2.0 * (W1 - 0.5) ** 2

plt.figure(figsize=(5, 3.5))
plt.contourf(W0, W1, L, levels=25, cmap="viridis")
# gradient descent path (closed-form update for this quadratic)
path = [(2.8, 2.8)]
for _ in range(30):
    x, y = path[-1]
    gx, gy = 2.4 * (x - 1), 4.0 * (y - 0.5)
    step = 0.08
    path.append((x - step * gx, y - step * gy))
px, py = zip(*path)
plt.plot(px, py, "ro-", ms=3, lw=1, label="gradient descent")
plt.plot(1, 0.5, "w*", ms=14, label="minimum")
plt.xlabel("w0"); plt.ylabel("w1")
plt.title("Gradient descent on a convex loss surface")
plt.legend(); plt.tight_layout(); plt.savefig("math_2_gd_contour.png"); plt.close()

# ------------------------------------------------------------
# 3. Bias-variance decomposition of error
# ------------------------------------------------------------
x = np.linspace(1, 6, 300)
plt.figure(figsize=(5, 3.5))
plt.plot(x, 0.25 * (x - 3.5) ** 2, label="bias^2", lw=2)
plt.plot(x, 0.15 * np.ones_like(x), label="variance", lw=2)
plt.plot(x, 0.25 * (x - 3.5) ** 2 + 0.15, label="total error", lw=2.5, ls="--")
plt.axvline(3.5, color="gray", ls=":", label="sweet spot")
plt.xlabel("model complexity"); plt.ylabel("error")
plt.title("Bias-variance trade-off")
plt.legend(); plt.tight_layout(); plt.savefig("math_3_bias_variance.png"); plt.close()

# ------------------------------------------------------------
# 4. Gaussian: PDF, mean +/- 1/2/3 sigma
# ------------------------------------------------------------
mu, sd = 0, 1
x = np.linspace(-4.5, 4.5, 500)
pdf = np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

plt.figure(figsize=(5, 3.5))
plt.plot(x, pdf, lw=2, label=f"N({mu}, {sd}^2)")
for k, c in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")]:
    lo, hi = mu - k * sd, mu + k * sd
    plt.fill_between(x, pdf, where=(x >= lo) & (x <= hi), alpha=0.12, color=c,
                     label=f"+/-{k} sigma ~{k*68.27 if k==1 else (k*95.45 if k==2 else 99.73):.0f}%")
plt.xlabel("x"); plt.ylabel("density")
plt.title("Normal distribution: 68-95-99.7 rule")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("math_4_gaussian.png"); plt.close()

# ------------------------------------------------------------
# 5. Entropy vs Gini for a two-class split (tree impurity)
# ------------------------------------------------------------
p = np.linspace(0.001, 0.999, 400)
def hb(p_):
    return -p_ * np.log2(p_) - (1 - p_) * np.log2(1 - p_)

gini = 2 * p * (1 - p)

plt.figure(figsize=(5, 3.5))
plt.plot(p, hb(p), label="entropy (bits)")
plt.plot(p, gini, label="Gini impurity")
plt.plot(p, np.minimum(p, 1 - p), label="classification error", ls="--")
plt.xlabel("P(class A)"); plt.ylabel("impurity")
plt.title("Impurity measures used by decision trees")
plt.legend(); plt.tight_layout(); plt.savefig("math_5_impurity.png"); plt.close()

print("Saved: math_1_sigmoid.png, math_2_gd_contour.png, math_3_bias_variance.png,")
print("       math_4_gaussian.png, math_5_impurity.png")
print("Tip: run from 05_MATHEMATICS/visualizations/ so PNGs land next to this script.")
