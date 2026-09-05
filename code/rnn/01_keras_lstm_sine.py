# ============================================================
# LSTM SEQUENCE FORECASTING (synthetic sine wave)
# Shows the full pipeline for time-series with recurrent nets:
# windowing -> scale -> train -> evaluate -> forecast.
# Run: python 01_keras_lstm_sine.py
# Requires: tensorflow, numpy, matplotlib
# ============================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Generate data: sine with slight drift + noise
# ------------------------------------------------------------
n = 3000
t = np.linspace(0, 40 * np.pi, n)
series = np.sin(t) + 0.05 * t / 40 + np.random.default_rng(0).normal(0, 0.05, n)

# Scale to [0, 1] using ONLY training stats (no leakage)
train_end = int(0.8 * n)
train_part = series[:train_end]
min_v, max_v = train_part.min(), train_part.max()
series_s = (series - min_v) / (max_v - min_v)

# ------------------------------------------------------------
# 2. Windowing: input = last LOOKBACK points, target = next point
# ------------------------------------------------------------
LOOKBACK = 60


def make_windows(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


X, y = make_windows(series_s, LOOKBACK)
X = X[..., None]                      # (samples, lookback, 1) for LSTM
Xtr, ytr = X[:train_end - LOOKBACK], y[:train_end - LOOKBACK]
Xte, yte = X[train_end - LOOKBACK:], y[train_end - LOOKBACK:]
print(f"train windows={Xtr.shape} test windows={Xte.shape}")

# ------------------------------------------------------------
# 3. Model — many-to-one LSTM
# ------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input((LOOKBACK, 1)),
    keras.layers.LSTM(32, return_sequences=True),
    keras.layers.LSTM(16),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
model.summary()

history = model.fit(Xtr, ytr, batch_size=128, epochs=30,
                    validation_split=0.1, verbose=1)
print(f"Final train loss: {history.history['loss'][-1]:.6f}")

# ------------------------------------------------------------
# 4. Evaluate + plot forecast vs truth
# ------------------------------------------------------------
pred = model.predict(Xte, verbose=0).flatten()
# back to original scale for plotting
pred_orig = pred * (max_v - min_v) + min_v
true_orig = yte * (max_v - min_v) + min_v
mse_orig = float(np.mean((pred_orig - true_orig) ** 2))
print(f"Test MSE (original scale): {mse_orig:.5f}")

plt.figure(figsize=(10, 3.5))
plt.plot(true_orig[:400], label="truth")
plt.plot(pred_orig[:400], label="LSTM forecast")
plt.xlabel("time step"); plt.ylabel("value"); plt.title("Sine forecast")
plt.legend(); plt.tight_layout()
plt.savefig("lstm_sine_forecast.png"); plt.close()
print("Saved: lstm_sine_forecast.png")

# ------------------------------------------------------------
# Golden rules for RNN/LSTM on real data:
#  1. Fit scaler on TRAIN only; apply same transform to test.
#  2. Respect temporal order — never shuffle test windows.
#  3. Think in "many-to-many vs many-to-one" input shapes.
#  4. Baselines first: naive persistence often beats a weak LSTM.
# ============================================================
