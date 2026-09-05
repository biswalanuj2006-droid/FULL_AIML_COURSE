# ============================================================
# KERAS / TENSORFLOW — MNIST MLP (template for any tabular MLP)
# Run: python 01_keras_mlp_mnist.py
# Requires: tensorflow (CPU build is fine for MNIST)
# ============================================================
import os
import tensorflow as tf
from tensorflow import keras

# CPU users: override the epoch count for a quick smoke test,
# e.g.  EPOCHS=3 python 01_keras_mlp_mnist.py
EPOCHS = int(os.environ.get("EPOCHS", "30"))

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and flatten 28x28 -> 784
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Hold out a validation split for early stopping
x_val, y_val = x_train[-5000:], y_train[-5000:]
x_train, y_train = x_train[:-5000], y_train[:-5000]
print(f"train={x_train.shape} val={x_val.shape} test={x_test.shape}")

# ------------------------------------------------------------
# 2. Model
# ------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax"),   # 10 classes
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",         # integer labels
    metrics=["accuracy"],
)
model.summary()

# ------------------------------------------------------------
# 3. Training with callbacks
# ------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    keras.callbacks.ModelCheckpoint("mnist_mlp.keras", save_best_only=True),
]

history = model.fit(
    x_train, y_train,
    batch_size=128, epochs=EPOCHS,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
)

# ------------------------------------------------------------
# 4. Evaluation + prediction
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

preds = model.predict(x_test[:5])
print("Predictions:", preds.argmax(axis=1), "actual:", y_test[:5])

# ------------------------------------------------------------
# Golden rules:
#  1. Normalize inputs; keep validation data separate from training.
#  2. EarlyStopping + restore_best_weights prevents overfitting.
#  3. sparse_categorical_crossentropy with integer labels.
#  4. Save best model with ModelCheckpoint; reload via
#     keras.models.load_model("mnist_mlp.keras").
# ============================================================
