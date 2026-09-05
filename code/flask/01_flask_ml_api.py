# ============================================================
# FLASK + ML — minimal inference service
# Same contract as code/fastapi/01_fastapi_ml_api.py so you can
# compare the two frameworks directly.
#
# Run:
#   pip install flask scikit-learn
#   python 01_flask_ml_api.py
# Then POST to http://127.0.0.1:5000/predict
# ============================================================
import json

from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Train once at import (dev pattern)
# ------------------------------------------------------------
iris = load_iris()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(iris.data, iris.target)
CLASS_NAMES = iris.target_names.tolist()

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ------------------------------------------------------------
# 2. Validation is manual in Flask (FastAPI/Pydantic do it for
#    you). Validate types + length before calling the model.
# ------------------------------------------------------------
@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    features = body.get("features")

    if not isinstance(features, list) or len(features) != 4:
        return jsonify({"error": "features must be a list of 4 numbers"}), 400
    if not all(isinstance(x, (int, float)) for x in features):
        return jsonify({"error": "all features must be numeric"}), 400

    proba = model.predict_proba([features])[0]
    pred_idx = int(proba.argmax())
    return jsonify({
        "predicted_class": CLASS_NAMES[pred_idx],
        "probabilities": {name: round(float(p), 4)
                          for name, p in zip(CLASS_NAMES, proba)},
    })


# ------------------------------------------------------------
# 3. Test
#   curl -X POST http://127.0.0.1:5000/predict \
#        -H "Content-Type: application/json" \
#        -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

# ------------------------------------------------------------
# FastAPI vs Flask for AI services (module 36_FLASK):
#   FastAPI: pydantic validation, auto OpenAPI docs, async,
#            typed — the default choice for new ML APIs.
#   Flask:   simpler, huge ecosystem, synchronous — good for
#            small services or when team already uses it.
# ============================================================
