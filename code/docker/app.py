# ============================================================
# app.py — the service containerized by Dockerfile + compose
# (condensed copy of code/fastapi/01_fastapi_ml_api.py so the
#  docker build context stays self-contained)
# ============================================================
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(iris.data, iris.target)
CLASS_NAMES = iris.target_names.tolist()

app = FastAPI(title="iris-classifier", version="1.0.0")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 4:
        return {"error": "exactly 4 features required"}
    proba = model.predict_proba([req.features])[0]
    return {
        "predicted_class": CLASS_NAMES[int(proba.argmax())],
        "probabilities": {name: round(float(p), 4)
                          for name, p in zip(CLASS_NAMES, proba)},
    }
