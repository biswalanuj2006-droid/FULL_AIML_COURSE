# ============================================================
# FASTAPI + ML — minimal inference service
# Trains a tiny model at startup (dev pattern), serves /predict.
#
# Run:
#   pip install fastapi uvicorn scikit-learn
#   uvicorn 01_fastapi_ml_api:app --reload
# Then open http://127.0.0.1:8000/docs (Swagger UI)
# ============================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Load + train once at import time (dev pattern; production
#    loads a serialized model instead — see mlops/ model registry)
# ------------------------------------------------------------
iris = load_iris()
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(iris.data, iris.target)
CLASS_NAMES = iris.target_names.tolist()


# ------------------------------------------------------------
# 2. Request/response contracts (Pydantic = validation + docs)
# ------------------------------------------------------------
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4,
        description="Iris sepal/petal: [sepal_l, sepal_w, petal_l, petal_w]",
    )


class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]


app = FastAPI(title="Iris Classifier", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    proba = model.predict_proba([req.features])[0]
    pred_idx = int(proba.argmax())
    return PredictResponse(
        predicted_class=CLASS_NAMES[pred_idx],
        probabilities={name: round(float(p), 4)
                       for name, p in zip(CLASS_NAMES, proba)},
    )


# ------------------------------------------------------------
# Test from another shell:
#   curl -X POST http://127.0.0.1:8000/predict \
#        -H "Content-Type: application/json" \
#        -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
#
# Example request -> {"predicted_class": "setosa", ...}
#
# Production upgrades (see modules 35_FASTAPI, 33_MLOPS):
#   - load model from registry / artifact store, not train at boot
#   - add request logging + error handling
#   - add API key / JWT auth
#   - add a /models endpoint for metadata + version
#   - run behind uvicorn workers, containerize with Docker
# ============================================================
