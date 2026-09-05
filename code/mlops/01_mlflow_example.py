# ============================================================
# MLFLOW — experiment tracking in ~20 lines
# Logs params, metrics, and the model artifact. Results appear
# in ./mlruns; view them with:  mlflow ui
#
# Run: python 01_mlflow_example.py
# Requires: pip install mlflow scikit-learn
# ============================================================
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

mlflow.set_experiment("diabetes-demo")

# ------------------------------------------------------------
# One run per experiment: log everything needed to reproduce
# ------------------------------------------------------------
for model_name, model in [
    ("ridge", Ridge(alpha=1.0)),
    ("random_forest", RandomForestRegressor(n_estimators=100, random_state=0)),
]:
    with mlflow.start_run(run_name=model_name):
        # params -> hyperparameters
        params = model.get_params()
        mlflow.log_params({f"param.{k}": v for k, v in params.items()})

        # train + metric
        model.fit(Xtr, ytr)
        mse = mean_squared_error(yte, model.predict(Xte))
        mlflow.log_metric("test_mse", mse)

        # artifact -> the model itself (load later with mlflow.pyfunc.load_model)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"{model_name}: test_mse={mse:.1f}  run_id={mlflow.active_run().info.run_id}")

# ------------------------------------------------------------
# Viewing results:
#   mlflow ui --port 5000   -> http://127.0.0.1:5000
# Comparing the two runs answers "which model do I ship?" —
# Ridge and Random Forest differ, and you keep the evidence.
#
# Registry + serving (module 33_MLOPS):
#   from mlflow.tracking import MlflowClient
#   client = MlflowClient()
#   client.register_model(f"runs:/{run_id}/model", "diabetes-model")
# ============================================================
