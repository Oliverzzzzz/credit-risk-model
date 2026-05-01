"""
End-to-end training pipeline. Run from project root:
    python pipeline.py
"""
import pickle
from pathlib import Path

import mlflow

from src.data.loader import load_raw, save_processed
from src.features.engineer import engineer, split
from src.models.train import train_xgboost
from src.models.evaluate import report

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def run():
    print("=== Credit Risk Model — Training Pipeline ===\n")

    print("[1/4] Loading data...")
    df_raw = load_raw()
    print(f"      {df_raw.shape[0]:,} rows loaded")

    print("[2/4] Engineering features...")
    df = engineer(df_raw)
    save_processed(df)
    X_train, X_test, y_train, y_test = split(df)
    print(f"      Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    print("[3/4] Training XGBoost with MLflow tracking...")
    mlflow.set_experiment("credit-risk-model")
    with mlflow.start_run(run_name="xgboost_pipeline"):
        model = train_xgboost(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = report(y_test, y_prob)
        mlflow.log_params({"model": "xgboost", "n_estimators": 300,
                           "learning_rate": 0.05, "max_depth": 6})
        mlflow.log_metrics(metrics)

    print("[4/4] Saving model...")
    model_path = MODELS_DIR / "xgboost.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n=== Done ===")
    print(f"  Model : {model_path}")
    print(f"  ROC-AUC : {metrics['roc_auc']}")
    print(f"  KS      : {metrics['ks']}")
    print(f"  Gini    : {metrics['gini']}")
    print(f"\n  MLflow UI: mlflow ui  →  http://localhost:5000")


if __name__ == "__main__":
    run()
