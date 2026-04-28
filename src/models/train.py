import mlflow
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.evaluate import report


def train_logistic(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    # scale_pos_weight handles class imbalance — ratio of negatives to positives
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def run_experiment(X_train, X_test, y_train, y_test, model_name: str = "xgboost"):
    with mlflow.start_run(run_name=model_name):
        if model_name == "logistic":
            model = train_logistic(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model = train_xgboost(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

        metrics = report(y_test, y_prob)
        mlflow.log_params({"model": model_name})
        mlflow.log_metrics(metrics)
        print(f"{model_name}: {metrics}")

    return model, y_prob
