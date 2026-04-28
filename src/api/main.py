import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Credit Risk Scorer")

MODEL_PATH = Path(__file__).parents[2] / "models" / "xgboost.pkl"
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)


class CreditFeatures(BaseModel):
    revolving_utilization: float
    age: int
    times_30_59_days_late: int
    debt_ratio: float
    monthly_income: float
    open_credit_lines: int
    times_90_days_late: int
    real_estate_loans: int
    times_60_89_days_late: int
    dependents: int


class RiskScore(BaseModel):
    probability_of_default: float
    risk_tier: str


@app.post("/score", response_model=RiskScore)
def score(features: CreditFeatures):
    debt_to_income = features.debt_ratio * features.monthly_income
    total_past_due = features.times_30_59_days_late + features.times_60_89_days_late + features.times_90_days_late
    income_per_dependent = features.monthly_income / (features.dependents + 1)

    X = np.array([[
        features.revolving_utilization,
        features.age,
        features.times_30_59_days_late,
        features.debt_ratio,
        features.monthly_income,
        features.open_credit_lines,
        features.times_90_days_late,
        features.real_estate_loans,
        features.times_60_89_days_late,
        features.dependents,
        debt_to_income,
        total_past_due,
        income_per_dependent,
    ]])

    prob = float(model.predict_proba(X)[0, 1])
    tier = "high" if prob > 0.15 else "medium" if prob > 0.05 else "low"

    return RiskScore(probability_of_default=round(prob, 4), risk_tier=tier)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
