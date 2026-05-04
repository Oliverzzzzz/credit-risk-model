# Credit Risk Model

An end-to-end credit default prediction system built to production standards. Predicts the probability that a borrower will experience serious financial distress within 2 years, using the same metrics and methodology used by ML teams at banks.

[![CI](https://github.com/Oliverzzzzz/credit-risk-model/actions/workflows/ci.yml/badge.svg)](https://github.com/Oliverzzzzz/credit-risk-model/actions/workflows/ci.yml)

---

## Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| ROC-AUC | 0.8026 | **0.8641** |
| KS Statistic | 0.4552 | **0.5776** |
| Gini Coefficient | 0.6052 | **0.7282** |

At decision threshold 0.15: **95.1% recall** on defaulters (catches 95 out of every 100 real defaults).

---

## Project Structure

```
credit-risk-model/
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # Imputation and derived features
│   ├── 03_modeling.ipynb               # Logistic regression + XGBoost
│   ├── 04_evaluation.ipynb             # KS, Gini, ROC, confusion matrix
│   ├── 05_explainability.ipynb         # SHAP global and local explanations
│   └── 06_hyperparameter_tuning.ipynb  # Optuna Bayesian search
├── src/
│   ├── data/loader.py                  # Data loading and column normalisation
│   ├── features/engineer.py            # Feature engineering and train/test split
│   ├── models/
│   │   ├── train.py                    # Model training with MLflow tracking
│   │   └── evaluate.py                 # KS statistic, Gini, ROC-AUC
│   └── api/main.py                     # FastAPI scoring endpoint
├── tests/
│   ├── test_engineer.py                # Unit tests for feature pipeline
│   └── test_evaluate.py                # Unit tests for evaluation metrics
├── pipeline.py                         # End-to-end training script
└── Makefile                            # Common commands
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Oliverzzzzz/credit-risk-model.git
cd credit-risk-model
```

**2. Install dependencies**
```bash
make install
```

**3. Download the dataset**

Go to [kaggle.com/c/GiveMeSomeCredit/data](https://www.kaggle.com/c/GiveMeSomeCredit/data), download `cs-training.csv`, and place it at:
```
data/raw/cs-training.csv
```

---

## Usage

**Train the model**
```bash
make train
```

**Run the notebooks** (in order, 01 → 06)
```bash
make notebook
```

**Start the scoring API**
```bash
make api
# POST http://localhost:8000/score
```

**View MLflow experiment runs**
```bash
make mlflow
# http://localhost:5000
```

**Run tests**
```bash
pytest tests/ -v
```

---

## API

`POST /score` — returns probability of default and risk tier.

**Request**
```json
{
  "revolving_utilization": 0.75,
  "age": 45,
  "times_30_59_days_late": 0,
  "debt_ratio": 0.38,
  "monthly_income": 6500,
  "open_credit_lines": 8,
  "times_90_days_late": 0,
  "real_estate_loans": 1,
  "times_60_89_days_late": 0,
  "dependents": 2
}
```

**Response**
```json
{
  "probability_of_default": 0.0312,
  "risk_tier": "low"
}
```

Risk tiers: `low` (< 5%), `medium` (5–15%), `high` (> 15%)

---

## Key Concepts

**Why KS and Gini instead of accuracy?**
With a 6.68% default rate, predicting "no default" always achieves 93.3% accuracy but catches zero defaulters. KS and Gini directly measure the model's ability to separate defaulters from non-defaulters regardless of threshold.

**Why threshold 0.15?**
Missing a defaulter costs far more than rejecting a creditworthy applicant. Lowering the threshold biases the model toward recall — at 0.15 the model catches 95.1% of real defaulters at the cost of 8.3 false alarms per true catch.

**Why SHAP?**
Basel regulations require banks to explain credit decisions to customers. SHAP computes each feature's exact marginal contribution to each prediction, enabling both global model interpretation and per-borrower explanations.

---

## Tech Stack

`XGBoost` `scikit-learn` `SHAP` `MLflow` `Optuna` `FastAPI` `pandas` `pytest` `GitHub Actions`
