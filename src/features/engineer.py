import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "serious_dlqin2yrs"

FEATURES = [
    "revolving_utilization_of_unsecured_lines",
    "age",
    "number_of_time30_59_days_past_due_not_worse",
    "debt_ratio",
    "monthly_income",
    "number_of_open_credit_lines_and_loans",
    "number_of_times90_days_late",
    "number_real_estate_loans_or_lines",
    "number_of_time60_89_days_past_due_not_worse",
    "number_of_dependents",
    # engineered
    "debt_to_income",
    "total_past_due",
    "income_per_dependent",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
    df["number_of_dependents"] = df["number_of_dependents"].fillna(0)

    df["debt_to_income"] = df["debt_ratio"] * df["monthly_income"]
    df["total_past_due"] = (
        df["number_of_time30_59_days_past_due_not_worse"]
        + df["number_of_time60_89_days_past_due_not_worse"]
        + df["number_of_times90_days_late"]
    )
    # +1 avoids division by zero for customers with no dependents
    df["income_per_dependent"] = df["monthly_income"] / (df["number_of_dependents"] + 1)

    return df


def split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
