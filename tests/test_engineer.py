import numpy as np
import pandas as pd
import pytest

from src.features.engineer import engineer, split, TARGET, FEATURES


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        TARGET: np.random.binomial(1, 0.07, n),
        "revolving_utilization_of_unsecured_lines": np.random.uniform(0, 1, n),
        "age": np.random.randint(18, 80, n),
        "number_of_time30_59_days_past_due_not_worse": np.random.randint(0, 5, n),
        "debt_ratio": np.random.uniform(0, 2, n),
        "monthly_income": np.where(
            np.random.random(n) < 0.2, np.nan, np.random.uniform(1000, 20000, n)
        ),
        "number_of_open_credit_lines_and_loans": np.random.randint(0, 20, n),
        "number_of_times90_days_late": np.random.randint(0, 5, n),
        "number_real_estate_loans_or_lines": np.random.randint(0, 5, n),
        "number_of_time60_89_days_past_due_not_worse": np.random.randint(0, 5, n),
        "number_of_dependents": np.where(
            np.random.random(n) < 0.05, np.nan, np.random.randint(0, 5, n)
        ),
    })


def test_no_missing_after_engineer(sample_df):
    df = engineer(sample_df)
    assert df[FEATURES].isnull().sum().sum() == 0


def test_derived_features_exist(sample_df):
    df = engineer(sample_df)
    for col in ["debt_to_income", "total_past_due", "income_per_dependent"]:
        assert col in df.columns


def test_debt_to_income_formula(sample_df):
    df = engineer(sample_df)
    expected = df["debt_ratio"] * df["monthly_income"]
    pd.testing.assert_series_equal(df["debt_to_income"], expected, check_names=False)


def test_total_past_due_formula(sample_df):
    df = engineer(sample_df)
    expected = (
        df["number_of_time30_59_days_past_due_not_worse"]
        + df["number_of_time60_89_days_past_due_not_worse"]
        + df["number_of_times90_days_late"]
    )
    pd.testing.assert_series_equal(df["total_past_due"], expected, check_names=False)


def test_income_per_dependent_no_division_by_zero(sample_df):
    df = engineer(sample_df)
    assert df["income_per_dependent"].isnull().sum() == 0
    assert np.isfinite(df["income_per_dependent"]).all()


def test_split_sizes(sample_df):
    df = engineer(sample_df)
    X_train, X_test, y_train, y_test = split(df)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - 0.2) < 0.01


def test_split_stratification(sample_df):
    df = engineer(sample_df)
    _, _, y_train, y_test = split(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_features_list_matches_columns(sample_df):
    df = engineer(sample_df)
    for f in FEATURES:
        assert f in df.columns, f"Feature '{f}' missing after engineer()"
