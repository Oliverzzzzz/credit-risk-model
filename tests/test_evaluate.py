import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.models.evaluate import gini_coefficient, ks_statistic, report


def test_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    assert ks_statistic(y_true, y_prob) == pytest.approx(1.0)
    assert gini_coefficient(y_true, y_prob) == pytest.approx(1.0)


def test_random_classifier_gini_near_zero():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 10_000)
    y_prob = np.random.uniform(0, 1, 10_000)
    assert abs(gini_coefficient(y_true, y_prob)) < 0.05


def test_gini_equals_2_auc_minus_1():
    np.random.seed(0)
    y_true = np.random.randint(0, 2, 500)
    y_prob = np.random.uniform(0, 1, 500)
    auc = roc_auc_score(y_true, y_prob)
    assert gini_coefficient(y_true, y_prob) == pytest.approx(2 * auc - 1, abs=1e-6)


def test_ks_range():
    np.random.seed(1)
    y_true = np.random.randint(0, 2, 500)
    y_prob = np.random.uniform(0, 1, 500)
    ks = ks_statistic(y_true, y_prob)
    assert 0.0 <= ks <= 1.0


def test_report_returns_all_keys():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.7])
    result = report(y_true, y_prob)
    assert set(result.keys()) == {"roc_auc", "ks", "gini"}


def test_report_values_in_valid_range():
    np.random.seed(2)
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.random.uniform(0, 1, 200)
    result = report(y_true, y_prob)
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["ks"] <= 1.0
    assert -1.0 <= result["gini"] <= 1.0
