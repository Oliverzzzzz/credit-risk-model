import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def ks_statistic(y_true, y_prob) -> float:
    """KS statistic: max separation between cumulative good/bad distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def gini_coefficient(y_true, y_prob) -> float:
    """Gini = 2 * AUC - 1. Industry standard for scorecard performance."""
    return 2 * roc_auc_score(y_true, y_prob) - 1


def report(y_true, y_prob) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "ks": round(ks_statistic(y_true, y_prob), 4),
        "gini": round(gini_coefficient(y_true, y_prob), 4),
    }


def find_threshold(y_true, y_prob, target_precision: float = 0.5) -> float:
    """Return the score threshold that achieves at least target_precision."""
    precisions, _, thresholds = precision_recall_curve(y_true, y_prob)
    for precision, threshold in zip(precisions, thresholds):
        if precision >= target_precision:
            return float(threshold)
    return float(thresholds[-1])
