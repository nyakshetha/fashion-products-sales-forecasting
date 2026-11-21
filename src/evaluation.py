"""Metrics and leakage-aware splitting for the purchase-likelihood models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

import config as C


def grouped_split(df: pd.DataFrame, group_col: str, test_size: float = 0.2, seed: int = C.SEED):
    """Hold out whole customers (or whole garments) so no entity appears in both train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df[group_col]))
    return tr, te


def classification_metrics(y_true, y_prob, threshold: float = C.CLASS_THRESHOLD) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    base = float(y_true.mean())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "brier_baseline": base * (1 - base),       # Brier of always predicting the base rate
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6)),
        "majority_class_accuracy": max(base, 1 - base),
        "test_rows": int(len(y_true)),
        "test_positive_rate": base,
    }


def confusion(y_true, y_prob, threshold: float = C.CLASS_THRESHOLD) -> np.ndarray:
    return confusion_matrix(y_true, (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])


def report_text(y_true, y_prob, threshold: float = C.CLASS_THRESHOLD) -> str:
    return classification_report(y_true, (np.asarray(y_prob) >= threshold).astype(int),
                                 target_names=["Not purchased", "Purchased"], zero_division=0)


def calibration_table(y_true, y_prob, n_bins: int = 8) -> pd.DataFrame:
    frac, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac})
