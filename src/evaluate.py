# -*- coding: utf-8 -*-
"""평가 지표. 모든 지표는 inverse_transform 으로 복원한 원래 kW 단위 값으로 계산한다."""

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def evaluate_regression(y_true, y_pred, peak_threshold):
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    mse = mean_squared_error(y_true, y_pred)

    true_peak = y_true >= peak_threshold
    pred_peak = y_pred >= peak_threshold
    n_peak = int(true_peak.sum())

    if n_peak > 0:
        peak_mae = mean_absolute_error(y_true[true_peak], y_pred[true_peak])
        peak_recall = recall_score(true_peak, pred_peak, zero_division=0)
        peak_precision = precision_score(true_peak, pred_peak, zero_division=0)
        peak_f1 = f1_score(true_peak, pred_peak, zero_division=0)
    else:
        peak_mae = peak_recall = peak_precision = peak_f1 = np.nan

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "peak_threshold": peak_threshold,
        "n_peak": n_peak,
        "peak_mae": peak_mae,
        "peak_recall": peak_recall,
        "peak_precision": peak_precision,
        "peak_f1": peak_f1,
    }
