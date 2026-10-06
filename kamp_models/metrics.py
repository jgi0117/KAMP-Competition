from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the three competition metrics over every sample and horizon."""

    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={truth.shape}, y_pred={pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError("Predictions contain NaN or infinite values")

    truth_flat = truth.reshape(-1)
    pred_flat = pred.reshape(-1)
    return {
        "rmse": float(mean_squared_error(truth_flat, pred_flat) ** 0.5),
        "mae": float(mean_absolute_error(truth_flat, pred_flat)),
        "r2": float(r2_score(truth_flat, pred_flat)),
    }

