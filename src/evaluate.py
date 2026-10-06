# -*- coding: utf-8 -*-
"""평가 지표. 모든 지표는 inverse_transform 으로 복원한 원래 kW 단위 값으로 계산한다."""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

# 경보 기준 확률 후보 (검증 예측에서 F1 이 가장 높은 값을 고른다)
ALERT_CUTOFF_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def peak_probability(y_pred, sigma, threshold):
    """이상(피크) 확률 = P(실제 전력 ≥ 기준) ≈ 1 − Φ((기준 − 예측값) / σ).

    σ 는 검증 구간 예측 오차(실제 − 예측)의 표준편차. 예측값이 기준보다 낮아도
    오차를 감안하면 기준을 넘을 수 있다는 점을 확률로 나타낸다.
    """
    return 1.0 - norm.cdf((threshold - np.asarray(y_pred, dtype=np.float64)) / sigma)


def choose_alert_cutoff(y_true, prob, threshold):
    """검증 구간에서 F1 이 가장 높은 경보 기준 확률. 동점이면 더 낮은 값(놓침이 적은 쪽)."""
    event = np.asarray(y_true) >= threshold
    scores = [f1_score(event, prob >= c, zero_division=0) for c in ALERT_CUTOFF_GRID]
    return float(ALERT_CUTOFF_GRID[int(np.argmax(scores))])


def evaluate_alerts(y_true, prob, threshold, cutoff):
    """확률 ≥ 경보 기준이면 경보. 이상(피크) 탐지 지표와 확률 품질 지표."""
    event = np.asarray(y_true) >= threshold
    alert = np.asarray(prob) >= cutoff
    has_event = event.any()
    return {
        "alert_cutoff": cutoff,
        "n_event": int(event.sum()),
        "n_alert": int(alert.sum()),
        "tp": int((event & alert).sum()),
        "fn": int((event & ~alert).sum()),
        "fp": int((~event & alert).sum()),
        "alert_recall": recall_score(event, alert, zero_division=0) if has_event else np.nan,
        "alert_precision": precision_score(event, alert, zero_division=0),
        "alert_f1": f1_score(event, alert, zero_division=0) if has_event else np.nan,
        "brier": brier_score_loss(event, prob),
        "pr_auc": average_precision_score(event, prob) if has_event else np.nan,
    }


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
