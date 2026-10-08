"""Validated held-out Test evidence for the five base-nine models."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PEAK_KW = 177.0
CONTROL_SIGMA = 3.0


def load_evidence():
    comparison = pd.read_csv(ROOT / "results/model_comparison.csv", encoding="utf-8-sig")
    expected = {"lstm", "tcn", "ensemble", "xgboost", "lightgbm"}
    if set(comparison.model) != expected:
        raise ValueError(f"Expected five model results, found {set(comparison.model)}")
    if not comparison.seed.eq(42).all() or not comparison.n_features.eq(9).all():
        raise ValueError("Comparison must use seed 42 and nine input variables")

    predictions = pd.read_csv(ROOT / "neural/results/final/test_predictions.csv")
    predictions["datetime"] = pd.to_datetime(predictions.datetime)
    predictions = predictions.sort_values("datetime").reset_index(drop=True)
    if len(predictions) != 703 or predictions.datetime.duplicated().any():
        raise ValueError("Expected 703 unique held-out Test hours")
    if not predictions.datetime.diff().dropna().le(pd.Timedelta(hours=18)).all():
        raise ValueError("Unexpected gap in Test predictions")

    settings = pd.read_csv(ROOT / "neural/results/final/alert_settings.csv", encoding="utf-8-sig").set_index("model")
    weights = pd.read_csv(ROOT / "neural/results/final/ensemble_weights.csv", encoding="utf-8-sig").set_index("model")
    cutoff = float(settings.loc["ensemble", "alert_cutoff"])
    residual_sigma = float(settings.loc["ensemble", "sigma"])
    predictions["predicted"] = predictions["pred_ensemble_seed42"]
    predictions["probability"] = predictions["prob_ensemble_seed42"]
    predictions["control_upper"] = predictions.predicted + CONTROL_SIGMA * residual_sigma
    predictions["control_lower"] = np.maximum(
        0.0, predictions.predicted - CONTROL_SIGMA * residual_sigma
    )
    predictions["control_violation"] = (
        (predictions.actual > predictions.control_upper)
        | (predictions.actual < predictions.control_lower)
    )
    predictions.attrs["residual_sigma"] = residual_sigma
    predictions["actual_peak"] = predictions.actual >= PEAK_KW
    predictions["alert"] = predictions.probability >= cutoff
    predictions["outcome"] = np.select(
        [predictions.actual_peak & ~predictions.alert,
         ~predictions.actual_peak & predictions.alert,
         predictions.actual_peak & predictions.alert],
        ["FN", "FP", "TP"], default="TN",
    )
    winner = comparison.loc[comparison.rmse.idxmin()]
    if winner.model != "ensemble":
        raise ValueError(f"Dashboard expects ensemble winner; found {winner.model}")
    return comparison, predictions, cutoff, weights.weight.to_dict()
