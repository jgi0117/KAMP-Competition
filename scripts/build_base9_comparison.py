"""Combine the held-out Test results after both full tree searches finish."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LHS_RESULTS = ROOT / "lhs_cleaned" / "results" / "final"
TREE_RESULTS = ROOT / "results" / "base9_tree"


def main():
    lhs_predictions = pd.read_csv(LHS_RESULTS / "test_predictions.csv")
    lhs_metrics = pd.read_csv(LHS_RESULTS / "test_metrics_per_seed.csv", encoding="utf-8-sig")
    tree_metrics = pd.read_csv(TREE_RESULTS / "test_metrics.csv", encoding="utf-8-sig")
    expected = {"xgboost", "lightgbm"}
    if set(tree_metrics.model) != expected:
        raise ValueError(f"Tree results incomplete: {set(tree_metrics.model)}")

    for name in sorted(expected):
        detail = pd.read_csv(TREE_RESULTS / "grid_search" / f"{name}.csv")
        counts = detail.groupby("candidate").fold.nunique()
        if len(counts) != 108 or not counts.eq(3).all() or len(detail) != 324:
            raise ValueError(f"{name} grid incomplete: {len(detail)} fold rows")
        prediction = pd.read_csv(TREE_RESULTS / "predictions" / f"{name}_test.csv")
        if not pd.to_datetime(prediction.datetime).equals(pd.to_datetime(lhs_predictions.datetime)):
            raise ValueError(f"{name} Test timestamps differ from LHS")
        if not np.allclose(prediction.actual, lhs_predictions.actual, atol=1e-6):
            raise ValueError(f"{name} Test target differs from LHS")

    models = ["lstm", "tcn", "ensemble", "xgboost", "lightgbm"]
    rows = pd.concat(
        [lhs_metrics.loc[lhs_metrics.model.isin(models)], tree_metrics],
        ignore_index=True,
    )
    rows = rows.set_index("model").loc[models].reset_index()
    rows.insert(1, "source", ["lhs_existing"] * 3 + ["base9_retrained"] * 2)
    rows["n_features"] = 9
    rows["lookback"] = rows.model.map(
        {"lstm": 168, "tcn": 24, "ensemble": np.nan, "xgboost": 168, "lightgbm": 168}
    )
    columns = [
        "model", "source", "seed", "n_features", "lookback", "rmse", "mae", "r2",
        "alert_f1", "alert_recall", "alert_precision", "fn", "fp", "pr_auc",
    ]
    output = ROOT / "results" / "base9_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows[columns].to_csv(output, index=False, encoding="utf-8-sig")
    print(rows[columns].to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
