from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.multioutput import MultiOutputRegressor

from ..data import DatasetBundle
from ..tree_models import TreeRunResult, run_tree_search


FULL_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 500, 1000],
    "min_child_weight": [1, 5],
    "colsample_bytree": [0.8, 1.0],
}

QUICK_GRID = {
    "max_depth": [5],
    "learning_rate": [0.05],
    "n_estimators": [20],
    "min_child_weight": [1],
    "colsample_bytree": [1.0],
}


def _build_estimator(
    params: dict[str, Any], seed: int, n_jobs: int
) -> MultiOutputRegressor:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost to run the XGBoost candidate") from exc

    base = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        **params,
    )
    return MultiOutputRegressor(base, n_jobs=n_jobs)


def run_xgboost(
    data: DatasetBundle,
    output_dir: Path,
    **settings: Any,
) -> TreeRunResult:
    return run_tree_search(
        "xgboost",
        data,
        output_dir,
        full_grid=FULL_GRID,
        quick_grid=QUICK_GRID,
        estimator_factory=_build_estimator,
        **settings,
    )
