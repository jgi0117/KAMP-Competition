from __future__ import annotations

import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_preprocessed_npz
from .finetuning import FineTuneSettings
from .foundation_common import FoundationComparisonResult, resolve_device
from .models import (
    run_chronos2,
    run_lightgbm,
    run_moirai2,
    run_timesfm3,
    run_xgboost,
)
from .metrics import regression_metrics


SUPPORTED_MODELS = ("xgboost", "lightgbm", "timesfm3", "chronos2", "moirai2")


def load_config(path: str | Path) -> dict[str, Any]:
    import tomllib

    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def _write_predictions(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "horizon_step", "y_true", "y_pred"])
        for sample_index in range(y_true.shape[0]):
            for horizon_index in range(y_true.shape[1]):
                writer.writerow(
                    [
                        sample_index,
                        horizon_index + 1,
                        float(y_true[sample_index, horizon_index]),
                        float(y_pred[sample_index, horizon_index]),
                    ]
                )


def _base_row(model_name: str) -> dict[str, Any]:
    return {
        "model": model_name,
        "status": "error",
        "mode": "",
        "rmse": np.nan,
        "mae": np.nan,
        "r2": np.nan,
        "train_seconds": np.nan,
        "inference_seconds": np.nan,
        "best_params": "",
        "error": "",
    }


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    *,
    models: list[str] | None = None,
    quick: bool = False,
) -> pd.DataFrame:
    experiment = config.get("experiment", {})
    tree_config = config.get("tree", {})
    foundation = config.get("foundation", {})
    selected_models = models or list(experiment.get("models", SUPPORTED_MODELS))
    unknown = sorted(set(selected_models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {', '.join(unknown)}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = load_preprocessed_npz(
        data_path,
        context_length=int(experiment.get("context_length", 168)),
        horizon=int(experiment.get("horizon", 24)),
        target_feature_index=int(experiment.get("target_feature_index", 0)),
    )
    device = resolve_device(str(foundation.get("device", "auto")))
    batch_size = int(foundation.get("batch_size", 16))
    finetune_settings = FineTuneSettings.from_config(
        foundation.get("finetuning", {}),
        cv_splits=int(experiment.get("cv_splits", 3)),
        search_seed=int(experiment.get("seed", 42)),
        quick=quick,
    )
    continue_on_error = bool(experiment.get("continue_on_error", True))
    rows: list[dict[str, Any]] = []

    for model_name in selected_models:
        row = _base_row(model_name)
        try:
            if model_name in {"xgboost", "lightgbm"}:
                tree_runner = run_xgboost if model_name == "xgboost" else run_lightgbm
                result = tree_runner(
                    data,
                    destination,
                    seed=int(experiment.get("seed", 42)),
                    cv_splits=int(experiment.get("cv_splits", 3)),
                    n_jobs=int(tree_config.get("n_jobs", 1)),
                    search_profile=(
                        "quick" if quick else str(tree_config.get("search_profile", "full"))
                    ),
                )
                result.search_results.to_csv(
                    destination / f"search_results_{model_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                predictions = result.predictions
                row.update(
                    mode="trained_grid_search",
                    train_seconds=result.train_seconds,
                    inference_seconds=result.inference_seconds,
                    best_params=json.dumps(
                        result.best_params, ensure_ascii=False, sort_keys=True
                    ),
                )
            elif model_name == "timesfm3":
                settings = foundation.get("timesfm3", {})
                result = run_timesfm3(
                    data,
                    destination,
                    checkpoint=str(
                        settings.get("checkpoint", "google/timesfm-3.0-pytorch")
                    ),
                    device=device,
                    batch_size=batch_size,
                    settings=finetune_settings,
                )
            elif model_name == "chronos2":
                settings = foundation.get("chronos2", {})
                result = run_chronos2(
                    data,
                    destination,
                    checkpoint=str(settings.get("checkpoint", "amazon/chronos-2")),
                    device=device,
                    batch_size=batch_size,
                    settings=finetune_settings,
                )
            else:
                settings = foundation.get("moirai2", {})
                result = run_moirai2(
                    data,
                    destination,
                    checkpoint=str(
                        settings.get("checkpoint", "Salesforce/moirai-2.0-R-small")
                    ),
                    device=device,
                    batch_size=batch_size,
                    settings=finetune_settings,
                )

            if isinstance(result, FoundationComparisonResult):
                result.search_results.to_csv(
                    destination / f"finetune_search_results_{model_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                for variant in (result.zero_shot, result.finetuned):
                    variant_row = _base_row(variant.model_name)
                    variant_row.update(
                        status="ok",
                        mode=variant.mode,
                        train_seconds=variant.train_seconds,
                        inference_seconds=variant.inference_seconds,
                        best_params=json.dumps(
                            variant.details, ensure_ascii=False, sort_keys=True
                        ),
                        **regression_metrics(data.test.y, variant.predictions),
                    )
                    _write_predictions(
                        destination / "predictions" / f"{variant.model_name}.csv",
                        data.test.y,
                        variant.predictions,
                    )
                    rows.append(variant_row)
                continue

            metrics = regression_metrics(data.test.y, predictions)
            row.update(status="ok", **metrics)
            _write_predictions(
                destination / "predictions" / f"{model_name}.csv",
                data.test.y,
                predictions,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if not continue_on_error:
                rows.append(row)
                pd.DataFrame(rows).to_csv(
                    destination / "comparison.csv", index=False, encoding="utf-8-sig"
                )
                raise
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        ["status", "rmse"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)
    comparison.to_csv(destination / "comparison.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "created_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "data_path": str(Path(data_path).resolve()),
        "models": selected_models,
        "device": device,
        "context_length": data.context_length,
        "horizon": data.horizon,
        "finetuning": {
            "scopes": list(finetune_settings.scopes),
            "learning_rates": list(finetune_settings.learning_rates),
            "effective_batch_size": finetune_settings.effective_batch_size,
            "train_batch_size": finetune_settings.train_batch_size,
            "gradient_accumulation_steps": finetune_settings.gradient_accumulation_steps,
            "max_steps": finetune_settings.max_steps,
            "validation_interval": finetune_settings.validation_interval,
            "early_stopping_patience": finetune_settings.early_stopping_patience,
            "cv_splits": finetune_settings.cv_splits,
            "search_seed": finetune_settings.search_seed,
            "final_seed": finetune_settings.final_seed,
        },
        "sample_counts": {
            "train": data.train.n_samples,
            "val": data.val.n_samples,
            "test": data.test.n_samples,
        },
    }
    (destination / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison
