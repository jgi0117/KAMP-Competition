from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import DatasetBundle, require_target_history
from .finetuning import (
    FineTuneSettings,
    apply_finetune_scope,
    clone_trainable_state,
    seed_everything,
    train_torch_point_model,
)
from .metrics import regression_metrics
from .splitting import TemporalFold, get_temporal_folds


@dataclass
class FoundationVariant:
    model_name: str
    predictions: np.ndarray
    mode: str
    train_seconds: float
    inference_seconds: float
    details: dict[str, Any]


@dataclass
class FoundationComparisonResult:
    zero_shot: FoundationVariant
    finetuned: FoundationVariant
    search_results: pd.DataFrame


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def histories(
    data: DatasetBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        require_target_history(data.train, "train"),
        require_target_history(data.val, "val"),
        require_target_history(data.test, "test"),
    )


def expanding_folds(
    data: DatasetBundle, n_splits: int
) -> tuple[np.ndarray, np.ndarray, list[TemporalFold]]:
    train_history, val_history, _ = histories(data)
    history = np.concatenate([train_history, val_history], axis=0)
    targets = np.concatenate([data.train.y, data.val.y], axis=0)
    folds = get_temporal_folds(data, n_splits)
    return history, targets, folds


def best_setting(search_results: pd.DataFrame) -> tuple[str, float, int]:
    summary = (
        search_results.groupby(["scope", "learning_rate"], as_index=False)
        .agg(
            mean_val_rmse=("rmse", "mean"),
            median_best_step=("best_step", "median"),
        )
        .sort_values(["mean_val_rmse", "scope", "learning_rate"])
        .iloc[0]
    )
    return (
        str(summary["scope"]),
        float(summary["learning_rate"]),
        max(1, int(round(float(summary["median_best_step"])))),
    )


def predict_torch(
    model: Any,
    contexts: np.ndarray,
    *,
    device: str,
    batch_size: int,
    forward_fn: Callable[[Any, Any, bool], Any],
) -> tuple[np.ndarray, float]:
    import torch

    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            batch = torch.as_tensor(
                contexts[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(
                forward_fn(model, batch, False).float().cpu().numpy()
            )
    if not predictions:
        raise ValueError("Cannot forecast an empty split")
    return np.concatenate(predictions, axis=0), time.perf_counter() - started


def save_trainable_checkpoint(
    model: Any,
    path: Path,
    *,
    checkpoint: str,
    scope: str,
    learning_rate: float,
    seed: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_checkpoint": checkpoint,
            "scope": scope,
            "learning_rate": learning_rate,
            "seed": seed,
            "trainable_state_dict": clone_trainable_state(model),
        },
        path,
    )


def run_torch_foundation_search(
    model_name: str,
    data: DatasetBundle,
    output_dir: Path,
    *,
    checkpoint: str,
    device: str,
    inference_batch_size: int,
    settings: FineTuneSettings,
    loader: Callable[[], Any],
    forward_fn: Callable[[Any, Any, bool], Any],
) -> FoundationComparisonResult:
    history, targets, folds = expanding_folds(data, settings.cv_splits)
    _, _, test_history = histories(data)

    load_started = time.perf_counter()
    zero_model = loader()
    load_seconds = time.perf_counter() - load_started
    zero_predictions, zero_inference = predict_torch(
        zero_model,
        test_history,
        device=device,
        batch_size=inference_batch_size,
        forward_fn=forward_fn,
    )
    del zero_model

    records: list[dict[str, Any]] = []
    for scope in settings.scopes:
        for learning_rate in settings.learning_rates:
            for fold_number, fold in enumerate(folds, start=1):
                seed_everything(settings.search_seed)
                model = loader()
                counts = apply_finetune_scope(model_name, model, scope)
                trained = train_torch_point_model(
                    model,
                    history[fold.train],
                    targets[fold.train],
                    history[fold.early_stop],
                    targets[fold.early_stop],
                    device=device,
                    learning_rate=learning_rate,
                    physical_batch_size=settings.train_batch_size,
                    gradient_accumulation_steps=settings.gradient_accumulation_steps,
                    max_steps=settings.max_steps,
                    validation_interval=settings.validation_interval,
                    early_stopping_patience=settings.early_stopping_patience,
                    seed=settings.search_seed,
                    forward_fn=forward_fn,
                )
                val_prediction, inference_seconds = predict_torch(
                    model,
                    history[fold.evaluate],
                    device=device,
                    batch_size=inference_batch_size,
                    forward_fn=forward_fn,
                )
                records.append(
                    {
                        "scope": scope,
                        "learning_rate": learning_rate,
                        "fold": fold.name or fold_number,
                        "n_train": len(fold.train),
                        "n_early_stop": len(fold.early_stop),
                        "n_eval": len(fold.evaluate),
                        "best_step": trained.best_step,
                        "train_seconds": trained.elapsed_seconds,
                        "inference_seconds": inference_seconds,
                        **counts,
                        **regression_metrics(
                            targets[fold.evaluate], val_prediction
                        ),
                    }
                )
                pd.DataFrame(records).to_csv(
                    output_dir / f"finetune_search_results_{model_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                del model
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    search_results = pd.DataFrame(records)
    best_scope, best_lr, best_steps = best_setting(search_results)
    seed = settings.final_seed
    seed_everything(seed)
    model = loader()
    final_counts = apply_finetune_scope(model_name, model, best_scope)
    trained = train_torch_point_model(
        model,
        history,
        targets,
        None,
        None,
        device=device,
        learning_rate=best_lr,
        physical_batch_size=settings.train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        max_steps=best_steps,
        validation_interval=settings.validation_interval,
        early_stopping_patience=settings.early_stopping_patience,
        seed=seed,
        forward_fn=forward_fn,
    )
    final_prediction, final_inference_seconds = predict_torch(
        model,
        test_history,
        device=device,
        batch_size=inference_batch_size,
        forward_fn=forward_fn,
    )
    save_trainable_checkpoint(
        model,
        output_dir / "models" / model_name / f"seed_{seed}_trainable.pt",
        checkpoint=checkpoint,
        scope=best_scope,
        learning_rate=best_lr,
        seed=seed,
    )
    del model

    details = {
        "checkpoint": checkpoint,
        "scope": best_scope,
        "learning_rate": best_lr,
        "steps": best_steps,
        "seed": seed,
        "effective_batch_size": settings.effective_batch_size,
        **final_counts,
    }
    return FoundationComparisonResult(
        zero_shot=FoundationVariant(
            model_name=f"{model_name}_zs",
            predictions=zero_predictions,
            mode="zero_shot",
            train_seconds=0.0,
            inference_seconds=zero_inference,
            details={"checkpoint": checkpoint, "load_seconds": load_seconds},
        ),
        finetuned=FoundationVariant(
            model_name=f"{model_name}_ft",
            predictions=final_prediction,
            mode=f"finetune_{best_scope}",
            train_seconds=trained.elapsed_seconds,
            inference_seconds=final_inference_seconds,
            details=details,
        ),
        search_results=search_results,
    )
