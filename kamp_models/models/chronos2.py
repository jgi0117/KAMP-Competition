from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from ..data import DatasetBundle
from ..finetuning import FineTuneSettings, apply_finetune_scope, seed_everything
from ..foundation_common import (
    FoundationComparisonResult,
    FoundationVariant,
    best_setting,
    expanding_folds,
    histories,
)
from ..metrics import regression_metrics


class _ValidationTracker:
    best_step: int = 0
    best_loss: float = float("inf")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is not None and float(loss) < self.best_loss:
            self.best_loss = float(loss)
            self.best_step = int(state.global_step)
        return control


@contextmanager
def _finetune_scope(scope: str) -> Iterator[dict[str, int]]:
    from chronos.chronos2.trainer import Chronos2Trainer

    original_init = Chronos2Trainer.__init__
    counts: dict[str, int] = {}

    def scoped_init(instance, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if model is None:
            raise RuntimeError("Chronos trainer did not receive a model")
        counts.update(apply_finetune_scope("chronos2", model, scope))
        original_init(instance, *args, **kwargs)

    Chronos2Trainer.__init__ = scoped_init
    try:
        yield counts
    finally:
        Chronos2Trainer.__init__ = original_init


def _series(contexts: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    return [
        np.concatenate([context, target]).astype(np.float32, copy=False)
        for context, target in zip(contexts, targets, strict=True)
    ]


def _predict(
    pipeline: Any,
    contexts: np.ndarray,
    horizon: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    _, means = pipeline.predict_quantiles(
        inputs=[row.astype(np.float32, copy=False) for row in contexts],
        prediction_length=horizon,
        quantile_levels=[0.5],
        context_length=contexts.shape[1],
        batch_size=batch_size,
    )
    prediction = np.stack(
        [np.asarray(item, dtype=np.float32).reshape(-1)[:horizon] for item in means]
    )
    return prediction, time.perf_counter() - started


def _safe_remove(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if (
        resolved_path != resolved_root
        and resolved_root in resolved_path.parents
        and resolved_path.exists()
    ):
        shutil.rmtree(resolved_path)


def run_chronos2(
    data: DatasetBundle,
    output_dir: Path,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    settings: FineTuneSettings,
) -> FoundationComparisonResult:
    try:
        import torch
        from chronos import Chronos2Pipeline
        from transformers import EarlyStoppingCallback, TrainerCallback
    except ImportError as exc:
        raise ImportError("Install chronos-forecasting to run Chronos-2") from exc

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    def load_pipeline():
        return Chronos2Pipeline.from_pretrained(
            checkpoint, device_map=device, torch_dtype=dtype
        )

    history, targets, folds = expanding_folds(data, settings.cv_splits)
    _, _, test_history = histories(data)
    load_started = time.perf_counter()
    zero_pipeline = load_pipeline()
    load_seconds = time.perf_counter() - load_started
    zero_predictions, zero_inference = _predict(
        zero_pipeline, test_history, data.horizon, batch_size
    )
    del zero_pipeline

    callback_type = type(
        "ValidationTracker", (_ValidationTracker, TrainerCallback), {}
    )
    records: list[dict[str, Any]] = []
    candidate_root = output_dir / "models" / "chronos2_search"
    optimizer_name = "adamw_torch" if device == "cpu" else "adamw_torch_fused"

    for scope in settings.scopes:
        for learning_rate in settings.learning_rates:
            for fold_number, (train_index, val_index) in enumerate(folds, start=1):
                seed_everything(settings.search_seed)
                pipeline = load_pipeline()
                tracker = callback_type()
                candidate_dir = (
                    candidate_root
                    / f"{scope}_lr_{learning_rate:g}_fold_{fold_number}"
                )
                started = time.perf_counter()
                with _finetune_scope(scope) as counts:
                    pipeline = pipeline.fit(
                        inputs=_series(history[train_index], targets[train_index]),
                        validation_inputs=_series(
                            history[val_index], targets[val_index]
                        ),
                        prediction_length=data.horizon,
                        context_length=data.context_length,
                        finetune_mode="full",
                        learning_rate=learning_rate,
                        num_steps=settings.max_steps,
                        batch_size=settings.train_batch_size,
                        output_dir=candidate_dir,
                        callbacks=[
                            tracker,
                            EarlyStoppingCallback(
                                early_stopping_patience=settings.early_stopping_patience
                            ),
                        ],
                        remove_printer_callback=True,
                        gradient_accumulation_steps=settings.gradient_accumulation_steps,
                        eval_steps=settings.validation_interval,
                        save_steps=settings.validation_interval,
                        logging_steps=settings.validation_interval,
                        optim=optimizer_name,
                        seed=settings.search_seed,
                        data_seed=settings.search_seed,
                    )
                train_seconds = time.perf_counter() - started
                prediction, inference_seconds = _predict(
                    pipeline, history[val_index], data.horizon, batch_size
                )
                records.append(
                    {
                        "scope": scope,
                        "learning_rate": learning_rate,
                        "fold": fold_number,
                        "best_step": tracker.best_step or settings.max_steps,
                        "train_seconds": train_seconds,
                        "inference_seconds": inference_seconds,
                        **counts,
                        **regression_metrics(targets[val_index], prediction),
                    }
                )
                pd.DataFrame(records).to_csv(
                    output_dir / "finetune_search_results_chronos2.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                del pipeline
                _safe_remove(candidate_dir, candidate_root)

    search_results = pd.DataFrame(records)
    best_scope, best_lr, best_steps = best_setting(search_results)
    seed = settings.final_seed
    seed_everything(seed)
    pipeline = load_pipeline()
    final_dir = output_dir / "models" / "chronos2" / f"seed_{seed}"
    started = time.perf_counter()
    with _finetune_scope(best_scope) as final_counts:
        pipeline = pipeline.fit(
            inputs=_series(history, targets),
            prediction_length=data.horizon,
            context_length=data.context_length,
            finetune_mode="full",
            learning_rate=best_lr,
            num_steps=best_steps,
            batch_size=settings.train_batch_size,
            output_dir=final_dir,
            remove_printer_callback=True,
            gradient_accumulation_steps=settings.gradient_accumulation_steps,
            optim=optimizer_name,
            seed=seed,
            data_seed=seed,
        )
    final_train_seconds = time.perf_counter() - started
    final_prediction, final_inference_seconds = _predict(
        pipeline, test_history, data.horizon, batch_size
    )
    del pipeline

    details = {
        "checkpoint": checkpoint,
        "scope": best_scope,
        "learning_rate": best_lr,
        "steps": best_steps,
        "seed": seed,
        "effective_batch_size": settings.effective_batch_size,
        **final_counts,
    }
    selected_path = output_dir / "models" / "chronos2" / "selected_config.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return FoundationComparisonResult(
        zero_shot=FoundationVariant(
            model_name="chronos2_zs",
            predictions=zero_predictions,
            mode="zero_shot",
            train_seconds=0.0,
            inference_seconds=zero_inference,
            details={"checkpoint": checkpoint, "load_seconds": load_seconds},
        ),
        finetuned=FoundationVariant(
            model_name="chronos2_ft",
            predictions=final_prediction,
            mode=f"finetune_{best_scope}",
            train_seconds=final_train_seconds,
            inference_seconds=final_inference_seconds,
            details=details,
        ),
        search_results=search_results,
    )
