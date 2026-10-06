from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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
from ..progress import log_progress


class _ValidationTracker:
    def __init__(self, progress=None):
        self.best_step = 0
        self.best_loss = float("inf")
        self.progress = progress
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress is not None:
            current = int(state.global_step)
            self.progress.update(max(0, current - self.last_step))
            self.last_step = current
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is not None and self.progress is not None:
            self.progress.set_postfix(eval_loss=f"{float(loss):.4f}")
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
    log_progress(f"chronos2 | loading zero-shot checkpoint {checkpoint}")
    load_started = time.perf_counter()
    zero_pipeline = load_pipeline()
    load_seconds = time.perf_counter() - load_started
    log_progress(
        f"chronos2 | checkpoint loaded in {load_seconds:.1f}s; zero-shot inference started"
    )
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
    trial_total = len(settings.scopes) * len(settings.learning_rates) * len(folds)
    trial_number = 0

    for scope in settings.scopes:
        for learning_rate in settings.learning_rates:
            for fold_number, fold in enumerate(folds, start=1):
                trial_number += 1
                trial_name = (
                    f"chronos2 {scope} lr={learning_rate:g} "
                    f"{fold.name or fold_number}"
                )
                log_progress(
                    f"chronos2 fine-tuning trial {trial_number}/{trial_total} | "
                    f"{scope} | lr={learning_rate:g} | {fold.name or fold_number}"
                )
                seed_everything(settings.search_seed)
                pipeline = load_pipeline()
                step_progress = tqdm(
                    total=settings.max_steps,
                    desc=trial_name,
                    unit="step",
                    dynamic_ncols=True,
                    leave=False,
                )
                tracker = callback_type(step_progress)
                candidate_dir = (
                    candidate_root
                    / f"{scope}_lr_{learning_rate:g}_fold_{fold_number}"
                )
                started = time.perf_counter()
                try:
                    with _finetune_scope(scope) as counts:
                        pipeline = pipeline.fit(
                            inputs=_series(history[fold.train], targets[fold.train]),
                            validation_inputs=_series(
                                history[fold.early_stop], targets[fold.early_stop]
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
                finally:
                    step_progress.close()
                train_seconds = time.perf_counter() - started
                prediction, inference_seconds = _predict(
                    pipeline, history[fold.evaluate], data.horizon, batch_size
                )
                records.append(
                    {
                        "scope": scope,
                        "learning_rate": learning_rate,
                        "fold": fold.name or fold_number,
                        "n_train": len(fold.train),
                        "n_early_stop": len(fold.early_stop),
                        "n_eval": len(fold.evaluate),
                        "best_step": tracker.best_step or settings.max_steps,
                        "train_seconds": train_seconds,
                        "inference_seconds": inference_seconds,
                        **counts,
                        **regression_metrics(targets[fold.evaluate], prediction),
                    }
                )
                pd.DataFrame(records).to_csv(
                    output_dir / "finetune_search_results_chronos2.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                log_progress(
                    f"chronos2 trial {trial_number}/{trial_total} completed | "
                    f"best_step={tracker.best_step or settings.max_steps} | "
                    f"RMSE={records[-1]['rmse']:.4f}"
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
    log_progress(
        f"chronos2 final training | scope={best_scope} | lr={best_lr:g} | "
        f"steps={best_steps} | device={device}"
    )
    final_progress = tqdm(
        total=best_steps,
        desc=f"chronos2 final {best_scope}",
        unit="step",
        dynamic_ncols=True,
        leave=False,
    )
    final_tracker = callback_type(final_progress)
    try:
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
                callbacks=[final_tracker],
                remove_printer_callback=True,
                gradient_accumulation_steps=settings.gradient_accumulation_steps,
                logging_steps=settings.validation_interval,
                optim=optimizer_name,
                seed=seed,
                data_seed=seed,
            )
    finally:
        final_progress.close()
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
