from __future__ import annotations

from pathlib import Path

from ..data import DatasetBundle
from ..finetuning import FineTuneSettings
from ..foundation_common import FoundationComparisonResult, run_torch_foundation_search


def run_timesfm3(
    data: DatasetBundle,
    output_dir: Path,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    settings: FineTuneSettings,
) -> FoundationComparisonResult:
    try:
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError as exc:
        raise ImportError("Install timesfm to run TimesFM 3.0") from exc

    def loader():
        evaluator = TimesFM3Evaluator(
            ModelConfig(
                checkpoint_path=checkpoint,
                per_core_batch_size=batch_size,
                device=device,
            )
        )
        evaluator.model._kamp_median_quantile_index = (
            evaluator.config.median_quantile_index
        )
        return evaluator.model

    def forward_fn(model, contexts, training):
        import torch

        values = contexts.unsqueeze(1)
        if training:
            undecorated = getattr(model.decode, "__wrapped__", None)
            if undecorated is None:
                raise RuntimeError(
                    "TimesFM decode no longer exposes its differentiable implementation"
                )
            logits = undecorated(model, values, horizon=data.horizon)
        else:
            logits = model.decode(values, horizon=data.horizon)
        median_index = int(
            getattr(
                model,
                "_kamp_median_quantile_index",
                len(model.quantiles) // 2,
            )
        )
        return logits[:, 0, :, median_index].to(torch.float32)

    return run_torch_foundation_search(
        "timesfm3",
        data,
        output_dir,
        checkpoint=checkpoint,
        device=device,
        inference_batch_size=batch_size,
        settings=settings,
        loader=loader,
        forward_fn=forward_fn,
    )
