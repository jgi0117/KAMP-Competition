from __future__ import annotations

from pathlib import Path

from ..data import DatasetBundle
from ..finetuning import FineTuneSettings
from ..foundation_common import FoundationComparisonResult, run_torch_foundation_search


def run_moirai2(
    data: DatasetBundle,
    output_dir: Path,
    *,
    checkpoint: str,
    device: str,
    batch_size: int,
    settings: FineTuneSettings,
) -> FoundationComparisonResult:
    try:
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    except ImportError as exc:
        raise ImportError("Install uni2ts to run Moirai 2.0") from exc

    def loader():
        return Moirai2Forecast(
            module=Moirai2Module.from_pretrained(checkpoint),
            prediction_length=data.horizon,
            context_length=data.context_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(device)

    def forward_fn(model, contexts, training):
        import torch

        batch = contexts.unsqueeze(-1)
        prediction = model(
            past_target=batch,
            past_observed_target=torch.ones_like(batch, dtype=torch.bool),
            past_is_pad=torch.zeros(
                batch.shape[:2], dtype=torch.bool, device=batch.device
            ),
        )
        median_index = len(model.module.quantile_levels) // 2
        return prediction[:, median_index, : data.horizon].to(torch.float32)

    return run_torch_foundation_search(
        "moirai2",
        data,
        output_dir,
        checkpoint=checkpoint,
        device=device,
        inference_batch_size=batch_size,
        settings=settings,
        loader=loader,
        forward_fn=forward_fn,
    )
