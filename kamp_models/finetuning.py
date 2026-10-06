from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


SCOPE_BLOCKS: dict[str, dict[str, int | None]] = {
    "chronos2": {
        "F0": 0,
        "F1": 1,
        "F2": 2,
        "F3": 3,
        "F4": 6,
        "F5": 9,
        "F6": 12,
        "F7": None,
    },
    "timesfm3": {
        "F0": 0,
        "F1": 1,
        "F2": 2,
        "F3": 5,
        "F4": 10,
        "F5": 15,
        "F6": 20,
        "F7": None,
    },
    "moirai2": {
        "F0": 0,
        "F1": 1,
        "F2": 2,
        "F3": 3,
        "F4": 4,
        "F5": 5,
        "F6": 6,
        "F7": None,
    },
}


@dataclass(frozen=True)
class FineTuneSettings:
    scopes: tuple[str, ...]
    learning_rates: tuple[float, ...]
    effective_batch_size: int
    train_batch_size: int
    max_steps: int
    validation_interval: int
    early_stopping_patience: int
    cv_splits: int
    search_seed: int
    final_seed: int

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.effective_batch_size % self.train_batch_size:
            raise ValueError(
                "effective_batch_size must be divisible by train_batch_size"
            )
        return self.effective_batch_size // self.train_batch_size

    @classmethod
    def from_config(
        cls,
        value: dict[str, Any],
        *,
        cv_splits: int,
        search_seed: int,
        quick: bool = False,
    ) -> "FineTuneSettings":
        scopes = tuple(str(item).upper() for item in value.get("scopes", SCOPE_BLOCKS["chronos2"]))
        invalid = sorted(set(scopes) - set(SCOPE_BLOCKS["chronos2"]))
        if invalid:
            raise ValueError(f"Unknown fine-tuning scopes: {', '.join(invalid)}")
        learning_rates = tuple(float(item) for item in value.get("learning_rates", [1e-6, 1e-5, 1e-4]))
        if not learning_rates or any(item <= 0 for item in learning_rates):
            raise ValueError("learning_rates must contain positive values")

        if quick:
            scopes = scopes[:1]
            learning_rates = learning_rates[:1]

        result = cls(
            scopes=scopes,
            learning_rates=learning_rates,
            effective_batch_size=int(value.get("effective_batch_size", 32)),
            train_batch_size=int(value.get("train_batch_size", 1)),
            max_steps=1 if quick else int(value.get("max_steps", 1000)),
            validation_interval=1 if quick else int(value.get("validation_interval", 100)),
            early_stopping_patience=1 if quick else int(value.get("early_stopping_patience", 3)),
            cv_splits=max(2, min(cv_splits, 2)) if quick else cv_splits,
            search_seed=search_seed,
            final_seed=search_seed if quick else int(value.get("final_seed", search_seed)),
        )
        if min(
            result.effective_batch_size,
            result.train_batch_size,
            result.max_steps,
            result.validation_interval,
            result.early_stopping_patience,
        ) <= 0:
            raise ValueError("Fine-tuning batch, step, interval, and patience values must be positive")
        _ = result.gradient_accumulation_steps
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _set_all(module: Any, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _enable(modules: Iterable[Any]) -> None:
    for module in modules:
        _set_all(module, True)


def apply_finetune_scope(model_name: str, model: Any, scope: str) -> dict[str, int]:
    """Freeze a foundation model according to the DOCX F0--F7 definitions."""

    scope = scope.upper()
    if model_name not in SCOPE_BLOCKS or scope not in SCOPE_BLOCKS[model_name]:
        raise ValueError(f"Unsupported scope {scope!r} for {model_name}")

    _set_all(model, False)
    block_count = SCOPE_BLOCKS[model_name][scope]
    if scope == "F7":
        _set_all(model, True)
    elif model_name == "chronos2":
        blocks = list(model.encoder.block)
        _enable([model.output_patch_embedding, model.encoder.final_layer_norm])
        if block_count:
            _enable(blocks[-block_count:])
    elif model_name == "timesfm3":
        blocks = list(model.transformer_stack.layers)
        _enable([model.output_head])
        if block_count:
            _enable(blocks[-block_count:])
    else:
        # Moirai2Forecast owns the pretrained module at .module.
        module = model.module if hasattr(model, "module") else model
        blocks = list(module.encoder.layers)
        _enable([module.out_proj, module.encoder.norm])
        if block_count:
            _enable(blocks[-block_count:])

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable == 0:
        raise RuntimeError(f"Scope {scope} left no trainable parameters in {model_name}")
    return {"trainable_params": trainable, "total_params": total}


def _trainable_state(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(model: Any, state: dict[str, Any]) -> None:
    parameters = dict(model.named_parameters())
    for name, value in state.items():
        parameters[name].data.copy_(value.to(parameters[name].device))


@dataclass
class TorchTrainResult:
    best_step: int
    best_rmse: float
    elapsed_seconds: float


def train_torch_point_model(
    model: Any,
    train_context: np.ndarray,
    train_y: np.ndarray,
    val_context: np.ndarray | None,
    val_y: np.ndarray | None,
    *,
    device: str,
    learning_rate: float,
    physical_batch_size: int,
    gradient_accumulation_steps: int,
    max_steps: int,
    validation_interval: int,
    early_stopping_patience: int,
    seed: int,
    forward_fn: Callable[[Any, Any, bool], Any],
) -> TorchTrainResult:
    """Fine-tune a point forecaster with optimizer-step based early stopping."""

    import torch

    seed_everything(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_context))
    cursor = 0

    def next_indices() -> np.ndarray:
        nonlocal order, cursor
        if cursor >= len(order):
            order = rng.permutation(len(train_context))
            cursor = 0
        end = min(cursor + physical_batch_size, len(order))
        selected = order[cursor:end]
        cursor = end
        return selected

    def evaluate() -> float:
        if val_context is None or val_y is None:
            return float("nan")
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(val_context), physical_batch_size):
                batch = torch.as_tensor(
                    val_context[start : start + physical_batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    predictions.append(forward_fn(model, batch, False).float().cpu())
        prediction = torch.cat(predictions).numpy()
        return float(np.sqrt(np.mean((prediction - val_y) ** 2)))

    best_rmse = float("inf")
    best_step = max_steps
    best_state: dict[str, Any] | None = None
    validations_without_improvement = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, max_steps + 1):
        model.train()
        for _ in range(gradient_accumulation_steps):
            selected = next_indices()
            x_batch = torch.as_tensor(
                train_context[selected], dtype=torch.float32, device=device
            )
            y_batch = torch.as_tensor(train_y[selected], dtype=torch.float32, device=device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                prediction = forward_fn(model, x_batch, True)
                loss = torch.nn.functional.mse_loss(prediction, y_batch)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        should_validate = val_context is not None and (
            step % validation_interval == 0 or step == max_steps
        )
        if should_validate:
            rmse = evaluate()
            if rmse < best_rmse:
                best_rmse = rmse
                best_step = step
                best_state = _trainable_state(model)
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
                if validations_without_improvement >= early_stopping_patience:
                    break

    if best_state is not None:
        _restore_trainable_state(model, best_state)
    return TorchTrainResult(
        best_step=best_step,
        best_rmse=best_rmse,
        elapsed_seconds=time.perf_counter() - started,
    )


def clone_trainable_state(model: Any) -> dict[str, Any]:
    """Public helper used by model-specific final checkpoint writers."""

    return copy.deepcopy(_trainable_state(model))
