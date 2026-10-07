from __future__ import annotations

import numpy as np
import pytest

from kamp_models.data import DatasetBundle, DatasetSplit, load_preprocessed_npz
from kamp_models.finetuning import (
    FineTuneSettings,
    apply_finetune_scope,
    train_torch_point_model,
)
from kamp_models.foundation_common import expanding_folds, resolve_device
from kamp_models.metrics import regression_metrics
from kamp_models.models.lightgbm import _build_estimator as build_lightgbm_estimator
from kamp_models.models.lightgbm import _resolve_device as resolve_lightgbm_device
from kamp_models.models.xgboost import _build_estimator as build_xgboost_estimator
from kamp_models.models.xgboost import _resolve_device as resolve_xgboost_device
from kamp_models.tree_models import run_tree_model


def test_load_preprocessed_npz_and_infer_history(tmp_path):
    horizon = 3
    context = 5
    payload = {}
    for split, count in (("train", 8), ("val", 4), ("test", 3)):
        X = np.arange(count * context * 2, dtype=np.float32).reshape(
            count, context, 2
        )
        payload[f"X_{split}"] = X
        payload[f"y_{split}"] = np.ones((count, horizon), dtype=np.float32)
    path = tmp_path / "data.npz"
    np.savez(path, **payload)

    bundle = load_preprocessed_npz(
        path,
        context_length=context,
        horizon=horizon,
        target_feature_index=1,
    )

    assert bundle.train.target_history.shape == (8, context)
    np.testing.assert_array_equal(bundle.test.target_history, payload["X_test"][:, :, 1])


def test_regression_metrics():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [2.0, 5.0]])

    metrics = regression_metrics(y_true, y_pred)

    assert np.isclose(metrics["rmse"], np.sqrt(0.5))
    assert np.isclose(metrics["mae"], 0.5)
    assert np.isclose(metrics["r2"], 0.6)


@pytest.mark.parametrize(
    ("model_name", "param_grid"),
    [
        (
            "xgboost",
            {
                "max_depth": [2],
                "learning_rate": [0.1],
                "n_estimators": [4],
                "min_child_weight": [1],
                "colsample_bytree": [1.0],
            },
        ),
        (
            "lightgbm",
            {
                "num_leaves": [7],
                "learning_rate": [0.1],
                "n_estimators": [4],
                "min_child_samples": [2],
                "colsample_bytree": [1.0],
            },
        ),
    ],
)
def test_tree_model_training_pipeline(tmp_path, model_name, param_grid):
    rng = np.random.default_rng(42)

    def make_split(count):
        X = rng.normal(size=(count, 5)).astype(np.float32)
        y = (0.7 * X[:, -1] + 0.2 * X[:, -2])[:, None].astype(np.float32)
        return DatasetSplit(X=X, y=y, target_history=X)

    bundle = DatasetBundle(
        train=make_split(18),
        val=make_split(8),
        test=make_split(5),
        context_length=5,
        horizon=1,
    )

    result = run_tree_model(
        model_name,
        bundle,
        tmp_path,
        seed=42,
        cv_splits=2,
        n_jobs=1,
        param_grid=param_grid,
    )

    assert result.predictions.shape == (5, 1)
    assert result.model_path.is_file()
    assert set(("rmse", "mae", "r2")).issubset(result.search_results.columns)


def test_tree_estimators_map_cuda_device():
    xgboost = build_xgboost_estimator({}, seed=42, n_jobs=1, device="cuda")
    lightgbm = build_lightgbm_estimator({}, seed=42, n_jobs=1, device="cuda")

    assert xgboost.estimator.get_params()["device"] == "cuda"
    assert lightgbm.estimator.get_params()["device_type"] == "gpu"


def test_lightgbm_falls_back_to_cpu_when_gpu_build_is_missing(monkeypatch):
    import lightgbm
    from lightgbm.basic import LightGBMError

    class NoGpuRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, targets):
            raise LightGBMError("GPU Tree Learner was not enabled in this build.")

    monkeypatch.setattr(lightgbm, "LGBMRegressor", NoGpuRegressor)

    with pytest.warns(RuntimeWarning, match="retrying this model on CPU"):
        assert resolve_lightgbm_device("cuda") == "cpu"


def test_xgboost_falls_back_to_cpu_when_gpu_is_not_accessible(monkeypatch):
    import xgboost

    class NoGpuRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, targets):
            import warnings

            warnings.warn("No visible GPU is found, setting device to CPU.")

    monkeypatch.setattr(xgboost, "XGBRegressor", NoGpuRegressor)

    with pytest.warns(RuntimeWarning, match="cannot access a CUDA device"):
        assert resolve_xgboost_device("cuda") == "cpu"


def test_auto_device_matches_torch_cuda_availability():
    torch = pytest.importorskip("torch")

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected


def test_finetune_settings_match_document_conditions():
    settings = FineTuneSettings.from_config({}, cv_splits=3, search_seed=42)

    assert settings.scopes == ("F0", "F2", "F4", "F7")
    assert settings.learning_rates == (1e-6, 1e-5, 1e-4)
    assert settings.gradient_accumulation_steps == 32
    assert settings.max_steps == 1000
    assert settings.validation_interval == 100
    assert settings.early_stopping_patience == 3
    assert settings.final_seed == 42


def test_foundation_single_holdout_uses_train_and_validation_once():
    def make_split(count):
        values = np.arange(count * 4, dtype=np.float32).reshape(count, 4)
        return DatasetSplit(
            X=values,
            y=values[:, :1],
            target_history=values,
        )

    bundle = DatasetBundle(
        train=make_split(6),
        val=make_split(2),
        test=make_split(1),
        context_length=4,
        horizon=1,
    )

    _, _, folds = expanding_folds(bundle, n_splits=1)

    assert len(folds) == 1
    assert folds[0].name == "holdout"
    np.testing.assert_array_equal(folds[0].train, np.arange(6))
    np.testing.assert_array_equal(folds[0].early_stop, np.array([6, 7]))
    np.testing.assert_array_equal(folds[0].evaluate, np.array([6, 7]))


def test_timesfm_scope_only_unfreezes_requested_tail_and_head():
    torch = pytest.importorskip("torch")

    class FakeTimesFM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(2, 2)
            self.transformer_stack = torch.nn.Module()
            self.transformer_stack.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(20)]
            )
            self.output_head = torch.nn.Linear(2, 1)

    model = FakeTimesFM()
    counts = apply_finetune_scope("timesfm3", model, "F3")

    assert not any(parameter.requires_grad for parameter in model.input.parameters())
    assert not any(
        parameter.requires_grad
        for layer in model.transformer_stack.layers[:-5]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.transformer_stack.layers[-5:]
        for parameter in layer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.output_head.parameters())
    assert 0 < counts["trainable_params"] < counts["total_params"]


def test_common_finetune_loop_runs_optimizer_steps_on_cpu():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 4)).astype(np.float32)
    y = X.sum(axis=1, keepdims=True).astype(np.float32)
    model = torch.nn.Linear(4, 1)

    result = train_torch_point_model(
        model,
        X[:8],
        y[:8],
        X[8:],
        y[8:],
        device="cpu",
        learning_rate=1e-2,
        physical_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        validation_interval=1,
        early_stopping_patience=2,
        seed=42,
        forward_fn=lambda current, batch, training: current(batch),
    )

    assert 1 <= result.best_step <= 2
    assert np.isfinite(result.best_rmse)
