"""Search and evaluate XGBoost and LightGBM beside the LHS base-nine models.

Only the tree models are trained. LSTM, TCN, and their weighted ensemble are
read from the existing seed-42 LHS result CSVs. Each tree candidate uses the
same cleaned data, nine past-only inputs, chronological folds, held-out Test
period, and 177 kW peak definition as the LHS experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LHS = ROOT / "lhs_cleaned"
sys.path.insert(0, str(LHS))

from src import data_pipeline as dp  # noqa: E402
from src.evaluate import (  # noqa: E402
    choose_alert_cutoff,
    evaluate_alerts,
    evaluate_regression,
    peak_probability,
)

SEED = 42
LOOKBACK = 168
GRID = {
    "xgboost": {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "min_child_weight": [1, 5],
        "colsample_bytree": [0.8, 1.0],
    },
    "lightgbm": {
        "num_leaves": [7, 15, 31],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 500, 1000],
        "min_child_samples": [20, 50],
        "colsample_bytree": [0.8, 1.0],
    },
}


def candidate_grid(model_name: str):
    grid = GRID[model_name]
    for values in itertools.product(*(grid[key] for key in grid)):
        yield dict(zip(grid, values))


def build_model(name: str, params: dict, n_jobs: int):
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            subsample=1.0,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=SEED,
            n_jobs=n_jobs,
            verbosity=0,
            **params,
        )
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        objective="regression",
        device_type="cpu",
        max_depth=-1,
        subsample=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=SEED,
        n_jobs=n_jobs,
        verbosity=-1,
        **params,
    )


def make_windows(df: pd.DataFrame):
    features = df[dp.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        features, LOOKBACK, axis=0
    )[:-1].transpose(0, 2, 1)
    indices = np.arange(LOOKBACK, len(df))
    assert len(windows) == len(indices)
    X = np.ascontiguousarray(windows.reshape(len(windows), -1))
    y = df[dp.TARGET_COLUMN].to_numpy(dtype=np.float32)[indices]
    keep = ~df["공장_셧다운_여부"].to_numpy(dtype=bool)[indices]
    return X, y, indices, keep


def split_indices(indices, keep, split):
    return np.flatnonzero(split.train[indices] & keep), np.flatnonzero(
        split.eval[indices] & keep
    )


def atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def search(name, X, y, indices, keep, folds, output, n_jobs, max_candidates):
    path = output / "grid_search" / f"{name}.csv"
    previous = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()
    rows = previous.to_dict("records")
    done = set(zip(previous.get("candidate", []), previous.get("fold", [])))
    candidates = list(candidate_grid(name))
    if max_candidates:
        candidates = candidates[:max_candidates]
    print(f"{name}: {len(candidates)} candidates × {len(folds)} folds", flush=True)
    for candidate_number, params in enumerate(candidates, start=1):
        for split in folds:
            key = (candidate_number, split.name)
            if key in done:
                continue
            train, evaluate = split_indices(indices, keep, split)
            started = time.perf_counter()
            model = build_model(name, params, n_jobs)
            model.fit(X[train], y[train])
            prediction = model.predict(X[evaluate])
            metrics = evaluate_regression(y[evaluate], prediction, dp.PEAK_THRESHOLD_KW)
            row = {
                "candidate": candidate_number,
                "fold": split.name,
                "seed": SEED,
                "lookback": LOOKBACK,
                "n_features": len(dp.FEATURE_COLUMNS),
                "n_train": len(train),
                "n_eval": len(evaluate),
                "params": json.dumps(params, sort_keys=True),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                **metrics,
            }
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), path)
            done.add(key)
            print(
                f"{name} {candidate_number}/{len(candidates)} {split.name}: "
                f"RMSE={metrics['rmse']:.3f} peakMAE={metrics['peak_mae']:.3f} "
                f"{row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    detail = pd.DataFrame(rows)
    completed = detail.groupby("candidate").filter(lambda group: len(group) == len(folds))
    summary = completed.groupby(["candidate", "params"], as_index=False).agg(
        rmse=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
        peak_mae=("peak_mae", "mean"),
    )
    if summary.empty:
        raise RuntimeError(f"No complete candidates for {name}")
    limit = summary.rmse.min() * 1.02
    eligible = summary.loc[summary.rmse <= limit]
    winner = eligible.sort_values(["peak_mae", "rmse"], na_position="last").iloc[0]
    atomic_csv(summary, output / "grid_search" / f"{name}_summary.csv")
    return json.loads(winner.params), int(winner.candidate), len(summary)


def evaluate_final(name, params, X, y, indices, keep, folds, final_split, output, n_jobs):
    val_predictions, val_actual, val_indices, val_folds = [], [], [], []
    for split in folds:
        train, evaluate = split_indices(indices, keep, split)
        model = build_model(name, params, n_jobs)
        model.fit(X[train], y[train])
        val_predictions.append(model.predict(X[evaluate]))
        val_actual.append(y[evaluate])
        val_indices.append(indices[evaluate])
        val_folds.extend([split.name] * len(evaluate))

    train, evaluate = split_indices(indices, keep, final_split)
    model = build_model(name, params, n_jobs)
    model.fit(X[train], y[train])
    predicted = model.predict(X[evaluate])
    actual = y[evaluate]
    val_pred = np.concatenate(val_predictions)
    val_y = np.concatenate(val_actual)
    sigma = float(np.std(val_y - val_pred, ddof=1))
    val_prob = peak_probability(val_pred, sigma, dp.PEAK_THRESHOLD_KW)
    cutoff = choose_alert_cutoff(val_y, val_prob, dp.PEAK_THRESHOLD_KW)
    probability = peak_probability(predicted, sigma, dp.PEAK_THRESHOLD_KW)
    metrics = {
        "model": name,
        "seed": SEED,
        "lookback": LOOKBACK,
        "n_features": len(dp.FEATURE_COLUMNS),
        "n_train": len(train),
        "n_test": len(evaluate),
        "sigma": sigma,
        **evaluate_regression(actual, predicted, dp.PEAK_THRESHOLD_KW),
        **evaluate_alerts(actual, probability, dp.PEAK_THRESHOLD_KW, cutoff),
        "params": json.dumps(params, sort_keys=True),
    }
    timestamps = df_datetime[indices[evaluate]]
    val_timestamps = df_datetime[np.concatenate(val_indices)]
    pred_dir = output / "predictions"
    atomic_csv(
        pd.DataFrame(
            {"datetime": timestamps, "actual": actual, "predicted": predicted, "probability": probability}
        ),
        pred_dir / f"{name}_test.csv",
    )
    atomic_csv(
        pd.DataFrame(
            {"datetime": val_timestamps, "fold": val_folds, "actual": val_y, "predicted": val_pred}
        ),
        pred_dir / f"{name}_val.csv",
    )
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / f"{name}_seed42.joblib")
    return metrics


def hash_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "okm_cleaned_2021.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "base9_tree")
    parser.add_argument("--models", nargs="+", choices=tuple(GRID), default=["lightgbm", "xgboost"])
    parser.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--max-candidates", type=int, default=None, help="Smoke test only")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = dp.load_data(args.data)
    global df_datetime
    df_datetime = df.datetime.to_numpy()
    X, y, indices, keep = make_windows(df)
    folds, final_split = dp.get_folds(df), dp.get_final_split(df)
    assert [sum(split.eval[indices] & keep) for split in folds] == [744, 720, 744]
    assert sum(final_split.eval[indices] & keep) == 703
    metadata = {
        "cleaned_data_sha256": hash_file(args.data),
        "seed": SEED,
        "feature_columns": dp.FEATURE_COLUMNS,
        "lookback": LOOKBACK,
        "peak_threshold_kw": dp.PEAK_THRESHOLD_KW,
        "selection": "within 2% of minimum mean fold RMSE, then minimum mean peak MAE",
        "train_rule": "fold train only; 14-day early-stop block excluded",
        "grid_size_per_tree": 108,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = []
    for name in args.models:
        params, candidate, complete = search(
            name, X, y, indices, keep, folds, args.out, args.n_jobs, args.max_candidates
        )
        print(f"{name}: selected candidate {candidate}/{complete}: {params}", flush=True)
        row = evaluate_final(
            name, params, X, y, indices, keep, folds, final_split, args.out, args.n_jobs
        )
        summary.append(row)
        atomic_csv(pd.DataFrame(summary), args.out / "test_metrics.csv")
        print(f"{name}: Test RMSE={row['rmse']:.4f}, F1={row['alert_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
