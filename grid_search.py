# -*- coding: utf-8 -*-
"""LSTM · TCN · TCN-LSTM 단계별 그리드 서치 (CPU 기준).

전체 조합을 한 번에 곱하지 않고 단계별로 탐색한다. 각 단계는 이전 단계에서 고른 값을 고정한다.

    Stage 1  lookback
    Stage 2  구조 (units / layers / filters / kernel / dilation)
    Stage 3  learning_rate × dropout
    Stage 4  batch_size
    confirm  상위 3개 설정을 seed 17·42·73 으로 재확인

모든 설정은 시간순 확장 fold 3개(src/data_pipeline.FOLD_VAL_RANGES)의 평균으로 비교한다.
선택 규칙: fold 평균 RMSE 가 최저값 대비 2% 이내인 설정 중 피크 구간 MAE 가 가장 작은 설정.
Test 구간(8/16~)은 이 스크립트에서 사용하지 않는다.

사용 예:
    python grid_search.py --model lstm --stage 1
    python grid_search.py --model lstm --stage 2
    python grid_search.py --model lstm --stage confirm
    python grid_search.py --model lstm --summary
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from src import data_pipeline as dp
from src.evaluate import evaluate_regression

RESULTS_DIR = Path("results/grid_search")

DEFAULT_SEED = 42
CONFIRM_SEEDS = [17, 42, 73]
CONFIRM_TOP_K = 3
RMSE_TOLERANCE = 0.02
MAX_EPOCHS = 100
PATIENCE = 10

# 가이드 25장 공통 초기 조건 + 28~30장 튜닝 범위를 단계별로 나눈 것
SEARCH_SPACE = {
    "lstm": {
        "base": dict(lookback=24, hidden_units=64, num_layers=1,
                     dropout=0.2, learning_rate=1e-3, batch_size=32),
        "stages": [
            {"lookback": [24, 48, 72, 168]},
            {"hidden_units": [32, 64, 128], "num_layers": [1, 2]},
            {"learning_rate": [1e-4, 3e-4, 1e-3], "dropout": [0.0, 0.1, 0.2, 0.3]},
            {"batch_size": [16, 32, 64]},
        ],
    },
    "tcn": {
        "base": dict(lookback=24, filters=64, kernel_size=3, dilations="auto",
                     dropout=0.2, learning_rate=1e-3, batch_size=32),
        "stages": [
            {"lookback": [24, 48, 72, 168]},
            {"filters": [32, 64, 128], "kernel_size": [2, 3, 5], "dilations": ["auto", "auto+1"]},
            {"learning_rate": [1e-4, 3e-4, 1e-3], "dropout": [0.0, 0.1, 0.2, 0.3]},
            {"batch_size": [16, 32, 64]},
        ],
    },
    "tcn_lstm": {
        "base": dict(lookback=24, filters=64, kernel_size=3, dilations="auto", lstm_units=64,
                     dropout=0.2, learning_rate=1e-3, batch_size=32),
        "stages": [
            {"lookback": [24, 48, 72, 168]},
            {"filters": [32, 64], "kernel_size": [2, 3, 5], "lstm_units": [32, 64, 128]},
            {"learning_rate": [1e-4, 3e-4, 1e-3], "dropout": [0.1, 0.2, 0.3]},
            {"batch_size": [16, 32, 64]},
        ],
    },
}

KEY_COLS = ["config_id", "seed", "fold"]


# =========================================================
# 모델 생성
# =========================================================

def build_model(model_name, params, input_shape):
    if model_name == "lstm":
        from src.models.lstm_model import build_lstm
        return build_lstm(input_shape, hidden_units=params["hidden_units"],
                          num_layers=params["num_layers"], dropout=params["dropout"],
                          learning_rate=params["learning_rate"])
    if model_name == "tcn":
        from src.models.tcn_model import build_tcn
        return build_tcn(input_shape, filters=params["filters"],
                         kernel_size=params["kernel_size"], dilations=params["dilations"],
                         dropout=params["dropout"], learning_rate=params["learning_rate"])
    if model_name == "tcn_lstm":
        from src.models.tcn_lstm_model import build_tcn_lstm
        return build_tcn_lstm(input_shape, filters=params["filters"],
                              kernel_size=params["kernel_size"], dilations=params["dilations"],
                              lstm_units=params["lstm_units"], dropout=params["dropout"],
                              learning_rate=params["learning_rate"])
    raise ValueError(model_name)


def config_id(params):
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


# =========================================================
# 결과 저장 / 집계
# =========================================================

def results_path(model_name):
    return RESULTS_DIR / f"{model_name}.csv"


def load_results(model_name):
    path = results_path(model_name)
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


def append_result(model_name, row):
    path = results_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        # 기존 헤더 순서에 맞춰 써야 열이 밀리지 않는다
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns
        frame = frame.reindex(columns=header)
    frame.to_csv(path, mode="a", header=not path.exists(),
                 index=False, encoding="utf-8-sig")


def aggregate(results, config_ids, n_folds, seeds=None):
    """설정별 fold(및 seed) 평균. 모든 fold 가 끝난 설정만 남긴다."""
    df = results[results["config_id"].isin(config_ids)]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("config_id").agg(
        mse=("mse", "mean"),
        rmse=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae=("mae", "mean"),
        r2=("r2", "mean"),
        peak_mae=("peak_mae", "mean"),
        peak_recall=("peak_recall", "mean"),
        peak_f1=("peak_f1", "mean"),
        n_runs=("rmse", "size"),
        train_seconds=("train_seconds", "mean"),
    ).reset_index()
    expected = n_folds * (len(seeds) if seeds is not None else 1)
    return agg[agg["n_runs"] >= expected].sort_values("rmse").reset_index(drop=True)


def select_best(agg):
    """RMSE 최저 대비 2% 이내 후보 중 피크 MAE 최소 (피크 표본이 없으면 RMSE 순)."""
    limit = agg["rmse"].min() * (1 + RMSE_TOLERANCE)
    candidates = agg[agg["rmse"] <= limit]
    if candidates["peak_mae"].notna().any():
        return candidates.sort_values("peak_mae").iloc[0]
    return candidates.sort_values("rmse").iloc[0]


# =========================================================
# 단계별 설정 목록
# =========================================================

def stage_configs(model_name, stage, results, n_folds):
    """stage(1부터)의 설정 목록. 이전 단계 최적값을 base 에 반영한다."""
    space = SEARCH_SPACE[model_name]
    params = dict(space["base"])

    for prev in range(1, stage):
        prev_ids = [config_id(p) for p in expand(params, space["stages"][prev - 1])]
        agg = aggregate(results, prev_ids, n_folds, seeds=[DEFAULT_SEED])
        if len(agg) < len(prev_ids):
            raise RuntimeError(
                f"Stage {prev} 가 끝나지 않았습니다 ({len(agg)}/{len(prev_ids)} 설정 완료). "
                f"먼저 --stage {prev} 를 실행하세요."
            )
        params = json.loads(select_best(agg)["config_id"])

    return expand(params, space["stages"][stage - 1])


def expand(base, grid):
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        p = dict(base)
        p.update(dict(zip(keys, values)))
        configs.append(p)
    return configs


def all_stage_ids(model_name, results, n_folds):
    ids = []
    for stage in range(1, len(SEARCH_SPACE[model_name]["stages"]) + 1):
        try:
            ids += [config_id(p) for p in stage_configs(model_name, stage, results, n_folds)]
        except RuntimeError:
            break
    return list(dict.fromkeys(ids))


def final_choice(model_name, results, n_folds):
    """최종 설정 1개와 그 근거. confirm(seed 3개)이 끝났으면 그 결과로, 아니면 seed 42 결과로 고른다.

    반환: (설정 dict, 집계 행, 근거 문자열) 또는 결과가 없으면 (None, None, None)
    """
    if results.empty:
        return None, None, None
    ids = all_stage_ids(model_name, results, n_folds)
    agg = aggregate(results, ids, n_folds, CONFIRM_SEEDS)
    basis = "confirm (seed 3개 × fold 평균)"
    if agg.empty:
        agg = aggregate(results, ids, n_folds, [DEFAULT_SEED])
        basis = "seed 42 결과 (confirm 미완료)"
    if agg.empty:
        return None, None, None
    best = select_best(agg)
    return json.loads(best["config_id"]), best, basis


# =========================================================
# 학습 1회
# =========================================================

_split_cache = {}


def get_split_data(df, split, lookback):
    key = (split.name, lookback)
    if key not in _split_cache:
        _split_cache[key] = dp.prepare_split(df, split, lookback)
    return _split_cache[key]


def train_model(model_name, params, seed, data, max_epochs):
    """학습 구간으로 학습하고 EarlyStopping 구간으로 멈춘다. final_evaluate.py 도 이 함수를 쓴다."""
    from tensorflow import keras

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    model = build_model(model_name, params, data["X_train"].shape[1:])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    start = time.time()
    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_es"], data["y_es"]),
        epochs=max_epochs,
        batch_size=params["batch_size"],
        callbacks=[early_stopping],
        shuffle=True,
        verbose=0,
    )
    val_loss = history.history["val_loss"]
    info = {
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "epochs_run": len(val_loss),
        "train_seconds": round(time.time() - start, 1),
        "n_train": len(data["y_train"]),
        "n_eval": len(data["y_eval"]),
    }
    return model, info


def predict_kw(model, data):
    """평가 구간 예측값과 실제값을 kW 단위로 복원해 반환한다."""
    y_scaler = data["y_scaler"]
    pred = y_scaler.inverse_transform(model.predict(data["X_eval"], verbose=0)).ravel()
    actual = y_scaler.inverse_transform(data["y_eval"]).ravel()
    return pred, actual


def run_one(model_name, params, seed, df, split, max_epochs):
    data = get_split_data(df, split, params["lookback"])
    model, info = train_model(model_name, params, seed, data, max_epochs)
    pred, actual = predict_kw(model, data)
    metrics = evaluate_regression(actual, pred, data["peak_threshold"])
    return {**metrics, **info}


def run_configs(model_name, stage_label, configs, seeds, df, folds, max_epochs):
    results = load_results(model_name)
    done = set()
    if not results.empty:
        done = set(zip(results["config_id"], results["seed"], results["fold"]))

    jobs = [(p, s, f) for p in configs for s in seeds for f in folds
            if (config_id(p), s, f.name) not in done]
    total = len(configs) * len(seeds) * len(folds)
    print(f"[{model_name} / stage {stage_label}] 설정 {len(configs)}개 × seed {len(seeds)} × "
          f"fold {len(folds)} = {total}회 (이미 완료 {total - len(jobs)}회, 남은 {len(jobs)}회)")

    for n, (params, seed, split) in enumerate(jobs, start=1):
        metrics = run_one(model_name, params, seed, df, split, max_epochs)
        ordered = {k: params[k] for k in SEARCH_SPACE[model_name]["base"]}
        row = {"model": model_name, "stage": stage_label, "config_id": config_id(params),
               "seed": seed, "fold": split.name, **ordered, **metrics}
        append_result(model_name, row)
        changed = {k: v for k, v in params.items() if v != SEARCH_SPACE[model_name]["base"].get(k)}
        print(f"  ({n}/{len(jobs)}) {split.name} seed={seed} {changed or 'base'} "
              f"→ RMSE {metrics['rmse']:.3f}  MAE {metrics['mae']:.3f}  "
              f"peakMAE {metrics['peak_mae']:.3f}  epoch {metrics['best_epoch']}  "
              f"{metrics['train_seconds']}s")


# =========================================================
# 출력
# =========================================================

SHOW_COLS = ["rmse", "rmse_std", "mae", "r2", "peak_mae", "peak_recall", "peak_f1",
             "n_runs", "train_seconds"]


def print_table(agg, title, top=10):
    print(f"\n=== {title} ===")
    if agg.empty:
        print("(완료된 설정 없음)")
        return
    view = agg.head(top).copy()
    view.insert(0, "params", view["config_id"])
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(view[["params"] + SHOW_COLS].round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(SEARCH_SPACE))
    parser.add_argument("--stage", help="1, 2, 3, 4 또는 confirm")
    parser.add_argument("--summary", action="store_true", help="결과 요약만 출력")
    parser.add_argument("--data", help="전처리 CSV 경로 (기본: data/processed/okm_cleaned.csv)")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--folds", type=int, default=len(dp.FOLD_VAL_RANGES),
                        help="앞에서부터 사용할 fold 수 (동작 확인용)")
    parser.add_argument("--results-dir", help="결과 CSV 폴더 (기본: results/grid_search)")
    args = parser.parse_args()

    global RESULTS_DIR
    if args.results_dir:
        RESULTS_DIR = Path(args.results_dir)

    df = dp.load_data(args.data)
    folds = dp.get_folds(df)[:args.folds]
    n_folds = len(folds)
    model_name = args.model
    n_stages = len(SEARCH_SPACE[model_name]["stages"])

    if args.summary:
        results = load_results(model_name)
        if results.empty:
            print("결과가 없습니다.")
            return
        for stage in range(1, n_stages + 1):
            try:
                ids = [config_id(p) for p in stage_configs(model_name, stage, results, n_folds)]
            except RuntimeError:
                break
            print_table(aggregate(results, ids, n_folds, [DEFAULT_SEED]), f"Stage {stage}")
        ids = all_stage_ids(model_name, results, n_folds)
        print_table(aggregate(results, ids, n_folds, CONFIRM_SEEDS), "Confirm (seed 3개 평균)")
        return

    if args.stage is None:
        parser.error("--stage 또는 --summary 를 지정하세요.")

    if args.stage == "confirm":
        results = load_results(model_name)
        ids = all_stage_ids(model_name, results, n_folds)
        agg = aggregate(results, ids, n_folds, [DEFAULT_SEED])
        top = [json.loads(c) for c in agg["config_id"].head(CONFIRM_TOP_K)]
        run_configs(model_name, "confirm", top, CONFIRM_SEEDS, df, folds, args.max_epochs)
        results = load_results(model_name)
        agg = aggregate(results, [config_id(p) for p in top], n_folds, CONFIRM_SEEDS)
        print_table(agg, "Confirm 결과 (seed 3개 × fold 평균)")
        params, _, _ = final_choice(model_name, results, n_folds)
        print(f"\n최종 선택: {config_id(params)}")
        return

    stage = int(args.stage)
    if not 1 <= stage <= n_stages:
        parser.error(f"--stage 는 1~{n_stages} 또는 confirm 입니다.")

    results = load_results(model_name)
    configs = stage_configs(model_name, stage, results, n_folds)
    run_configs(model_name, str(stage), configs, [DEFAULT_SEED], df, folds, args.max_epochs)

    results = load_results(model_name)
    agg = aggregate(results, [config_id(p) for p in configs], n_folds, [DEFAULT_SEED])
    print_table(agg, f"Stage {stage} 결과 (fold 평균)")
    print(f"\n선택: {select_best(agg)['config_id']}")


if __name__ == "__main__":
    main()
