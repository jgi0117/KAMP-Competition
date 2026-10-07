# -*- coding: utf-8 -*-
"""Train and evaluate seed-42 LSTM, TCN, and their weighted ensemble.

Uses the cleaned CSV and nine base inputs. Three time-ordered validation
folds select settings, residual sigma, ensemble weights, and alert cutoffs.
The held-out Test period is evaluated once. Baselines are also calculated
for context; saved imported results include only the requested three models.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import grid_search as gs
from src import data_pipeline as dp
from src.evaluate import (
    choose_alert_cutoff,
    evaluate_alerts,
    evaluate_regression,
    peak_probability,
)

MODELS = ["lstm", "tcn"]
BASELINES = {"naive_lag1": 1, "naive_lag168": 168}
ORDER = ["naive_lag1", "naive_lag168", "lstm", "tcn", "ensemble"]
LABELS = {
    "naive_lag1": "베이스라인: 직전 시간 그대로",
    "naive_lag168": "베이스라인: 지난주 같은 시간",
    "lstm": "LSTM",
    "tcn": "TCN",
    "ensemble": "Weighted Ensemble (LSTM+TCN)",
}


def predict_split(name, params, seed, df, split, max_epochs):
    """한 Split 에서 학습하고 평가 구간을 예측한다. (모델, 예측 kW, 실제 kW, Target 행 번호, 학습 정보)"""
    data = dp.prepare_split(df, split, params["lookback"], params.get("feature_set", "base"))
    model, info = gs.train_model(name, params, seed, data, max_epochs)
    pred, actual = gs.predict_kw(model, data)
    return model, pred, actual, data["idx_eval"], info


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="전처리 CSV 경로")
    parser.add_argument("--grid-dir", default=str(Path(__file__).resolve().parent / "results" / "grid_search"), help="그리드 서치 결과 폴더")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results" / "reproduced"), help="최종 결과 저장 폴더")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--max-epochs", type=int, default=gs.MAX_EPOCHS)
    parser.add_argument("--folds", type=int, default=len(dp.FOLD_VAL_RANGES),
                        help="검증 fold 수 (기본 3)")
    parser.add_argument("--space", choices=list(gs.SPACES), default="v1",
                        help="그리드 서치 탐색 계획 (v1 만 지원)")
    parser.add_argument("--selection", choices=["seed42", "confirm"], default="seed42",
                        help="최종 설정 선택 기준 (기본 seed42: 팀 공통 기준)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[gs.DEFAULT_SEED],
                        help="학습 seed (기본 42 한 번)")
    args = parser.parse_args()
    seeds = args.seeds
    thr = dp.PEAK_THRESHOLD_KW

    gs.use_space(args.space)
    gs.RESULTS_DIR = Path(args.grid_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    df = dp.load_data(args.data)
    folds = dp.get_folds(df)[:args.folds]
    final_split = dp.get_final_split(df)
    power = df[dp.TARGET_COLUMN]

    # ---------- 1. 모델별 최종 설정 ----------
    choices = {}
    for name in args.models:
        params, row, basis = gs.final_choice(name, gs.load_results(name), args.folds, args.selection)
        if params is None:
            print(f"[{name}] 그리드 서치 결과가 없어 건너뜁니다.")
            continue
        choices[name] = params
        print(f"[{name}] 최종 설정 ({basis}): {gs.config_id(params)}")
    if not choices:
        raise SystemExit("평가할 모델이 없습니다. 그리드 서치를 먼저 실행하세요.")

    # ---------- 2~3. 검증 예측 + Test 예측 ----------
    val, test = {}, {}          # (이름, seed) → 예측 kW
    val_idx = val_fold = test_idx = None
    train_log = []
    for name, params in choices.items():
        for seed in seeds:
            parts, idxs, fold_names = [], [], []
            for split in folds:
                _, pred, _, idx, info = predict_split(name, params, seed, df, split, args.max_epochs)
                parts.append(pred); idxs.append(idx); fold_names += [split.name] * len(idx)
                train_log.append({"model": name, "seed": seed, "split": split.name, **info})
            idx = np.concatenate(idxs)
            if val_idx is None:
                val_idx, val_fold = idx, np.array(fold_names)
            assert np.array_equal(idx, val_idx), "모델 간 검증 시점이 다릅니다."
            val[(name, seed)] = np.concatenate(parts)

            model, pred, _, idx, info = predict_split(name, params, seed, df, final_split, args.max_epochs)
            if test_idx is None:
                test_idx = idx
            assert np.array_equal(idx, test_idx), "모델 간 Test 시점이 다릅니다."
            test[(name, seed)] = pred
            model.save(out_dir / "models" / f"{name}_seed{seed}.keras")
            train_log.append({"model": name, "seed": seed, "split": "final", **info})
            print(f"  [{name}] seed={seed} 검증 fold {len(folds)}개 + Test 학습 완료")

    y_val = power.to_numpy()[val_idx]
    y_test = power.to_numpy()[test_idx]

    # ---------- 4. Weighted Ensemble ----------
    weights = {}
    if "lstm" in choices and "tcn" in choices:
        for seed in seeds:
            inv = {m: 1.0 / np.mean((y_val - val[(m, seed)]) ** 2) for m in ("lstm", "tcn")}
            w = {m: v / sum(inv.values()) for m, v in inv.items()}
            weights[seed] = w
            for store in (val, test):
                store[("ensemble", seed)] = w["lstm"] * store[("lstm", seed)] + w["tcn"] * store[("tcn", seed)]
        w0 = weights[seeds[0]]
        print(f"\nWeighted Ensemble 가중치 (1/검증 MSE, seed {seeds[0]}): "
              f"LSTM {w0['lstm']:.3f}, TCN {w0['tcn']:.3f}")
    else:
        print("\nLSTM 과 TCN 이 모두 있어야 Weighted Ensemble 을 만들 수 있어 건너뜁니다.")

    # ---------- 5. 베이스라인 (seed 와 무관하지만 표 형식을 맞추려고 seed 마다 넣는다) ----------
    for name, lag in BASELINES.items():
        shifted = power.shift(lag).to_numpy()
        for seed in seeds:
            val[(name, seed)] = shifted[val_idx]
            test[(name, seed)] = shifted[test_idx]

    # ---------- 6. 평가 ----------
    rows, settings = [], []
    probs = {}
    for (name, seed), pv in val.items():
        resid = y_val - pv
        sigma = float(np.std(resid, ddof=1))
        cutoff = choose_alert_cutoff(y_val, peak_probability(pv, sigma, thr), thr)
        pt = peak_probability(test[(name, seed)], sigma, thr)
        probs[(name, seed)] = pt
        val_reg = evaluate_regression(y_val, pv, thr)
        settings.append({"model": name, "seed": seed, "sigma": sigma, "alert_cutoff": cutoff,
                         "val_rmse": val_reg["rmse"], "val_mse": val_reg["mse"], "val_r2": val_reg["r2"],
                         "val_alert_f1": evaluate_alerts(y_val, peak_probability(pv, sigma, thr), thr, cutoff)["alert_f1"]})
        rows.append({"model": name, "seed": seed,
                     **evaluate_regression(y_test, test[(name, seed)], thr),
                     **evaluate_alerts(y_test, pt, thr, cutoff)})
    per_seed = pd.DataFrame(rows)
    order = [m for m in ORDER if m in per_seed["model"].unique()]
    cols = ["mse", "rmse", "r2", "mae", "alert_f1", "alert_recall", "alert_precision", "brier", "pr_auc",
            "tp", "fn", "fp", "alert_cutoff"]
    summary = per_seed.groupby("model")[cols].agg(["mean", "std"]).loc[order]
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, "label", summary["model"].map(LABELS))
    summary.insert(2, "config", summary["model"].map(
        lambda m: gs.config_id(choices[m]) if m in choices
        else ("LSTM·TCN 가중 평균" if m == "ensemble" else f"lag {BASELINES[m]}")))

    # ---------- 저장 ----------
    dt = df["datetime"].to_numpy()
    val_df = pd.DataFrame({"datetime": dt[val_idx], "fold": val_fold, "actual": y_val})
    test_df = pd.DataFrame({"datetime": dt[test_idx], "actual": y_test, "is_peak": y_test >= thr})
    for (name, seed), pv in val.items():
        val_df[f"pred_{name}_seed{seed}"] = pv
    for (name, seed), pt in test.items():
        test_df[f"pred_{name}_seed{seed}"] = pt
        test_df[f"prob_{name}_seed{seed}"] = probs[(name, seed)]
    val_df.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(out_dir / "test_metrics_per_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "test_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(settings).to_csv(out_dir / "alert_settings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_log).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    if weights:
        pd.DataFrame([{"seed": s, "model": m, "weight": v} for s, w in weights.items() for m, v in w.items()]
                     ).to_csv(out_dir / "ensemble_weights.csv", index=False, encoding="utf-8-sig")

    # ---------- 출력 ----------
    one = len(seeds) == 1
    print(f"\n=== Test 결과 (8/16~9/14, 셧다운 제외 {len(y_test)}시간, 이상 기준 {thr:.0f} kW, "
          f"이상 {int((y_test >= thr).sum())}시간, {'seed ' + str(seeds[0]) if one else f'seed {len(seeds)}개 평균'}) ===")
    table = pd.DataFrame({"Model": summary["label"]})
    for col, title, d in [("mse", "MSE", 1), ("rmse", "RMSE", 2), ("r2", "R²", 3), ("mae", "MAE", 2),
                          ("alert_f1", "F1", 3), ("alert_recall", "Recall", 3), ("alert_precision", "Precision", 3),
                          ("brier", "Brier", 4), ("pr_auc", "PR-AUC", 3), ("fn", "FN", 0), ("fp", "FP", 0),
                          ("alert_cutoff", "경보기준", 2)]:
        table[title] = [f"{m:.{d}f}" if one else f"{m:.{d}f}±{s:.{d}f}"
                        for m, s in zip(summary[f"{col}_mean"], summary[f"{col}_std"])]
    with pd.option_context("display.width", 250):
        print(table.to_string(index=False))
    print("\nF1·Recall·Precision·FN·FP: 이상 확률 ≥ 경보기준이면 경보. 경보기준은 검증 구간 F1 최대값.")
    print(f"저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
