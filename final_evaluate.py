# -*- coding: utf-8 -*-
"""최종 학습 + Test 평가 + Weighted Ensemble (구현가이드 19~22·26장).

1. 모델별 최종 설정을 그리드 서치 결과(confirm 우선)에서 가져온다.
2. Test 이전 데이터로 다시 학습한다 (1/1~8/1 학습, 8/2~8/15 EarlyStopping).
   seed 17·42·73 으로 3번 학습해 seed 에 따른 변동을 함께 보고한다.
3. Test 구간(8/16~9/14, 셧다운 제외)에서 한 번만 평가한다.
4. IEEE EECR 2023 방식 Weighted Ensemble:
   가중치 = 1 / (검증 MSE), 합이 1이 되도록 정규화.
   검증 MSE 는 그리드 서치 fold 검증 결과(seed·fold 평균)를 쓴다. Test 는 가중치 계산에 쓰지 않는다.
   같은 seed 의 LSTM·TCN 예측끼리 결합한다.

사용 예:
    python final_evaluate.py --data <csv> --grid-dir results/grid_search --out-dir results/final
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import grid_search as gs
from src import data_pipeline as dp
from src.evaluate import evaluate_regression

MODELS = ["lstm", "tcn", "tcn_lstm"]
SEEDS = gs.CONFIRM_SEEDS
LABELS = {
    "lstm": "LSTM",
    "tcn": "TCN",
    "ensemble": "Weighted Ensemble (LSTM+TCN)",
    "tcn_lstm": "TCN-LSTM",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="전처리 CSV 경로")
    parser.add_argument("--grid-dir", default="results/grid_search", help="그리드 서치 결과 폴더")
    parser.add_argument("--out-dir", default="results/final", help="최종 결과 저장 폴더")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--max-epochs", type=int, default=gs.MAX_EPOCHS)
    parser.add_argument("--folds", type=int, default=len(dp.FOLD_VAL_RANGES),
                        help="그리드 서치에 사용한 fold 수 (기본 3)")
    args = parser.parse_args()

    gs.RESULTS_DIR = Path(args.grid_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    df = dp.load_data(args.data)
    split = dp.get_final_split(df)

    # ---------- 1. 모델별 최종 설정 ----------
    choices = {}
    for name in args.models:
        params, row, basis = gs.final_choice(name, gs.load_results(name), args.folds)
        if params is None:
            print(f"[{name}] 그리드 서치 결과가 없어 건너뜁니다.")
            continue
        choices[name] = {"params": params, "val_mse": row["mse"], "val_rmse": row["rmse"],
                         "basis": basis}
        print(f"[{name}] 최종 설정 ({basis}): {gs.config_id(params)}")
        print(f"        검증 RMSE {row['rmse']:.3f} kW (fold 평균)")

    if not choices:
        raise SystemExit("평가할 모델이 없습니다. 그리드 서치를 먼저 실행하세요.")

    # ---------- 2~3. 최종 학습 + Test 예측 ----------
    preds = {}          # (모델, seed) → kW 예측
    actual = idx_eval = None
    peak_threshold = None
    train_log = []

    for name, choice in choices.items():
        params = choice["params"]
        data = dp.prepare_split(df, split, params["lookback"])
        if actual is None:
            actual = data["y_scaler"].inverse_transform(data["y_eval"]).ravel()
            idx_eval = data["idx_eval"]
            peak_threshold = data["peak_threshold"]
        # lookback 이 달라도 Test Target 시점은 같아야 비교가 공정하다
        assert np.array_equal(idx_eval, data["idx_eval"]), "모델 간 Test 시점이 다릅니다."

        for seed in SEEDS:
            model, info = gs.train_model(name, params, seed, data, args.max_epochs)
            pred, _ = gs.predict_kw(model, data)
            preds[(name, seed)] = pred
            model.save(out_dir / "models" / f"{name}_seed{seed}.keras")
            train_log.append({"model": name, "seed": seed, **info})
            print(f"  [{name}] seed={seed} 학습 완료 (best epoch {info['best_epoch']}, "
                  f"{info['train_seconds']}s)")

    # ---------- 4. Weighted Ensemble ----------
    weights = None
    if "lstm" in choices and "tcn" in choices:
        inv = {m: 1.0 / choices[m]["val_mse"] for m in ("lstm", "tcn")}
        total = sum(inv.values())
        weights = {m: inv[m] / total for m in inv}
        print(f"\nWeighted Ensemble 가중치 (1/검증 MSE): "
              f"LSTM {weights['lstm']:.3f}, TCN {weights['tcn']:.3f}")
        for seed in SEEDS:
            preds[("ensemble", seed)] = (weights["lstm"] * preds[("lstm", seed)]
                                         + weights["tcn"] * preds[("tcn", seed)])
    else:
        print("\nLSTM 과 TCN 이 모두 있어야 Weighted Ensemble 을 만들 수 있어 건너뜁니다.")

    # ---------- 평가 ----------
    rows = []
    for (name, seed), pred in preds.items():
        rows.append({"model": name, "seed": seed,
                     **evaluate_regression(actual, pred, peak_threshold)})
    per_seed = pd.DataFrame(rows)

    order = [m for m in ["lstm", "tcn", "ensemble", "tcn_lstm"] if m in per_seed["model"].unique()]
    metric_cols = ["rmse", "mae", "r2", "peak_mae", "peak_recall", "peak_f1"]
    summary = per_seed.groupby("model")[metric_cols].agg(["mean", "std"]).loc[order]
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.insert(1, "label", summary["model"].map(LABELS))
    summary.insert(2, "config", summary["model"].map(
        lambda m: gs.config_id(choices[m]["params"]) if m in choices
        else f"LSTM {weights['lstm']:.3f} + TCN {weights['tcn']:.3f}"))

    # ---------- 저장 ----------
    pred_df = pd.DataFrame({"datetime": df["datetime"].iloc[idx_eval].to_numpy(),
                            "actual": actual})
    for (name, seed), pred in preds.items():
        pred_df[f"{name}_seed{seed}"] = pred
    for name in order:
        pred_df[f"{name}_mean"] = pred_df[[f"{name}_seed{s}" for s in SEEDS]].mean(axis=1)

    pred_df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(out_dir / "test_metrics_per_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "test_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(train_log).to_csv(out_dir / "train_log.csv", index=False, encoding="utf-8-sig")
    if weights:
        pd.DataFrame([{"model": m, "val_mse": choices[m]["val_mse"], "weight": w}
                      for m, w in weights.items()]).to_csv(
            out_dir / "ensemble_weights.csv", index=False, encoding="utf-8-sig")

    # ---------- 출력 ----------
    print(f"\n=== Test 결과 (8/16~9/14, 셧다운 제외 {len(actual)}시간, "
          f"피크 기준 {peak_threshold:.1f} kW, seed {len(SEEDS)}개 평균 ± 표준편차) ===")
    table = pd.DataFrame({"Model": summary["label"]})
    for col, title in [("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R²"), ("peak_mae", "Peak MAE"),
                       ("peak_recall", "Peak Recall"), ("peak_f1", "Peak F1")]:
        digits = 3 if col in ("r2", "peak_recall", "peak_f1") else 2
        table[title] = [f"{m:.{digits}f} ± {s:.{digits}f}"
                        for m, s in zip(summary[f"{col}_mean"], summary[f"{col}_std"])]
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False))
    print(f"\n저장 위치: {out_dir}")


if __name__ == "__main__":
    main()
