"""Recalculate tables cited in the competition report from saved experiment outputs."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/report/evidence"
sys.path.insert(0, str(ROOT))
from neural import grid_search as deep_search  # noqa: E402
from neural.core import data_pipeline as dp  # noqa: E402
from neural.core.evaluate import choose_alert_cutoff, evaluate_alerts, peak_probability  # noqa: E402


def baseline(actual, predicted, validation_actual, validation_predicted, threshold=177.0):
    """Choose the alarm cutoff on validation, then score the held-out Test."""
    sigma = float(np.std(validation_actual - validation_predicted, ddof=1))
    cutoff = choose_alert_cutoff(
        validation_actual,
        peak_probability(validation_predicted, sigma, threshold),
        threshold,
    )
    alerts = evaluate_alerts(
        actual, peak_probability(predicted, sigma, threshold), threshold, cutoff,
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "f1": float(alerts["alert_f1"]),
        "recall": float(alerts["alert_recall"]),
        "precision": float(alerts["alert_precision"]),
        "sigma": sigma,
        "alert_cutoff": float(cutoff),
    }


def diagnostics(clean, validation, test):
    raw = pd.read_csv(ROOT / "data/okm_augumented_2021.csv", encoding="utf-8-sig")
    clean_file = pd.read_csv(ROOT / "data/okm_cleaned_2021.csv", encoding="utf-8-sig")
    timeline = clean.set_index("datetime")
    actual = test.actual.to_numpy()
    validation_actual = validation.actual.to_numpy()
    baselines = {}
    for hours in (1, 168):
        prediction = timeline[dp.TARGET_COLUMN].shift(hours).reindex(test.datetime).to_numpy()
        validation_prediction = timeline[dp.TARGET_COLUMN].shift(hours).reindex(
            validation.datetime
        ).to_numpy()
        baselines[f"lag{hours}"] = baseline(
            actual, prediction, validation_actual, validation_prediction,
        )
    return {
        "raw_rows": len(raw), "clean_rows": len(clean_file),
        "raw_columns": len(raw.columns), "clean_columns": len(clean_file.columns),
        "start": str(clean.datetime.min()), "end": str(clean.datetime.max()),
        "duplicate_raw_rows": int(raw.duplicated().sum()),
        "duplicate_hours": int(clean.datetime.duplicated().sum()),
        "raw_missing": {k: int(v) for k, v in raw.isna().sum().items() if v},
        "clean_missing": int(clean_file.isna().sum().sum()),
        "restored_hours": int(clean["시간_복원여부"].sum()),
        "shutdown_hours": int(clean["공장_셧다운_여부"].sum()),
        "test_hours": len(test), "test_peaks": int((test.actual >= dp.PEAK_THRESHOLD_KW).sum()),
        "test_peak_rate": float((test.actual >= dp.PEAK_THRESHOLD_KW).mean()),
        "baselines": baselines,
    }


def grid_evidence():
    rows = []
    for model in ("lstm", "tcn"):
        result = pd.read_csv(ROOT / f"neural/results/grid_search/{model}.csv", encoding="utf-8-sig")
        for stage in range(1, 5):
            configs = deep_search.stage_configs(model, stage, result, 3)
            ids = [deep_search.config_id(config) for config in configs]
            ranked = deep_search.aggregate(result, ids, 3, [42])
            winner = deep_search.select_best(ranked)
            rows.append({"model": model, "stage": stage, "candidate_count": len(configs),
                         "completed_candidates": len(ranked), "fold_results": len(ranked) * 3,
                         "selected_params": winner.config_id,
                         "validation_rmse": float(winner.rmse),
                         "validation_peak_mae": float(winner.peak_mae)})
    for model in ("xgboost", "lightgbm"):
        ranked = pd.read_csv(ROOT / f"results/tree_models/grid_search/{model}_summary.csv")
        lowest = ranked.loc[ranked.rmse.idxmin()]
        finalists = ranked.loc[ranked.rmse <= lowest.rmse * 1.02]
        winner = finalists.sort_values(["peak_mae", "rmse"]).iloc[0]
        rows.append({"model": model, "stage": "full grid", "candidate_count": len(ranked),
                     "completed_candidates": len(ranked), "fold_results": len(ranked) * 3,
                     "selected_params": winner.params,
                     "validation_rmse": float(winner.rmse),
                     "validation_peak_mae": float(winner.peak_mae),
                     "lowest_rmse": float(lowest.rmse),
                     "rmse_2pct_candidates": len(finalists),
                     "selected_candidate": int(winner.candidate)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "grid_search_summary.csv", index=False, encoding="utf-8-sig")
    return table


def condition_analysis(clean, test):
    source = clean.set_index("datetime")
    joined = test.copy()
    joined["previous_production"] = source["생산량"].shift(1).reindex(test.datetime).to_numpy()
    joined["previous_temperature"] = source["기온"].shift(1).reindex(test.datetime).to_numpy()
    joined["hour"] = joined.datetime.dt.hour
    joined["peak"] = joined.actual >= dp.PEAK_THRESHOLD_KW
    joined["alert"] = joined.prob_ensemble_seed42 >= 0.2
    joined["false_positive"] = joined.alert & ~joined.peak
    joined["false_negative"] = ~joined.alert & joined.peak
    joined["morning"] = joined.hour.between(7, 11)
    joined["high_production"] = joined.previous_production > 1000
    joined["hot"] = joined.previous_temperature >= 28
    hourly = joined.groupby("hour").agg(hours=("peak", "size"), peaks=("peak", "sum"),
                                          false_positive=("false_positive", "sum"),
                                          false_negative=("false_negative", "sum"))
    hourly.to_csv(OUT / "test_error_by_hour.csv", encoding="utf-8-sig")
    interaction = joined.groupby(["morning", "high_production"]).agg(
        hours=("peak", "size"), peaks=("peak", "sum"), false_positive=("false_positive", "sum"))
    interaction["peak_rate"] = interaction.peaks / interaction.hours
    interaction.to_csv(OUT / "production_time_interaction.csv", encoding="utf-8-sig")
    hot = joined.groupby(["hot", "high_production"]).agg(hours=("peak", "size"), peaks=("peak", "sum"))
    hot["peak_rate"] = hot.peaks / hot.hours
    hot.to_csv(OUT / "production_temperature_interaction.csv", encoding="utf-8-sig")
    misses = joined.loc[joined.false_negative, ["datetime", "actual", "pred_ensemble_seed42",
                                                   "prob_ensemble_seed42", "previous_production",
                                                   "previous_temperature", "hour"]]
    misses.to_csv(OUT / "test_missed_peaks.csv", index=False, encoding="utf-8-sig")
    return joined, interaction, hot, hourly, misses


def combined_error_by_hour(test):
    """Summarize validation and Test errors with the saved alert rule."""
    validation = pd.read_csv(ROOT / "neural/results/final/val_predictions.csv", encoding="utf-8-sig")
    settings = pd.read_csv(ROOT / "neural/results/final/alert_settings.csv", encoding="utf-8-sig")
    ensemble = settings.set_index("model").loc["ensemble"]
    joined = pd.concat([validation, test], ignore_index=True)
    probability = peak_probability(joined.pred_ensemble_seed42.to_numpy(),
                                   float(ensemble.sigma), dp.PEAK_THRESHOLD_KW)
    actual_peak = joined.actual.to_numpy() >= dp.PEAK_THRESHOLD_KW
    alert = probability >= float(ensemble.alert_cutoff)
    joined["hour"] = pd.to_datetime(joined.datetime).dt.hour
    joined["false_negative"] = actual_peak & ~alert
    joined["false_positive"] = ~actual_peak & alert
    grouped = joined.groupby("hour", as_index=False).agg(
        hours=("actual", "size"), false_negative=("false_negative", "sum"),
        false_positive=("false_positive", "sum"))
    grouped.to_csv(OUT / "combined_error_by_hour.csv", index=False, encoding="utf-8-sig")
    return grouped


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clean = dp.load_data()
    test = pd.read_csv(ROOT / "neural/results/final/test_predictions.csv", encoding="utf-8-sig")
    test["datetime"] = pd.to_datetime(test.datetime)
    validation = pd.read_csv(ROOT / "neural/results/final/val_predictions.csv", encoding="utf-8-sig")
    validation["datetime"] = pd.to_datetime(validation.datetime)
    info = diagnostics(clean, validation, test)
    (OUT / "data_diagnostics.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    grid = grid_evidence()
    joined, interaction, hot, hourly, misses = condition_analysis(clean, test)
    combined_error_by_hour(test)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(grid.to_string(index=False))
    print("production × time\n", interaction.to_string())
    print("production × temperature\n", hot.to_string())
    print("missed peaks\n", misses.to_string(index=False))
    print("FP by hour\n", hourly.sort_values("false_positive", ascending=False).head(7).to_string())


if __name__ == "__main__":
    main()
