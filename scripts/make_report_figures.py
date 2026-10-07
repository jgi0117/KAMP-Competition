"""Create publication-ready charts directly from the held-out result files."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from data import PEAK_KW, load_evidence  # noqa: E402


OUT = ROOT / "report" / "figures"
COLORS = {"lstm": "#64748b", "tcn": "#64748b", "ensemble": "#0d9488",
          "xgboost": "#64748b", "lightgbm": "#64748b"}
LABELS = {"lstm": "LSTM", "tcn": "TCN", "ensemble": "LSTM + TCN",
          "xgboost": "XGBoost", "lightgbm": "LightGBM"}


def setup():
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)


def model_comparison(comparison):
    rows = comparison.sort_values("rmse", ascending=False)
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for ax, key, title, fmt in [(axes[0], "rmse", "전력 예측 오차 · RMSE (kW)", "{:.2f}"),
                                 (axes[1], "alert_f1", "177 kW 피크 경보 · F1", "{:.3f}")]:
        values = rows[key].to_numpy()
        ax.barh(y, values, color=[COLORS[m] for m in rows.model], height=.64)
        ax.set_yticks(y, [LABELS[m] for m in rows.model])
        ax.set_title(title, loc="left", weight="bold", pad=12)
        ax.set_xlim(0, max(values) * 1.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e2e8f0", zorder=0)
        ax.set_axisbelow(True)
        for yi, value in zip(y, values):
            ax.text(value + max(values) * .025, yi, fmt.format(value), va="center", weight="bold")
    fig.suptitle("동일 Test 703시간 · 기본 9개 변수 · seed 42", fontsize=16, weight="bold")
    fig.savefig(OUT / "01_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def peak_timeline(test):
    # Show the seven-day interval containing the most actual peaks.
    peaks_by_day = test.groupby(test.datetime.dt.date).actual_peak.sum()
    best = peaks_by_day.rolling(7, min_periods=1).sum().idxmax()
    end = np.datetime64(best) + np.timedelta64(1, "D")
    start = end - np.timedelta64(7, "D")
    view = test.loc[(test.datetime >= start) & (test.datetime < end)]
    fig, ax = plt.subplots(figsize=(11.2, 4.3), constrained_layout=True)
    ax.plot(view.datetime, view.actual, label="실측", color="#172554", lw=1.8)
    ax.plot(view.datetime, view.predicted, label="앙상블 예측", color="#0d9488", lw=1.8)
    alarms = view.loc[view.alert]
    misses = view.loc[view.outcome.eq("FN")]
    ax.scatter(alarms.datetime, alarms.actual, label="경보", color="#ef4444", s=20, zorder=4)
    ax.scatter(misses.datetime, misses.actual, label="미탐지", marker="x", color="#f59e0b", s=75, zorder=5)
    ax.axhline(PEAK_KW, color="#ef4444", ls="--", lw=1, label="피크 기준 177 kW")
    ax.set_title("피크가 집중된 7일: 실측·예측·경보", loc="left", fontsize=15, weight="bold")
    ax.set_ylabel("전력 (kW)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.02), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.savefig(OUT / "02_peak_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_profile():
    data = pd.read_csv(ROOT / "data/okm_cleaned_2021.csv", encoding="utf-8-sig")
    data["weekend"] = pd.to_datetime(data["날짜"]).dt.dayofweek >= 5
    means = data.groupby(["시간", "weekend"])["전력_평균_실수"].mean().unstack()
    fig, ax = plt.subplots(figsize=(11.2, 3.7), constrained_layout=True)
    ax.plot(means.index, means[False], label="평일", color="#0d9488", lw=2.6)
    ax.plot(means.index, means[True], label="주말", color="#64748b", lw=2.6)
    ax.axhline(PEAK_KW, color="#ef4444", ls="--", lw=1, label="피크 기준 177 kW")
    ax.set_title("시간별 평균 전력: 조업 일정에 따른 패턴", loc="left", fontsize=15, weight="bold")
    ax.set_xlabel("시간"); ax.set_ylabel("전력 (kW)")
    ax.set_xticks(range(0, 24, 2)); ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, ncol=3)
    fig.savefig(OUT / "04_data_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def grid_search_chart():
    rows = pd.read_csv(ROOT / "report/evidence/grid_search_summary.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    left, right = axes
    for model, color in [("lstm", "#0d9488"), ("tcn", "#2563eb")]:
        stage = rows.loc[rows.model.eq(model) & rows.stage.isin(["1", "2", "3", "4"])]
        left.plot(stage.stage.astype(int), stage.validation_rmse, marker="o", lw=2.4,
                  color=color, label=model.upper())
        for x, y in zip(stage.stage.astype(int), stage.validation_rmse):
            left.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points",
                          ha="center", fontsize=9)
    left.set_xticks([1, 2, 3, 4]); left.set_xlabel("순차 탐색 단계")
    left.set_ylabel("선택 후보의 검증 RMSE (kW)")
    left.set_title("LSTM·TCN 단계별 선택", loc="left", weight="bold")
    left.grid(alpha=.2); left.legend(frameon=False)
    for model, color in [("xgboost", "#2563eb"), ("lightgbm", "#0d9488")]:
        all_results = pd.read_csv(ROOT / f"results/base9_tree/grid_search/{model}_summary.csv")
        selected = rows.loc[rows.model.eq(model)].iloc[0]
        right.scatter(all_results.rmse, all_results.peak_mae, alpha=.45, s=19,
                      color=color, label=f"{model.upper()} 108개")
        right.scatter(selected.validation_rmse, selected.validation_peak_mae,
                      s=160, marker="*", color=color, edgecolors="black", linewidths=.7, zorder=5)
    right.set_xlabel("검증 RMSE (kW)"); right.set_ylabel("피크 구간 MAE (kW)")
    right.set_title("트리 전체 그리드와 최종 선택(★)", loc="left", weight="bold")
    right.grid(alpha=.2); right.legend(frameon=False, fontsize=8)
    fig.suptitle("3개 시간순 fold 평균 · seed 42", fontsize=15, weight="bold")
    fig.savefig(OUT / "05_grid_search.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_interaction_chart():
    importance = pd.read_csv(ROOT / "report/evidence/grouped_permutation_importance.csv", encoding="utf-8-sig")
    interaction = pd.read_csv(ROOT / "report/evidence/production_time_interaction.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    left, right = axes
    selected = importance.loc[importance.model.eq("xgboost")].sort_values("rmse_increase_mean", ascending=False).head(5)
    left.barh(selected.feature.iloc[::-1], selected.rmse_increase_mean.iloc[::-1], color="#0d9488")
    left.set_xlabel("묶음 순열 후 RMSE 증가 (kW)")
    left.set_title("XGBoost 입력 영향: 168시간 묶음 순열", loc="left", weight="bold")
    left.spines[["top", "right"]].set_visible(False); left.grid(axis="x", alpha=.2)
    labels = ["그 외·낮은 생산", "그 외·높은 생산", "오전·낮은 생산", "오전·높은 생산"]
    rates = []
    for morning, high in [(False, False), (False, True), (True, False), (True, True)]:
        row = interaction.loc[interaction.morning.eq(morning) & interaction.high_production.eq(high)].iloc[0]
        rates.append(row.peak_rate * 100)
    bars = right.barh(labels[::-1], rates[::-1], color=["#0d9488", "#64748b", "#64748b", "#64748b"])
    for bar, rate in zip(bars, rates[::-1]):
        right.text(rate + .6, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center")
    right.set_xlim(0, max(rates) * 1.3); right.set_xlabel("Test 피크 발생률 (%)")
    right.set_title("직전 생산량 × 오전 07~11시", loc="left", weight="bold")
    right.spines[["top", "right"]].set_visible(False); right.grid(axis="x", alpha=.2)
    fig.savefig(OUT / "06_feature_interaction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def error_conditions_chart():
    recent = pd.read_csv(ROOT / "report/evidence/test_error_by_hour.csv", encoding="utf-8-sig")
    diagnostic = pd.read_csv(ROOT / "report/evidence/combined_error_by_hour.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), constrained_layout=True)
    left, right = axes
    left.bar(recent.hour, recent.false_positive, color="#ef4444", alpha=.8)
    left.set_title("최종 Test 오경보 85건", loc="left", weight="bold")
    left.set_ylabel("오경보 건수"); left.set_xlabel("시간대")
    left.set_xticks(range(0, 24, 3)); left.grid(axis="y", alpha=.2)
    top = diagnostic.sort_values("false_negative", ascending=False).head(6).sort_values("hour")
    right.bar(top.hour.astype(str) + "시", top.false_negative, color="#f59e0b")
    right.set_title("검증+Test 미탐지 집중 시간", loc="left", weight="bold")
    right.set_ylabel("미탐지 건수"); right.set_xlabel("시간대")
    right.grid(axis="y", alpha=.2)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT / "07_error_conditions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _table_figure(rows, headers, title, filename, highlight=None):
    fig, ax = plt.subplots(figsize=(11.2, 3.5), constrained_layout=True)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=17)
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                     bbox=[0, .05, 1, .85])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0")
        cell.set_linewidth(.8)
        if row == 0:
            cell.set_facecolor("#172554")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row == highlight:
            cell.set_facecolor("#ccfbf1")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f8fafc" if row % 2 == 0 else "white")
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def grid_selection_table():
    grid = pd.read_csv(ROOT / "report/evidence/grid_search_summary.csv", encoding="utf-8-sig")
    rows = []
    for model in ("lstm", "tcn", "xgboost", "lightgbm"):
        selected = grid.loc[grid.model.eq(model)].iloc[-1]
        if model in ("lstm", "tcn"):
            all_folds = pd.read_csv(ROOT / f"neural/results/grid_search/{model}.csv")
            candidates, folds = all_folds.config_id.nunique(), len(all_folds)
        else:
            candidates, folds = int(selected.completed_candidates), int(selected.fold_results)
        params = json.loads(selected.selected_params)
        if model == "lstm":
            setting = f"{params['lookback']}h · {params['hidden_units']}유닛 · {params['num_layers']}층"
        elif model == "tcn":
            setting = f"{params['lookback']}h · {params['filters']}필터 · 커널 {params['kernel_size']}"
        else:
            setting = f"트리 {params['n_estimators']} · 학습률 {params['learning_rate']}"
        rows.append([LABELS[model], str(candidates), str(folds),
                     f"{selected.validation_rmse:.2f}",
                     f"{selected.validation_peak_mae:.2f}", setting])
    _table_figure(rows, ["모델", "고유 후보", "fold 결과", "검증 RMSE", "피크 MAE", "선택 설정"],
                  "3개 시간순 fold의 전체 탐색과 최종 선택", "12_grid_selection_table.png")


def test_metrics_table(comparison):
    rows = []
    for row in comparison.itertuples():
        rows.append([LABELS[row.model], f"{row.rmse:.2f}", f"{row.mae:.2f}",
                     f"{row.alert_f1:.3f}", f"{row.alert_recall:.1%}",
                     str(row.fn), str(row.fp)])
    _table_figure(rows, ["모델", "RMSE kW", "MAE kW", "피크 F1", "재현율", "미탐지", "오경보"],
                  "동일 Test 703시간 · 177 kW 피크 경보 성능", "13_test_metrics_table.png",
                  highlight=3)


def main():
    setup()
    comparison, test, _, _ = load_evidence()
    model_comparison(comparison)
    peak_timeline(test)
    data_profile()
    grid_search_chart()
    feature_interaction_chart()
    error_conditions_chart()
    grid_selection_table()
    test_metrics_table(comparison)
    print("\n".join(str(path) for path in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
