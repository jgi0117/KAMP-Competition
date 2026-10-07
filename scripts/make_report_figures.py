"""Create publication-ready charts directly from the held-out result files."""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


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


def main():
    setup()
    comparison, test, _, _ = load_evidence()
    model_comparison(comparison)
    peak_timeline(test)
    print("\n".join(str(path) for path in sorted(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
