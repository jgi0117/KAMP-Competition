"""Recreate the KJH notebook's data diagnostics for the submission report."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report/figures"
EVIDENCE = ROOT / "report/evidence"
POWER = "전력_평균_실수"
PRODUCTION = "생산량"
WORKERS = "공장인원"
DATE = "날짜"
HOUR = "시간"
RESTORED = "시간_복원여부"
SHUTDOWN = "공장_셧다운_여부"
WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ROOT / "data/okm_augumented_2021.csv", encoding="utf-8-sig")
    clean = pd.read_csv(ROOT / "data/okm_cleaned_2021.csv", encoding="utf-8-sig")
    clean["datetime"] = pd.to_datetime(clean["날짜"]) + pd.to_timedelta(clean["시간"], unit="h")
    clean["weekday"] = clean.datetime.dt.dayofweek
    if len(raw) != len(clean) or clean.datetime.duplicated().any():
        raise ValueError("Raw and cleaned hourly records do not align")
    return raw, clean


def setup() -> None:
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    FIGURES.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)


def overview(clean: pd.DataFrame) -> None:
    daily = clean.groupby(clean.datetime.dt.date).agg(
        power=(POWER, "mean"), production=(PRODUCTION, "mean"))
    dates = pd.to_datetime(daily.index)
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 4.7), sharex=True, constrained_layout=True)
    for ax, column, color, label in zip(axes, ("power", "production"),
                                         ("#0d9488", "#2563eb"), ("평균 전력 (kW)", "평균 생산량")):
        ax.plot(dates, daily[column], color=color, linewidth=1.3)
        ax.axvspan(pd.Timestamp("2021-08-28"), pd.Timestamp("2021-08-30"),
                   color="#f59e0b", alpha=.17, label="8월 공장 셧다운")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title("2021년 1~9월 전력과 생산량의 동시 변화", loc="left", weight="bold")
    axes[-1].set_xlabel("날짜")
    fig.savefig(FIGURES / "08_eda_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def quality(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    names = [WORKERS, "풍속", "강수량"]
    before = [int(raw[name].isna().sum()) for name in names]
    after = [int(clean[name].isna().sum()) for name in names]
    x = np.arange(len(names))
    axes[0].bar(x - .19, before, width=.38, color="#ef4444", label="원본")
    axes[0].bar(x + .19, after, width=.38, color="#0d9488", label="정제")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("결측 건수")
    axes[0].set_title("원본 결측과 정제 결과", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=.2)
    for i, count in enumerate(before):
        axes[0].text(i - .19, count + .35, str(count), ha="center", fontsize=9)

    shutdown = clean.loc[clean["공장_셧다운_여부"]]
    start, end = shutdown.datetime.min() - pd.Timedelta(hours=5), shutdown.datetime.max() + pd.Timedelta(hours=5)
    view = clean.loc[clean.datetime.between(start, end)]
    axes[1].plot(view.datetime, view[POWER], marker="o", ms=3, color="#0d9488", label="전력")
    axes[1].axvspan(shutdown.datetime.min(), shutdown.datetime.max(), color="#f59e0b", alpha=.2,
                    label=f"셧다운 {len(shutdown)}시간")
    axes[1].set_title("8월 28~29일 셧다운 구간", loc="left", weight="bold")
    axes[1].set_ylabel("전력 (kW)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.savefig(FIGURES / "09_eda_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def heatmap(clean: pd.DataFrame) -> None:
    matrix = clean.pivot_table(index="weekday", columns="시간", values=POWER, aggfunc="mean")
    matrix = matrix.reindex(index=range(7), columns=range(24))
    fig, ax = plt.subplots(figsize=(11.2, 4.0), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_yticks(range(7), WEEKDAYS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시간")
    ax.set_title("요일 × 시간 평균 전력: 평일 조업과 주말 부하", loc="left", weight="bold")
    fig.colorbar(image, ax=ax, label="평균 전력 (kW)", fraction=.027, pad=.02)
    fig.savefig(FIGURES / "10_eda_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def operating_modes(clean: pd.DataFrame) -> None:
    active = clean.loc[~clean["공장_셧다운_여부"]]
    idle = active.loc[active[PRODUCTION].eq(0), POWER]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    axes[0].scatter(active[PRODUCTION], active[POWER], s=8, alpha=.18, color="#0d9488")
    axes[0].set_xlabel("생산량"); axes[0].set_ylabel("전력 (kW)")
    axes[0].set_title("생산 강도와 전력 부하", loc="left", weight="bold")
    axes[1].hist(active[POWER], bins=np.arange(0, 226, 5), color="#2563eb", alpha=.8)
    axes[1].axvline(idle.mean(), color="#ef4444", ls="--", lw=2,
                    label=f"무생산 평균 {idle.mean():.1f} kW")
    axes[1].set_xlabel("전력 (kW)"); axes[1].set_ylabel("시간 수")
    axes[1].set_title("대기·가동 부하의 분포", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
    fig.savefig(FIGURES / "11_eda_operating_modes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build() -> dict:
    setup()
    raw, clean = load_data()
    idle = clean.loc[clean[PRODUCTION].eq(0) & ~clean["공장_셧다운_여부"], POWER]
    weekday_11 = clean.loc[clean.weekday.lt(5) & clean["시간"].eq(11), POWER]
    weekday_12 = clean.loc[clean.weekday.lt(5) & clean["시간"].eq(12), POWER]
    summary = {
        "raw_rows": int(len(raw)), "clean_rows": int(len(clean)),
        "raw_missing": {name: int(raw[name].isna().sum()) for name in (WORKERS, "풍속", "강수량")},
        "clean_missing": int(clean.isna().sum().sum()),
        "restored_hours": int(clean["시간_복원여부"].sum()),
        "shutdown_hours": int(clean["공장_셧다운_여부"].sum()),
        "idle_hours": int(len(idle)), "idle_mean_kw": float(idle.mean()),
        "weekday_11_kw": float(weekday_11.mean()),
        "weekday_12_kw": float(weekday_12.mean()),
        "lag_correlations": {str(lag): float(clean[POWER].corr(clean[POWER].shift(lag)))
                             for lag in (1, 24, 168)},
    }
    (EVIDENCE / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    overview(clean)
    quality(raw, clean)
    heatmap(clean)
    operating_modes(clean)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    build()
