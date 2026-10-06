# -*- coding: utf-8 -*-
"""요구사항 3번: 영향요인 및 오류분석 (팀 공통 평가 기준, docs/modeling/팀_공통_평가기준.md).

입력: final_evaluate.py 결과 폴더 (val_predictions.csv, test_predictions.csv, alert_settings.csv,
      test_summary.csv, models/*.keras) + 정현님 피처 데이터 (okm_features2_2021.csv)

출력 (--out-dir):
  1. 영향 변수 — permutation importance (Test 구간, 변수 한 개를 시간 순서와 무관하게 섞었을 때 RMSE 증가량)
     tables/importance_feature.csv, tables/importance_group.csv, figures/01_importance_*.png
  2. 변수 간 상호작용 — 상위 변수 두 개를 함께 섞었을 때 RMSE 증가량이 따로 섞은 합보다 얼마나 큰지
     tables/interaction_pairs.csv, figures/02_interaction_matrix.png
     조건 조합 히트맵 (시간대 × 요일, 기온 구간 × 조업시간대): figures/03_*.png
  3. FN·FP 집중 공정조건 — 검증(5/16~8/15) + Test(8/16~9/14) 시간별 경보 결과를 조건별로 집계
     tables/fnfp_by_condition.csv, tables/fnfp_top_conditions.csv, figures/04~06_*.png
  summary.md — 주요 결과 자동 요약 (PPT 초안용)

사용 예 (저장소 최상위에서):
    python analysis/error_analysis.py --pred-dir results/final --data data/processed/okm_features2_2021.csv
    python analysis/error_analysis.py --pred-dir results/final_v2 --data ... --model tcn
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from src import data_pipeline as dp
from src.evaluate import peak_probability

THR = dp.PEAK_THRESHOLD_KW
SEED = 42
N_REPEATS = 3
LABELS = {"lstm": "LSTM", "tcn": "TCN", "tcn_lstm": "TCN-LSTM", "ensemble": "Weighted Ensemble",
          "naive_lag1": "직전 시간 그대로", "naive_lag168": "지난주 같은 시간"}

# 색: 차트 기본 잉크 + 상태색(FN=놓침 critical, FP=헛경보 serious)은 항상 라벨과 함께 쓴다
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, FN_COLOR, FP_COLOR = "#2a78d6", "#d03b3b", "#ec835a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FEATURE_GROUPS = {
    "과거 전력": ["전력_평균_실수"],
    "생산·설비": ["생산량", "가동상태_lag1", "설비기동여부", "연속조업시간_lag1", "생산량_lag1",
               "생산량_diff1", "일일누적생산량_lag1", "휴일특근조업_lag1"],
    "기상": ["기온", "풍속", "습도", "강수량"] + dp.WEATHER_DERIVED,
    "달력·조업 일정": dp.CALENDAR + ["점심반등예상량"],
}


# =========================================================
# 공통
# =========================================================

def setup_font():
    # Colab 에서 apt 로 설치한 나눔 글꼴은 matplotlib 캐시에 없을 수 있어 직접 등록한다
    for d in ["/usr/share/fonts/truetype/nanum", "/usr/share/fonts/opentype/noto"]:
        if os.path.isdir(d):
            for f in Path(d).glob("*.[ot]tf"):
                font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "Noto Sans KR", "Noto Sans CJK KR", "AppleGothic"]:
        if name in names:
            plt.rcParams["font.family"] = name
            break
    else:
        print("[경고] 한글 글꼴이 없어 그래프 한글이 깨질 수 있습니다. "
              "Colab: !apt-get -qq install fonts-nanum 후 런타임 재시작")
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
                         "axes.titlesize": 13, "font.size": 11, "figure.dpi": 150})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def base_name(col):
    """'점심시간여부(t+1)' → '점심시간여부'"""
    return col.replace("(t+1)", "")


def group_of(col):
    b = base_name(col)
    for g, members in FEATURE_GROUPS.items():
        if b in members:
            return g
    return "기타"


def load_choice(pred_dir, model):
    """test_summary.csv 의 config 에서 모델 설정을 읽는다."""
    s = pd.read_csv(pred_dir / "test_summary.csv", encoding="utf-8-sig")
    return json.loads(s.loc[s["model"] == model, "config"].iloc[0])


def pick_model(pred_dir, requested):
    if requested:
        return requested
    s = pd.read_csv(pred_dir / "test_summary.csv", encoding="utf-8-sig")
    dl = s[s["model"].isin(["lstm", "tcn", "tcn_lstm"])]
    return dl.sort_values("alert_f1_mean", ascending=False)["model"].iloc[0]


# =========================================================
# 1~2. Permutation importance + 상호작용
# =========================================================

def permutation_analysis(model_name, pred_dir, df, out):
    from tensorflow import keras

    params = load_choice(pred_dir, model_name)
    fs = params.get("feature_set", "base")
    data = dp.prepare_split(df, dp.get_final_split(df), params["lookback"], fs)
    _, names = dp.build_features(df, fs)
    model = keras.models.load_model(pred_dir / "models" / f"{model_name}_seed{SEED}.keras")
    ysc = data["y_scaler"]
    X = data["X_eval"]
    y = ysc.inverse_transform(data["y_eval"]).ravel()

    def rmse(Xp):
        p = ysc.inverse_transform(model.predict(Xp, verbose=0, batch_size=256)).ravel()
        return float(np.sqrt(np.mean((y - p) ** 2)))

    rng = np.random.default_rng(SEED)
    base = rmse(X)

    def permuted(cols):
        vals = []
        for _ in range(N_REPEATS):
            Xp = X.copy()
            for j in cols:
                Xp[:, :, j] = X[rng.permutation(len(X)), :, j]
            vals.append(rmse(Xp) - base)
        return float(np.mean(vals)), float(np.std(vals))

    print(f"  permutation importance: 변수 {len(names)}개 × {N_REPEATS}회 (기준 RMSE {base:.2f} kW)")
    rows = []
    for j, n in enumerate(names):
        m, s = permuted([j])
        rows.append({"feature": n, "group": group_of(n), "rmse_increase": m, "std": s})
    imp = pd.DataFrame(rows).sort_values("rmse_increase", ascending=False).reset_index(drop=True)
    imp.to_csv(out / "tables" / "importance_feature.csv", index=False, encoding="utf-8-sig")

    grows = []
    for g in dict.fromkeys(imp["group"]):
        cols = [names.index(n) for n in imp.loc[imp["group"] == g, "feature"]]
        m, s = permuted(cols)
        grows.append({"group": g, "n_features": len(cols), "rmse_increase": m, "std": s})
    gimp = pd.DataFrame(grows).sort_values("rmse_increase", ascending=False)
    gimp.to_csv(out / "tables" / "importance_group.csv", index=False, encoding="utf-8-sig")

    # 상호작용: 상위 6개 변수 쌍
    top = imp.head(6)["feature"].tolist()
    single = dict(zip(imp["feature"], imp["rmse_increase"]))
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            m, _ = permuted([names.index(top[a]), names.index(top[b])])
            pairs.append({"feature_a": top[a], "feature_b": top[b], "joint": m,
                          "interaction": m - single[top[a]] - single[top[b]]})
    inter = pd.DataFrame(pairs).sort_values("interaction", ascending=False)
    inter.to_csv(out / "tables" / "interaction_pairs.csv", index=False, encoding="utf-8-sig")

    plot_importance(imp, gimp, base, model_name, out)
    plot_interaction(inter, top, out)
    return imp, gimp, inter, base


def plot_importance(imp, gimp, base, model_name, out):
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top) + 1.2))
    ax.barh(top["feature"], top["rmse_increase"], color=BLUE, height=0.6)
    ax.errorbar(top["rmse_increase"], top["feature"], xerr=top["std"], fmt="none", ecolor=MUTED, capsize=2)
    for yv, v in zip(top["feature"], top["rmse_increase"]):
        ax.text(max(v, 0), yv, f"  {v:+.2f}", va="center", color=INK2, fontsize=9)
    ax.set_xlabel("변수를 섞었을 때 RMSE 증가 (kW)")
    ax.set_title(f"{LABELS[model_name]} 주요 영향 변수 (Test, 기준 RMSE {base:.2f} kW)")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "figures" / "01_importance_feature.png")

    g = gimp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(g) + 1.2))
    ax.barh(g["group"], g["rmse_increase"], color=BLUE, height=0.55)
    for yv, v, n in zip(g["group"], g["rmse_increase"], g["n_features"]):
        ax.text(max(v, 0), yv, f"  {v:+.2f} kW  (변수 {n}개)", va="center", color=INK2, fontsize=10)
    ax.set_xlabel("그룹 전체를 섞었을 때 RMSE 증가 (kW)")
    ax.set_title(f"{LABELS[model_name]} 변수 그룹별 영향")
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, g["rmse_increase"].max() * 1.45)
    save(fig, out / "figures" / "01_importance_group.png")


def plot_interaction(inter, top, out):
    n = len(top)
    M = np.full((n, n), np.nan)
    for _, r in inter.iterrows():
        a, b = top.index(r["feature_a"]), top.index(r["feature_b"])
        M[a, b] = M[b, a] = r["interaction"]
    lim = np.nanmax(np.abs(M)) or 1.0
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n), [base_name(t) for t in top], rotation=35, ha="right")
    ax.set_yticks(range(n), [base_name(t) for t in top])
    for a in range(n):
        for b in range(n):
            if not np.isnan(M[a, b]):
                ax.text(b, a, f"{M[a, b]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[a, b]) > lim * 0.6 else INK)
    fig.colorbar(im, ax=ax, label="상호작용 (함께 섞은 RMSE 증가 − 각각 섞은 합, kW)")
    ax.set_title("상위 변수 간 상호작용 (양수: 두 변수가 함께 작용)")
    save(fig, out / "figures" / "02_interaction_matrix.png")


# =========================================================
# 3. FN·FP 집중 공정조건
# =========================================================

def alert_frame(model_name, pred_dir, df):
    """검증 + Test 시간별 실제·예측·확률·경보·FN/FP 와 공정조건."""
    st = pd.read_csv(pred_dir / "alert_settings.csv", encoding="utf-8-sig")
    st = st[(st["model"] == model_name) & (st["seed"] == SEED)].iloc[0]
    sigma, cutoff = float(st["sigma"]), float(st["alert_cutoff"])
    frames = []
    for name, period in [("val_predictions.csv", "검증"), ("test_predictions.csv", "Test")]:
        p = pd.read_csv(pred_dir / name, encoding="utf-8-sig", parse_dates=["datetime"])
        f = pd.DataFrame({"datetime": p["datetime"], "period": period, "actual": p["actual"],
                          "pred": p[f"pred_{model_name}_seed{SEED}"]})
        f["prob"] = peak_probability(f["pred"], sigma, THR)
        frames.append(f)
    a = pd.concat(frames, ignore_index=True)
    a["event"] = a["actual"] >= THR
    a["alert"] = a["prob"] >= cutoff
    a["outcome"] = np.select([a.event & a.alert, a.event & ~a.alert, ~a.event & a.alert],
                             ["TP", "FN", "FP"], "TN")
    a["resid"] = a["actual"] - a["pred"]
    cond = df.set_index("datetime")
    a = a.join(cond, on="datetime", rsuffix="_src")
    yoil = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    zone = {0: "심야 00-06", 1: "오전 07-11", 2: "점심 12", 3: "오후 13-17", 4: "야간 18-23"}
    a["요일명"] = a["요일"].map(yoil)
    a["조업시간대명"] = pd.Categorical(a["조업시간대"].map(zone), categories=list(zone.values()), ordered=True)
    a["요일명"] = pd.Categorical(a["요일명"], categories=list(yoil.values()), ordered=True)
    a["일 유형"] = pd.Categorical(np.select([a["공휴일여부"] == 1, a["주말여부"] == 1], ["공휴일", "주말"], "영업일"),
                               categories=["영업일", "주말", "공휴일"], ordered=True)
    a["기온 구간"] = pd.cut(a["기온"], [-50, 0, 10, 20, 25, 28, 50],
                          labels=["0℃ 미만", "0~10℃", "10~20℃", "20~25℃", "25~28℃", "28℃ 이상"])
    a["불쾌지수 구간"] = pd.cut(a["불쾌지수_DI"], [0, 70, 75, 80, 100], labels=["70 미만", "70~75", "75~80", "80 이상(혹서)"])
    a["직전 생산량 구간"] = pd.cut(a["생산량_lag1"].fillna(0), [-1, 0, 500, 1000, 2000, 1e9],
                             labels=["0 (정지)", "1~500", "501~1000", "1001~2000", "2000 초과"])
    a["설비 상태"] = pd.Categorical(np.select([a["설비기동여부"] == 1, a["가동상태_lag1"] == 1], ["기동 직후", "가동 중"], "정지 중"),
                                 categories=["정지 중", "기동 직후", "가동 중"], ordered=True)
    a["시간대"] = a["시간"].map(lambda h: f"{h:02d}시")
    return a, sigma, cutoff


CONDITIONS = ["시간대", "요일명", "일 유형", "조업시간대명", "설비 상태", "월요일기동시간여부",
              "기온 구간", "불쾌지수 구간", "직전 생산량 구간", "월"]


def fnfp_tables(a, out):
    tot_fn, tot_fp, tot_h = (a.outcome == "FN").sum(), (a.outcome == "FP").sum(), len(a)
    rows = []
    for c in CONDITIONS:
        for v, g in a.groupby(c, observed=True):
            ev = g.event.sum()
            rows.append({
                "condition": c, "value": str(v), "hours": len(g), "peaks": int(ev),
                "FN": int((g.outcome == "FN").sum()), "FP": int((g.outcome == "FP").sum()),
                "miss_rate": (g.outcome == "FN").sum() / ev if ev else np.nan,
                "false_alarm_rate": (g.outcome == "FP").sum() / max((~g.event).sum(), 1),
                "FN_share": (g.outcome == "FN").sum() / tot_fn if tot_fn else np.nan,
                "FP_share": (g.outcome == "FP").sum() / tot_fp if tot_fp else np.nan,
                "hour_share": len(g) / tot_h,
                "mean_resid_on_peaks": g.loc[g.event, "resid"].mean() if ev else np.nan,
            })
    t = pd.DataFrame(rows)
    # 집중도(lift) = 오류 비중 / 시간 비중. 1보다 크면 그 조건에 오류가 몰려 있다
    t["FN_lift"] = t["FN_share"] / t["hour_share"]
    t["FP_lift"] = t["FP_share"] / t["hour_share"]
    t.to_csv(out / "tables" / "fnfp_by_condition.csv", index=False, encoding="utf-8-sig")
    fn_top = t[(t.FN >= 3)].sort_values(["FN_lift", "FN"], ascending=False).head(10)
    fp_top = t[(t.FP >= 5)].sort_values(["FP_lift", "FP"], ascending=False).head(10)
    pd.concat([fn_top.assign(kind="FN"), fp_top.assign(kind="FP")]).to_csv(
        out / "tables" / "fnfp_top_conditions.csv", index=False, encoding="utf-8-sig")
    return t, fn_top, fp_top


def heatmap(ax, M, xlabels, ylabels, title, cmap, fmt="{:.0f}", cbar_label=""):
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    vmax = np.nanmax(M) if np.isfinite(np.nanmax(M)) else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isfinite(v) and v != 0:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v > vmax * 0.6 else INK)
    ax.set_title(title)
    return im


def plot_fnfp(a, t, model_name, out):
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff"] + SEQ)
    red = LinearSegmentedColormap.from_list("red", ["#ffffff", "#f3c1c1", FN_COLOR])
    org = LinearSegmentedColormap.from_list("org", ["#ffffff", "#f8d3c3", FP_COLOR])
    days = ["월", "화", "수", "목", "금", "토", "일"]
    hours = list(range(24))

    # 03-a: 시간대 × 요일 피크 발생 수 (상호작용: 언제 피크가 몰리는가)
    piv = a[a.event].pivot_table(index="요일명", columns="시간", values="actual", aggfunc="size").reindex(index=days, columns=hours)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    im = heatmap(ax, piv.to_numpy(dtype=float), hours, days, f"피크(≥{THR:.0f} kW) 발생 시간 수 — 요일 × 시간대 (검증+Test)", seq)
    fig.colorbar(im, ax=ax, label="시간 수"); ax.set_xlabel("시간")
    save(fig, out / "figures" / "03_peak_hour_weekday.png")

    # 03-b: 기온 구간 × 조업시간대 평균 전력 (상호작용)
    piv = a.pivot_table(index="기온 구간", columns="조업시간대명", values="actual", aggfunc="mean", observed=False)
    cols = [c for c in ["심야 00-06", "오전 07-11", "점심 12", "오후 13-17", "야간 18-23"] if c in piv.columns]
    piv = piv[cols]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = heatmap(ax, piv.to_numpy(dtype=float), cols, [str(i) for i in piv.index],
                 "평균 전력 (kW) — 기온 구간 × 조업시간대 (검증+Test)", seq, "{:.0f}")
    fig.colorbar(im, ax=ax, label="kW")
    save(fig, out / "figures" / "03_temp_x_zone_power.png")

    # 04: FN / FP 요일 × 시간대
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8))
    for ax, kind, cmap in [(axes[0], "FN", red), (axes[1], "FP", org)]:
        piv = a[a.outcome == kind].pivot_table(index="요일명", columns="시간", values="actual",
                                              aggfunc="size").reindex(index=days, columns=hours)
        label = "FN: 피크를 놓친 시간" if kind == "FN" else "FP: 헛경보 시간"
        im = heatmap(ax, piv.fillna(0).to_numpy(dtype=float), hours, days, f"{label} — 요일 × 시간대", cmap)
        fig.colorbar(im, ax=ax, label="시간 수")
    axes[1].set_xlabel("시간")
    fig.suptitle(f"{LABELS[model_name]} 오류가 몰리는 시간 (검증+Test)", fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, out / "figures" / "04_fnfp_hour_weekday.png")

    # 05: 조건별 놓침률 · 헛경보율
    show = ["일 유형", "조업시간대명", "설비 상태", "불쾌지수 구간", "기온 구간", "직전 생산량 구간"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), show):
        sub = t[t.condition == c].copy()
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["miss_rate"].fillna(0), width=0.38, color=FN_COLOR, label="놓침률 (FN / 피크)")
        ax.bar(x + 0.2, sub["false_alarm_rate"], width=0.38, color=FP_COLOR, label="헛경보율 (FP / 비피크)")
        for xi, (pk, fn) in enumerate(zip(sub["peaks"], sub["FN"])):
            ax.text(xi - 0.2, (sub["miss_rate"].fillna(0).iloc[xi]) + 0.02, f"{fn}/{pk}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, sub["value"], rotation=25, ha="right")
        ax.set_ylim(0, 1.05); ax.set_title(c)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(f"{LABELS[model_name]} 공정조건별 놓침률·헛경보율 (막대 위: FN/피크 수, 검증+Test)", fontweight="bold")
    fig.tight_layout()
    save(fig, out / "figures" / "05_fnfp_rate_by_condition.png")

    # 06: Test 구간 실제 vs 예측 + FN/FP 표시
    s = a[a.period == "Test"].sort_values("datetime")
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.plot(s.datetime, s.actual, color=INK2, linewidth=1, label="실제")
    ax.plot(s.datetime, s.pred, color=BLUE, linewidth=1.2, label=f"예측 ({LABELS[model_name]})")
    ax.axhline(THR, color=MUTED, linestyle="--", linewidth=1)
    ax.text(s.datetime.iloc[0], THR + 2, f"이상 기준 {THR:.0f} kW", color=MUTED, fontsize=9)
    for kind, color, marker in [("FN", FN_COLOR, "v"), ("FP", FP_COLOR, "^")]:
        k = s[s.outcome == kind]
        ax.scatter(k.datetime, k.actual, color=color, marker=marker, s=36, zorder=3,
                   edgecolors="white", linewidths=0.8, label=f"{kind} ({len(k)}시간)")
    ax.set_ylabel("kW"); ax.set_title(f"Test 구간 실제·예측과 경보 오류 (8/16~9/14)")
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "figures" / "06_test_timeline_fnfp.png")


# =========================================================
# 요약
# =========================================================

def write_summary(model_name, a, sigma, cutoff, imp, gimp, inter, base, fn_top, fp_top, out):
    te = a[a.period == "Test"]
    L = [f"# 영향요인 및 오류분석 요약 — {LABELS[model_name]}", "",
         f"- 이상 기준 {THR:.0f} kW, 이상 확률 σ = {sigma:.2f} kW, 경보 기준 확률 {cutoff:.2f}",
         f"- 검증+Test {len(a)}시간 중 피크 {int(a.event.sum())}시간 · FN {int((a.outcome == 'FN').sum())} · "
         f"FP {int((a.outcome == 'FP').sum())} (Test만: 피크 {int(te.event.sum())} · "
         f"FN {int((te.outcome == 'FN').sum())} · FP {int((te.outcome == 'FP').sum())})", ""]
    if imp is not None:
        L += [f"## 1. 주요 영향 변수 (Test, 기준 RMSE {base:.2f} kW)", "",
              "| 순위 | 변수 | 그룹 | 섞었을 때 RMSE 증가 (kW) |", "| --- | --- | --- | --- |"]
        L += [f"| {i + 1} | {r.feature} | {r.group} | {r.rmse_increase:+.2f} |" for i, r in imp.head(10).iterrows()]
        L += ["", "| 그룹 | 변수 수 | RMSE 증가 (kW) |", "| --- | --- | --- |"]
        L += [f"| {r.group} | {r.n_features} | {r.rmse_increase:+.2f} |" for r in gimp.itertuples()]
        L += ["", "## 2. 변수 간 상호작용 (상위 3쌍)", "", "| 변수 A | 변수 B | 상호작용 (kW) |", "| --- | --- | --- |"]
        L += [f"| {r.feature_a} | {r.feature_b} | {r.interaction:+.2f} |" for r in inter.head(3).itertuples()]
    L += ["", "## 3. FN(놓친 피크)이 몰리는 조건 (집중도 = FN 비중 / 시간 비중)", "",
          "| 조건 | 값 | 피크 | FN | 놓침률 | 집중도 | 피크 시 평균 과소예측 (kW) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r.condition} | {r.value} | {r.peaks} | {r.FN} | {r.miss_rate:.0%} | {r.FN_lift:.1f}배 | {r.mean_resid_on_peaks:+.1f} |"
          for r in fn_top.head(6).itertuples()]
    L += ["", "## 4. FP(헛경보)가 몰리는 조건", "",
          "| 조건 | 값 | 시간 | FP | 헛경보율 | 집중도 |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r.condition} | {r.value} | {r.hours} | {r.FP} | {r.false_alarm_rate:.0%} | {r.FP_lift:.1f}배 |"
          for r in fp_top.head(6).itertuples()]
    L += ["", "그림: figures/ · 표: tables/"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, help="final_evaluate.py 결과 폴더")
    ap.add_argument("--data", required=True, help="okm_features2_2021.csv 경로")
    ap.add_argument("--model", choices=["lstm", "tcn", "tcn_lstm", "ensemble"],
                    help="분석할 모델 (기본: Test F1 이 가장 높은 딥러닝 모델)")
    ap.add_argument("--out-dir", help="결과 폴더 (기본: <pred-dir>/analysis_<모델>)")
    ap.add_argument("--skip-importance", action="store_true", help="permutation importance 생략 (모델 파일 없을 때)")
    args = ap.parse_args()

    setup_font()
    pred_dir = Path(args.pred_dir)
    model_name = pick_model(pred_dir, args.model)
    out = Path(args.out_dir) if args.out_dir else pred_dir / f"analysis_{model_name}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df = dp.load_data(args.data)
    print(f"[분석 모델] {LABELS[model_name]}  →  {out}")

    imp = gimp = inter = None
    base = np.nan
    if not args.skip_importance and model_name != "ensemble":
        imp, gimp, inter, base = permutation_analysis(model_name, pred_dir, df, out)
    a, sigma, cutoff = alert_frame(model_name, pred_dir, df)
    a.to_csv(out / "tables" / "hourly_alerts.csv", index=False, encoding="utf-8-sig")
    t, fn_top, fp_top = fnfp_tables(a, out)
    plot_fnfp(a, t, model_name, out)
    write_summary(model_name, a, sigma, cutoff, imp, gimp, inter, base, fn_top, fp_top, out)
    print((out / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
