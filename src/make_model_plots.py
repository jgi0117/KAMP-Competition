# -*- coding: utf-8 -*-
"""모델 평가 결과 시각화 차트 생성 모듈."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 스타일 및 폰트 설정
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["Malgun Gothic", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

out_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

data_path = Path(__file__).resolve().parent.parent / "data" / "okm_model_predictions_test.csv"
df_pred = pd.read_csv(data_path, encoding="utf-8-sig")
df_pred["datetime"] = pd.to_datetime(df_pred["datetime"])

# 1. 모델 벤치마크 비교 바 차트 (64개 Iteration 4 피처셋 기준)
models = [
    "Top-3 GBDT (앙상블)", "CatBoost", "XGBoost", "Tri-GBDT (기존)", "LightGBM",
    "HistGBM", "RandomForest", "DeepPowerMLP", "Ridge (Linear)", "ElasticNet"
]
val_rmse = [13.58, 14.49, 14.76, 12.77, 12.51, 12.89, 17.58, 14.92, 18.32, 18.17]
test_rmse = [9.34, 9.51, 9.56, 9.57, 10.15, 10.30, 10.81, 12.64, 13.94, 14.15]
families = ["Ensemble", "Tree", "Tree", "Ensemble", "Tree", "Tree", "Tree", "Deep Learning", "Linear", "Linear"]

palette = {"Ensemble": "#10B981", "Tree": "#3B82F6", "Deep Learning": "#8B5CF6", "Linear": "#EF4444"}
bar_colors = [palette[f] for f in families]

fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
x = np.arange(len(models))
width = 0.38

rects1 = ax.bar(x - width/2, val_rmse, width, label="Validation RMSE (7/1~8/15)", color="#93C5FD", edgecolor="#1E40AF", alpha=0.85)
rects2 = ax.bar(x + width/2, test_rmse, width, label="Test RMSE (8/16~9/14)", color=bar_colors, edgecolor="#111827", alpha=0.95)

ax.set_ylabel("RMSE (kW) [낮을수록 우수]", fontsize=12, fontweight="bold")
ax.set_title("전력사용량 예측 다중 모델 종합 벤치마크 (Iteration 4: 64개 피처 기준)", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right", fontsize=10, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig.savefig(out_dir / "model_01_benchmark_comparison.png", dpi=300)
plt.close(fig)
print("Saved model_01_benchmark_comparison.png")

# 2. Test 기간 시계열 실제값 vs 예측값 비교 차트
col_actual = "실제값"
col_top3 = [c for c in df_pred.columns if "Top-3" in c][0]
col_cb = [c for c in df_pred.columns if "CatBoost" in c][0]
col_ridge = [c for c in df_pred.columns if "Ridge" in c][0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=300, sharex=False, gridspec_kw={"height_ratios": [2, 1.2]})

blackout_start = pd.to_datetime("2021-08-28 18:00:00")
blackout_end = pd.to_datetime("2021-08-29 10:00:00")

# 상단: 전체 Test 기간
ax1.plot(df_pred["datetime"], df_pred[col_actual], label="실제 전력 (Actual)", color="#111827", linewidth=1.2, alpha=0.9)
ax1.plot(df_pred["datetime"], df_pred[col_top3], label="Top-3 GBDT 앙상블 (권장, RMSE 9.34kW)", color="#10B981", linewidth=1.4, linestyle="--")
ax1.plot(df_pred["datetime"], df_pred[col_cb], label="CatBoost (RMSE 9.51kW)", color="#3B82F6", linewidth=1.0, alpha=0.7)
ax1.plot(df_pred["datetime"], df_pred[col_ridge], label="Ridge 선형회귀 (과소적합)", color="#EF4444", linewidth=0.9, linestyle=":", alpha=0.6)
ax1.axvspan(blackout_start, blackout_end, color="#F87171", alpha=0.3, label="블랙아웃 정전 구간 (실제 0 kW)")

ax1.set_ylabel("전력사용량 (kW)", fontsize=12, fontweight="bold")
ax1.set_title("Test 기간(8/16 ~ 9/14) 모델별 실제값 vs 예측값 시계열 추종성 비교 (Iteration 4)", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)

# 하단: 정전 구간 확대
zoom_mask = (df_pred["datetime"] >= "2021-08-27") & (df_pred["datetime"] <= "2021-09-02")
df_zoom = df_pred[zoom_mask]

ax2.plot(df_zoom["datetime"], df_zoom[col_actual], label="실제 전력 (Actual)", color="#111827", linewidth=1.8, marker="o", markersize=3)
ax2.plot(df_zoom["datetime"], df_zoom[col_top3], label="Top-3 앙상블 예측 (대기전력 ~30kW 유지 -> 누수 0% 입증)", color="#10B981", linewidth=1.8, linestyle="--")
ax2.axvspan(blackout_start, blackout_end, color="#F87171", alpha=0.25)
ax2.set_ylabel("전력사용량 (kW)", fontsize=11, fontweight="bold")
ax2.set_xlabel("일시 (Date & Time)", fontsize=11, fontweight="bold")
ax2.set_title("[확대 검증] 8/28~8/29 정전 구간 0% 데이터 누수 방어 검증 (사전 0kW 미인지, 기저 대기전력 정상 추정)", fontsize=12, fontweight="bold", color="#B91C1C")
ax2.legend(loc="upper right", fontsize=9.5)

plt.tight_layout()
fig.savefig(out_dir / "model_02_predictions_timeseries_test.png", dpi=300)
plt.close(fig)
print("Saved model_02_predictions_timeseries_test.png")

# 3. 잔차(Residual) 분포 차트
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
normal_mask = ~((df_pred["datetime"] >= blackout_start) & (df_pred["datetime"] <= blackout_end))
res_clean_ens = df_pred.loc[normal_mask, col_actual] - df_pred.loc[normal_mask, col_top3]
res_clean_ridge = df_pred.loc[normal_mask, col_actual] - df_pred.loc[normal_mask, col_ridge]

sns.kdeplot(res_clean_ens, ax=ax, label=f"Top-3 GBDT 앙상블 (Mean={res_clean_ens.mean():.2f}, Std={res_clean_ens.std():.2f})", color="#10B981", fill=True, alpha=0.3, linewidth=2)
sns.kdeplot(res_clean_ridge, ax=ax, label=f"Ridge 선형회귀 (Mean={res_clean_ridge.mean():.2f}, Std={res_clean_ridge.std():.2f})", color="#EF4444", fill=True, alpha=0.2, linewidth=2)
ax.axvline(0, color="#6B7280", linestyle="--", linewidth=1)
ax.set_title("Test 정상 조업 구간 잔차(Residual) 분포 비교 (Iteration 4: 정규성 및 무편향 검증)", fontsize=14, fontweight="bold")
ax.set_xlabel("잔차 = 실제값 - 예측값 (kW)", fontsize=12, fontweight="bold")
ax.set_ylabel("밀도 (Density)", fontsize=12, fontweight="bold")
ax.legend(fontsize=11)

plt.tight_layout()
fig.savefig(out_dir / "model_03_residual_distribution.png", dpi=300)
plt.close(fig)
print("Saved model_03_residual_distribution.png")
