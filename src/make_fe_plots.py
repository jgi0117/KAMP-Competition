# -*- coding: utf-8 -*-
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from src.preprocessing import run_preprocessing_pipeline
from src.config import FIGURES_DIR

df = run_preprocessing_pipeline()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# [플롯 1] 시간대별 일변화 및 요일별 주간 패턴
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# 1-1. 시간대별 박스플롯
sns.boxplot(data=df, x='시간', y='전력_평균_실수', color='#3498db', ax=ax1, fliersize=2)
# 점심시간(12시) 강조
ax1.axvspan(11.5, 12.5, color='#e74c3c', alpha=0.2, label='12시 점심시간 Dip (급감)')
# 주간 메인 가동시간(08~16시) 강조
ax1.set_title('[시간대별 패턴] 점심시간(12시) 급감 및 주간(08~16시) 고부하', fontsize=12, fontweight='bold')
ax1.set_xlabel('시간 (0 ~ 23시)')
ax1.set_ylabel('전력 (kW)')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper left')

# 1-2. 요일별 박스플롯
day_labels = {1: '월(1)', 2: '화(2)', 3: '수(3)', 4: '목(4)', 5: '금(5)', 6: '토(6)', 7: '일(7)'}
df['요일명'] = df['요일'].map(day_labels)
day_order = ['월(1)', '화(2)', '수(3)', '목(4)', '금(5)', '토(6)', '일(7)']
palette_day = ['#2980b9']*5 + ['#e67e22', '#e74c3c']
sns.boxplot(data=df, x='요일명', y='전력_평균_실수', order=day_order, palette=palette_day, ax=ax2, fliersize=2)
ax2.axhline(df[df['요일'].isin([6, 7])]['전력_평균_실수'].mean(), color='red', linestyle='--', label='주말 평균 (46.1 kW)')
ax2.axhline(df[~df['요일'].isin([6, 7])]['전력_평균_실수'].mean(), color='blue', linestyle='--', label='평일 평균 (112.4 kW)')
ax2.set_title('[요일별 패턴] 평일 고부하 vs 주말(토/일) 기저전력 급감', fontsize=12, fontweight='bold')
ax2.set_xlabel('요일')
ax2.set_ylabel('전력 (kW)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right')

plt.tight_layout()
p1 = FIGURES_DIR / "fe_01_time_day_patterns.png"
plt.savefig(p1, dpi=200)
plt.close()
print("Saved:", p1)

# -------------------------------------------------------------------------
# [플롯 2] 요일 × 시간 2차원 전력 히트맵
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
pivot_table = df.pivot_table(index='요일명', columns='시간', values='전력_평균_실수', aggfunc='mean')
pivot_table = pivot_table.reindex(['월(1)', '화(2)', '수(3)', '목(4)', '금(5)', '토(6)', '일(7)'])

sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='.0f', cbar_kws={'label': '평균 전력 (kW)'}, ax=ax)
ax.set_title('[2차원 패턴 히트맵] 요일 × 시간대별 평균 전력 소비 매트릭스', fontsize=13, fontweight='bold')
ax.set_xlabel('시간 (0 ~ 23시)')
ax.set_ylabel('요일')
plt.tight_layout()
p2 = FIGURES_DIR / "fe_02_heatmap_hour_day.png"
plt.savefig(p2, dpi=200)
plt.close()
print("Saved:", p2)

# -------------------------------------------------------------------------
# [플롯 3] 시계열 지연(Lag) 자기상관성 (1시간 전, 24시간 전, 1주일 전)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Lag 1h
df['lag1'] = df['전력_평균_실수'].shift(1)
axes[0].scatter(df['lag1'], df['전력_평균_실수'], alpha=0.25, color='#2980b9', s=12)
axes[0].plot([0, 210], [0, 210], color='red', linestyle='--')
axes[0].set_title(f'직전 1시간 전력 (Lag 1h)\n상관계수: {df["전력_평균_실수"].corr(df["lag1"]):.4f}', fontweight='bold')
axes[0].set_xlabel('t-1 전력 (kW)')
axes[0].set_ylabel('현재 t 전력 (kW)')
axes[0].grid(True, linestyle='--', alpha=0.5)

# Lag 24h
df['lag24'] = df['전력_평균_실수'].shift(24)
axes[1].scatter(df['lag24'], df['전력_평균_실수'], alpha=0.25, color='#16a085', s=12)
axes[1].plot([0, 210], [0, 210], color='red', linestyle='--')
axes[1].set_title(f'전일 동일시간 전력 (Lag 24h)\n상관계수: {df["전력_평균_실수"].corr(df["lag24"]):.4f}', fontweight='bold')
axes[1].set_xlabel('t-24 전력 (kW)')
axes[1].set_ylabel('현재 t 전력 (kW)')
axes[1].grid(True, linestyle='--', alpha=0.5)

# Lag 168h (1주일 전)
df['lag168'] = df['전력_평균_실수'].shift(168)
axes[2].scatter(df['lag168'], df['전력_평균_실수'], alpha=0.25, color='#8e44ad', s=12)
axes[2].plot([0, 210], [0, 210], color='red', linestyle='--')
axes[2].set_title(f'1주일 전 동일시간 전력 (Lag 168h)\n상관계수: {df["전력_평균_실수"].corr(df["lag168"]):.4f}', fontweight='bold')
axes[2].set_xlabel('t-168 전력 (kW)')
axes[2].set_ylabel('현재 t 전력 (kW)')
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
p3 = FIGURES_DIR / "fe_03_lag_correlations.png"
plt.savefig(p3, dpi=200)
plt.close()
print("Saved:", p3)

# -------------------------------------------------------------------------
# [플롯 4] 생산 가동 상태 및 설비 기동(Start-up) 효과
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 4-1. 비가동 vs 가동 전력 밀도 분포(KDE)
df['가동상태'] = np.where(df['생산량'] > 0, '가동 중 (생산량 > 0)', '비가동 (생산량 = 0)')
sns.kdeplot(data=df, x='전력_평균_실수', hue='가동상태', fill=True, common_norm=False, palette=['#e74c3c', '#2980b9'], ax=ax1)
ax1.set_title('[가동 상태별 밀도] 비가동(대기전력 ~23kW) vs 가동(100~160kW) 이봉(Bimodal) 분포', fontsize=11, fontweight='bold')
ax1.set_xlabel('전력 (kW)')
ax1.set_ylabel('밀도 (Density)')
ax1.grid(True, linestyle='--', alpha=0.5)

# 4-2. 설비 기동(Start-up: 생산량 0 -> 생산량 > 0) 시점의 전력 변화
df['직전_생산량'] = df['생산량'].shift(1).fillna(0)
df['설비기동여부'] = (df['직전_생산량'] == 0) & (df['생산량'] > 0)
df['가동전환유형'] = '연속가동'
df.loc[df['생산량'] == 0, '가동전환유형'] = '정지/대기'
df.loc[df['설비기동여부'], '가동전환유형'] = '설비기동(시작)'

sns.boxplot(data=df, x='가동전환유형', y='전력_평균_실수', order=['정지/대기', '설비기동(시작)', '연속가동'], palette=['#95a5a6', '#e67e22', '#2ecc71'], ax=ax2)
ax2.set_title('[설비 기동 효과] 정지에서 가동 전환 시 급격한 기동 부하 발생', fontsize=11, fontweight='bold')
ax2.set_ylabel('전력 (kW)')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
p4 = FIGURES_DIR / "fe_04_production_dynamics.png"
plt.savefig(p4, dpi=200)
plt.close()
print("Saved:", p4)

# -------------------------------------------------------------------------
# [플롯 5] 기상(기온, 불쾌지수)과 냉난방 부하 곡선
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# 5-1. 기온 vs 전력 (2차 다항 추세선)
# 평일 주간(가동시간)만 필터링하여 순수 날씨 영향 분리 관찰
df_work = df[(~df['요일'].isin([6, 7])) & (df['생산량'] > 0)].copy()
sns.regplot(data=df_work, x='기온', y='전력_평균_실수', order=2, scatter_kws={'alpha': 0.2, 'color': '#34495e', 's': 15}, line_kws={'color': '#e74c3c', 'lw': 2.5}, ax=ax1)
ax1.set_title('[기온-전력 비선형 곡선 (평일 가동시간 기준)]\n20~24℃ 대비 저온(난방) 및 고온(냉방피크) 전력 상승', fontsize=11, fontweight='bold')
ax1.set_xlabel('기온 (℃)')
ax1.set_ylabel('전력 (kW)')
ax1.grid(True, linestyle='--', alpha=0.5)

# 5-2. 불쾌지수(Discomfort Index) 계산 및 전력 관계
T = df_work['기온']
RH = df_work['습도']
df_work['불쾌지수'] = 0.81 * T + 0.01 * RH * (0.99 * T - 14.3) + 46.3
df_work['불쾌지수_구간'] = pd.cut(df_work['불쾌지수'], bins=[-10, 68, 75, 80, 100], labels=['쾌적(<68)', '보통(68~75)', '높음(75~80)', '매우높음(>80)'])

sns.boxplot(data=df_work, x='불쾌지수_구간', y='전력_평균_실수', palette='Blues', ax=ax2)
ax2.set_title('[불쾌지수 구간별 전력]\n불쾌지수 75 이상 고습/고온 구간에서 냉방 부하 급증', fontsize=11, fontweight='bold')
ax2.set_xlabel('불쾌지수 단계')
ax2.set_ylabel('전력 (kW)')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
p5 = FIGURES_DIR / "fe_05_weather_cooling_heating.png"
plt.savefig(p5, dpi=200)
plt.close()
print("Saved:", p5)

print("\n모든 5개 피처 엔지니어링 시각화 플롯이 성공적으로 생성되었습니다.")
