# -*- coding: utf-8 -*-
"""LSTM·TCN 계열 모델용 데이터 파이프라인.

docs/modeling/구현가이드 6~11장의 흐름(로드 → 시간순 분리 → Scaling → Sequence 생성)을 따르되,
팀 합의에 따라 고정 3분할 대신 시간순 확장(expanding) fold 3개 + 고정 Test 구간을 사용한다.

예측 문제: 시점 t-L ... t-1 의 입력(L = lookback)으로 시점 t 의 `전력_평균_실수`를 예측한다.
즉 가이드의 "과거 N시간 → 다음 1시간(t+1)" 과 같은 정의다.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================================================
# 기본 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

# 정현님 전처리 결과 파일. 경로가 다르면 환경변수 KAMP_DATA_PATH 로 지정한다.
DATA_PATH = Path(os.getenv("KAMP_DATA_PATH", ROOT_DIR / "data" / "processed" / "okm_cleaned.csv"))

TARGET_COLUMN = "전력_평균_실수"

# 가이드 5장의 시작 Feature. 시간_sin/cos, 주말여부는 파일에 없으면 load_data()에서 만든다.
FEATURE_COLUMNS = [
    "전력_평균_실수",
    "생산량",
    "기온",
    "풍속",
    "습도",
    "강수량",
    "시간_sin",
    "시간_cos",
    "주말여부",
]

# 최종 Test 구간 시작. 튜닝(그리드 서치) 과정에서는 이 이후 데이터를 평가에 쓰지 않는다.
TEST_START = pd.Timestamp("2021-08-16 00:00:00")

# 시간순 확장 fold 3개: (검증 시작일, 검증 종료일). 학습은 각 검증 시작 이전 전체 구간.
FOLD_VAL_RANGES = [
    ("2021-05-16", "2021-06-15"),
    ("2021-06-16", "2021-07-15"),
    ("2021-07-16", "2021-08-15"),
]

# 학습 구간 끝에서 떼어내 EarlyStopping 에만 쓰는 기간.
# 모델 선택용 검증 구간을 EarlyStopping 에 다시 쓰지 않기 위함이다.
EARLY_STOP_DAYS = 14

# 피크 기준: 각 fold 학습 구간 전력의 상위 5% 값 (Test 를 보기 전에 정해짐)
PEAK_QUANTILE = 0.95

# 8/28~8/29 셧다운(전력 0 kW) 같은 비정상 구간을 Target 으로 학습·평가하지 않는다.
# 입력 window 에는 그대로 남겨 시계열 연속성을 유지한다.
EXCLUDE_SHUTDOWN_TARGETS = True


# =========================================================
# 데이터 불러오기
# =========================================================

def load_data(path=None):
    """전처리 완료 CSV 를 읽고 datetime 정렬, 기본 파생변수, 셧다운 플래그를 보장한다."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"데이터 파일이 없습니다: {path}\n"
            "정현님 전처리 결과를 data/processed/okm_cleaned.csv 에 두거나 "
            "환경변수 KAMP_DATA_PATH 로 경로를 지정하세요."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # 날짜(20210101 또는 2021-01-01) + 시간(0~23) → datetime
        df["datetime"] = (
            pd.to_datetime(df["날짜"].astype(str).str.slice(0, 10))
            + pd.to_timedelta(df["시간"].astype(int), unit="h")
        )

    df = df.sort_values("datetime").reset_index(drop=True)

    gaps = df["datetime"].diff().dropna()
    if not (gaps == pd.Timedelta(hours=1)).all():
        raise ValueError("datetime 이 1시간 간격으로 연속되지 않습니다. 전처리 결과를 확인하세요.")

    hour = df["datetime"].dt.hour
    if "시간_sin" not in df.columns:
        df["시간_sin"] = np.sin(2 * np.pi * hour / 24)
    if "시간_cos" not in df.columns:
        df["시간_cos"] = np.cos(2 * np.pi * hour / 24)
    if "주말여부" not in df.columns:
        df["주말여부"] = (df["datetime"].dt.dayofweek >= 5).astype(int)

    if "공장_셧다운_여부" not in df.columns:
        df["공장_셧다운_여부"] = df[TARGET_COLUMN] == 0
    df["공장_셧다운_여부"] = df["공장_셧다운_여부"].astype(bool)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(f"필요한 컬럼이 없습니다: {missing}")
    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any().any():
        raise ValueError("Feature/Target 에 결측치가 있습니다.")

    return df


# =========================================================
# 시간순 분리
# =========================================================

@dataclass
class Split:
    """하나의 fold(또는 최종 Test 분할)에 대한 행 단위 마스크."""
    name: str
    train: np.ndarray        # Scaler fit + 모델 학습
    early_stop: np.ndarray   # EarlyStopping 전용
    eval: np.ndarray         # 검증(fold) 또는 Test(final)


def get_folds(df):
    """그리드 서치용 시간순 확장 fold 3개."""
    dt = df["datetime"]
    folds = []
    for i, (val_start, val_end) in enumerate(FOLD_VAL_RANGES, start=1):
        val_start = pd.Timestamp(val_start)
        val_end = pd.Timestamp(val_end) + pd.Timedelta(hours=23)
        es_start = val_start - pd.Timedelta(days=EARLY_STOP_DAYS)
        folds.append(Split(
            name=f"fold{i}",
            train=(dt < es_start).to_numpy(),
            early_stop=((dt >= es_start) & (dt < val_start)).to_numpy(),
            eval=((dt >= val_start) & (dt <= val_end)).to_numpy(),
        ))
    return folds


def get_final_split(df):
    """최종 평가용 분할: Test 이전 전체로 학습, Test 구간은 마지막에 한 번만 평가."""
    dt = df["datetime"]
    es_start = TEST_START - pd.Timedelta(days=EARLY_STOP_DAYS)
    return Split(
        name="final",
        train=(dt < es_start).to_numpy(),
        early_stop=((dt >= es_start) & (dt < TEST_START)).to_numpy(),
        eval=(dt >= TEST_START).to_numpy(),
    )


# =========================================================
# Scaling
# =========================================================

def scale_data(df, train_mask):
    """Scaler 는 학습 구간으로만 fit 하고, 전체에 transform 한다 (Leakage 방지)."""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[[TARGET_COLUMN]].to_numpy(dtype=np.float64)

    x_scaler = StandardScaler().fit(X[train_mask])
    y_scaler = StandardScaler().fit(y[train_mask])

    X_scaled = x_scaler.transform(X).astype(np.float32)
    y_scaled = y_scaler.transform(y).astype(np.float32)
    return X_scaled, y_scaled, x_scaler, y_scaler


# =========================================================
# Sequence 생성
# =========================================================

def create_sequences(X, y, lookback=24):
    """X[i-lookback:i] → y[i]. 반환하는 indices 는 Target 행 번호다."""
    indices = np.arange(lookback, len(X))
    windows = np.lib.stride_tricks.sliding_window_view(X, lookback, axis=0)
    # sliding_window_view 결과: (n - lookback + 1, n_features, lookback)
    X_seq = windows[:-1].transpose(0, 2, 1)
    y_seq = y[indices]
    return np.ascontiguousarray(X_seq), y_seq, indices


# =========================================================
# Split 별 데이터셋
# =========================================================

def prepare_split(df, split, lookback=24):
    """하나의 Split 에 대해 학습/EarlyStopping/평가용 Sequence 와 Scaler, 피크 기준을 만든다.

    각 Sequence 는 Target 시점이 속한 구간으로 배정한다. 입력 window 는 항상 Target 이전
    시점만 포함하므로, 평가 구간 Sequence 가 학습 구간 값을 입력으로 보는 것은 Leakage 가 아니다.
    """
    X_scaled, y_scaled, x_scaler, y_scaler = scale_data(df, split.train)
    X_seq, y_seq, indices = create_sequences(X_scaled, y_scaled, lookback)

    keep = np.ones(len(df), dtype=bool)
    if EXCLUDE_SHUTDOWN_TARGETS:
        keep = ~df["공장_셧다운_여부"].to_numpy()

    def pick(mask):
        sel = (mask & keep)[indices]
        return X_seq[sel], y_seq[sel], indices[sel]

    X_train, y_train, idx_train = pick(split.train)
    X_es, y_es, idx_es = pick(split.early_stop)
    X_eval, y_eval, idx_eval = pick(split.eval)

    train_target = df.loc[split.train & keep, TARGET_COLUMN]
    peak_threshold = float(train_target.quantile(PEAK_QUANTILE))

    return {
        "split": split.name,
        "X_train": X_train, "y_train": y_train, "idx_train": idx_train,
        "X_es": X_es, "y_es": y_es, "idx_es": idx_es,
        "X_eval": X_eval, "y_eval": y_eval, "idx_eval": idx_eval,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "peak_threshold": peak_threshold,
    }
