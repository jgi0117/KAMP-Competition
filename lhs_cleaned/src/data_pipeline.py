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
DATA_PATH = Path(os.getenv("KAMP_DATA_PATH", ROOT_DIR.parent / "data" / "okm_cleaned_2021.csv"))

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

FEATURE_SETS = {"base": {"observed": FEATURE_COLUMNS, "known": []}}

# Final held-out test period.
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

# 제조 이상(전력 피크) 기준: 팀 공통 고정값 177 kW.
# = Test 이전 전체 기간(1/1~8/15, 셧다운 제외) 전력의 상위 5% (peak_threshold_from_data() 로 재현).
# 모든 모델·모든 구간에 같은 값을 쓴다. Test 구간 피크 49시간(29건, 15일).
PEAK_THRESHOLD_KW = 177.0
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


def peak_threshold_from_data(df):
    """PEAK_THRESHOLD_KW(177 kW)의 근거: Test 이전 전체 기간(셧다운 제외) 전력의 상위 5%."""
    mask = (df["datetime"] < TEST_START) & ~df["공장_셧다운_여부"]
    return float(df.loc[mask, TARGET_COLUMN].quantile(PEAK_QUANTILE))


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

def build_features(df, feature_set="base"):
    """피처 조합에 맞는 입력 행렬과 컬럼 이름을 만든다.

    known 피처는 한 칸 앞당겨(shift -1) 붙인다. 행 i 에 i+1 시점 값이 들어가므로,
    X[t-L:t] 창의 마지막 칸에 예측할 시점 t 의 달력·조업 정보가 담긴다.
    """
    spec = FEATURE_SETS[feature_set]
    missing = [c for c in spec["observed"] + spec["known"] if c not in df.columns]
    if missing:
        raise KeyError(f"피처 조합 '{feature_set}'에 필요한 컬럼이 데이터에 없습니다: {missing}")

    observed = df[spec["observed"]].astype(np.float64)
    if observed.isnull().any().any():
        raise ValueError("observed 피처에 결측치가 있습니다.")
    frames = [observed]
    names = list(spec["observed"])
    if spec["known"]:
        # 과거값 파생의 맨 앞 몇 행(이전 값이 없는 행)은 0, 마지막 행(다음 시간이 없음)은 직전 값으로 채운다
        known = df[spec["known"]].astype(np.float64).fillna(0.0).shift(-1).ffill()
        frames.append(known)
        names += [f"{c}(t+1)" for c in spec["known"]]
    return pd.concat(frames, axis=1).to_numpy(dtype=np.float64), names


def scale_data(df, train_mask, feature_set="base"):
    """Scaler 는 학습 구간으로만 fit 하고, 전체에 transform 한다 (Leakage 방지)."""
    X, _ = build_features(df, feature_set)
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

def prepare_split(df, split, lookback=24, feature_set="base"):
    """하나의 Split 에 대해 학습/EarlyStopping/평가용 Sequence 와 Scaler, 피크 기준을 만든다.

    각 Sequence 는 Target 시점이 속한 구간으로 배정한다. 입력 window 는 항상 Target 이전
    시점의 측정값만 포함하므로, 평가 구간 Sequence 가 학습 구간 값을 입력으로 보는 것은 Leakage 가 아니다.
    """
    X_scaled, y_scaled, x_scaler, y_scaler = scale_data(df, split.train, feature_set)
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

    peak_threshold = PEAK_THRESHOLD_KW

    return {
        "split": split.name,
        "feature_set": feature_set,
        "X_train": X_train, "y_train": y_train, "idx_train": idx_train,
        "X_es": X_es, "y_es": y_es, "idx_es": idx_es,
        "X_eval": X_eval, "y_eval": y_eval, "idx_eval": idx_eval,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "peak_threshold": peak_threshold,
    }
