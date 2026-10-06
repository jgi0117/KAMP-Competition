# -*- coding: utf-8 -*-
"""피처 엔지니어링(Feature Engineering) 파이프라인 모듈.

이 모듈은 정제된 데이터셋(data/okm_cleaned_2021.csv)을 바탕으로
데이터 누수(Data Leakage), 과적합(Overfitting), 과소적합(Underfitting)을
원천 방어하는 전력사용량 예측용 특성(Features)을 생성하고 관리합니다.

팀 공통 분할 기준(Train: 1~6월, Validation: 7~8/15, Test: 8/16~9/14)을 지원합니다.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 등록
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from src.config import LOCAL_DATA_DIR, ROOT_DIR
    from src.preprocessing import load_cleaned_data
except ImportError:
    LOCAL_DATA_DIR = _ROOT_DIR / "data"
    ROOT_DIR = _ROOT_DIR
    from src.preprocessing import load_cleaned_data

# 공휴일 정의 (2021년 대한민국 법정공휴일)
HOLIDAYS_2021 = [
    "2021-01-01",  # 신정
    "2021-02-11", "2021-02-12", "2021-02-13",  # 설날 연휴
    "2021-03-01",  # 삼일절
    "2021-05-05",  # 어린이날
    "2021-05-19",  # 부처님오신날
    "2021-06-06",  # 현충일
    "2021-08-15",  # 광복절
    "2021-08-16",  # 광복절 대체공휴일
]

# 원본 및 정제 컬럼 중 모델 입력(X)에서 누수 방지를 위해 배제해야 하는 Red Flag 목록
LEAKAGE_COLUMNS = [
    "15분", "30분", "45분", "60분", "평균", "전력_평균_실수",  # 타겟 구성요소 (100% Target Leakage)
    "공장_셧다운_여부",  # 타겟=0 정의를 포함하는 정제 플래그
    "시간_raw", "시간_복원여부",  # 데이터 정제 메타데이터
    "생산량", "공장인원",  # 예측 시점 t의 사후 실적치 (lag1으로 대체)
    "날짜", "datetime",  # 비수치형 시계열 메타데이터
]


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 및 캘린더 주기성 피처를 생성합니다. (누수 위험 0%)"""
    df = df.copy()

    # 24시간 원형 순환성 (23시-00시 경계 단절 해결)
    df["시간_sin"] = np.sin(2 * np.pi * df["시간"] / 24.0)
    df["시간_cos"] = np.cos(2 * np.pi * df["시간"] / 24.0)

    # 연간 계절 순환성
    df["월_sin"] = np.sin(2 * np.pi * df["월"] / 12.0)
    df["월_cos"] = np.cos(2 * np.pi * df["월"] / 12.0)

    # 주간 7일 순환성
    df["요일_sin"] = np.sin(2 * np.pi * (df["요일"] - 1) / 7.0)
    df["요일_cos"] = np.cos(2 * np.pi * (df["요일"] - 1) / 7.0)

    # 주말 및 점심시간 급감(Dip: -37.8%) 플래그
    df["주말여부"] = df["요일"].isin([6, 7]).astype(int)
    df["점심시간여부"] = (df["시간"] == 12).astype(int)

    # 조업시간대 범주 (0:심야, 1:오전, 2:점심, 3:오후, 4:야간)
    def _classify_period(h: int) -> int:
        if h <= 6:
            return 0
        elif h <= 11:
            return 1
        elif h == 12:
            return 2
        elif h <= 17:
            return 3
        else:
            return 4
    df["조업시간대"] = df["시간"].apply(_classify_period)

    # 공휴일 및 실제 정상 영업일
    df["공휴일여부"] = df["날짜"].astype(str).isin(HOLIDAYS_2021).astype(int)
    df["영업일여부"] = ((df["주말여부"] == 0) & (df["공휴일여부"] == 0)).astype(int)

    # 당일 잔여 시간 (24 - 시간: 일일 누적 생산량과 결합하여 잔업/조기정리 스케줄링 의사결정 포착)
    df["일일잔여시간"] = 24 - df["시간"]

    return df


def create_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """공장 조업 일정 및 특수 피크 이벤트를 포착하는 피처를 생성합니다."""
    df = df.copy()

    # 월요일 08시 기동 부하 (+22.0 kW 예열 피크)
    df["월요일기동시간여부"] = ((df["요일"] == 1) & (df["시간"] == 8)).astype(int)

    # 주간 집중 조업 블록 (영업일 08~11시, 13~17시 풀가동 영역)
    work_hours = [8, 9, 10, 11, 13, 14, 15, 16, 17]
    df["조업집중시간여부"] = ((df["영업일여부"] == 1) & df["시간"].isin(work_hours)).astype(int)

    # 주차 (Week of Year)
    if "datetime" in df.columns:
        df["주차"] = df["datetime"].dt.isocalendar().week.astype(int)
    else:
        dt = pd.to_datetime(df["날짜"].astype(str) + " " + df["시간"].astype(str) + ":00:00")
        df["주차"] = dt.dt.isocalendar().week.astype(int)

    return df


def create_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    """시계열 자기상관 및 동역학 피처를 생성합니다. (Strict Causality: shift(1) 강제)"""
    df = df.copy()
    target_series = df["전력_평균_실수"]

    # 초단기 관성 (r = 0.9009)
    df["전력_lag1"] = target_series.shift(1)

    # 모멘텀/가속도 (lag1과 상관계수 r=0.223으로 다중공선성 해소 직교 피처)
    df["전력_diff1"] = target_series.shift(1) - target_series.shift(2)

    # 전일 동시간대 대비 증감량 (t-1 vs t-25: 일간 24시간 주기 변화 직교화 포착)
    df["전력_전일동시간_diff"] = target_series.shift(1) - target_series.shift(25)

    # 일간 24시간 주기 (r = 0.3749)
    df["전력_lag24"] = target_series.shift(24)

    # 주간 7일 조업 주기 (r = 0.7397)
    df["전력_lag168"] = target_series.shift(168)

    # 24시간 롤링 기저 추세 및 변동성 (t-1 기준)
    df["전력_rolling_mean_24h"] = target_series.shift(1).rolling(24, min_periods=1).mean()
    df["전력_rolling_std_24h"] = target_series.shift(1).rolling(24, min_periods=1).std().fillna(0.0)

    # 3시간 초단기 조업 모멘텀 이동평균 (t-1 기준: 직전 3시간 평균 조업 강도 포착, 1시간 노이즈 완충)
    df["전력_rolling_mean_3h"] = target_series.shift(1).rolling(3, min_periods=1).mean()

    # 영업일 13시 점심 복귀 반등 보정량 (t-2 vs t-1: 12시 점심 급감량만큼 13시 복귀 추정, 12시 lag1 왜곡 방지)
    df["점심반등예상량"] = np.where(
        (df["시간"] == 13) & (df["영업일여부"] == 1),
        np.maximum(target_series.shift(2) - target_series.shift(1), 0.0),
        0.0
    )

    return df


def create_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """생산 상태 및 효율 피처를 생성합니다. (사후 측정치 누수 방지: shift(1) 강제)"""
    df = df.copy()

    # Bimodal 작동 모드 판별 (대기전력 46.1 kW vs 가동 129.0 kW)
    df["가동상태_lag1"] = (df["생산량"].shift(1) > 0).astype(int)

    # 정지 후 재가동 시동 부하 (118.8 kW 튐 현상)
    df["설비기동여부"] = ((df["생산량"].shift(2) == 0) & (df["생산량"].shift(1) > 0)).astype(int)

    # 직전 생산량 및 증감 속도
    df["생산량_lag1"] = df["생산량"].shift(1)
    df["생산량_diff1"] = df["생산량"].shift(1) - df["생산량"].shift(2)

    # 인당 생산성 (조업 밀도)
    workers_lag1 = df["공장인원"].shift(1)
    df["인당생산량_lag1"] = df["생산량_lag1"] / (workers_lag1 + 0.1)

    # 주말 또는 공휴일인데도 직전 시간에 생산량이 발생한 특근 조업 식별 플래그 (t-1 생산 실적 기반)
    df["휴일특근조업_lag1"] = ((df["영업일여부"] == 0) & (df["생산량"].shift(1) > 0)).astype(int)

    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """외기 열역학 및 공조 부하 피처를 생성합니다. (U자형 비선형성 선형 분리)"""
    df = df.copy()
    temp = df["기온"]
    humidity = df["습도"]
    wind = df["풍속"]

    # 냉방도일 (CDD, 22도 초과 고온 냉방 부하)
    df["냉방도일_CDD"] = np.maximum(temp - 22.0, 0.0)

    # 난방도일 (HDD, 15도 미만 저온 난방 부하)
    df["난방도일_HDD"] = np.maximum(15.0 - temp, 0.0)

    # 불쾌지수 (DI, 온·습도 복합 체감 지표)
    df["불쾌지수_DI"] = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3

    # 체감온도 (풍속 냉각 효과)
    wind_safe = np.maximum(wind, 0.1)
    df["체감온도"] = 13.12 + 0.6215 * temp - 11.37 * np.power(wind_safe, 0.16) + 0.3965 * temp * np.power(wind_safe, 0.16)

    # 외기 온도 단기 변화율 (공조기 가동 트리거)
    df["기온_변화량"] = temp - temp.shift(1).fillna(temp)

    # 건물 축열 및 열용량 지연 효과 (외기 온도의 6시간 지수이동평균)
    df["기온_축열_ema_6h"] = temp.ewm(span=6).mean()

    # 대기 절대 수증기 분압 (hPa, Tetens 공식 기반 순수 대기 수분량 측정)
    svp = 6.1078 * np.exp((17.27 * temp) / (temp + 237.3))
    df["수증기압"] = svp * (humidity / 100.0)

    # 냉방 잠열 부하 지수 (CDD 현열 * 수증기 분압 잠열 가중치)
    df["냉방잠열부하"] = df["냉방도일_CDD"] * (1.0 + df["수증기압"] / 20.0)

    return df


def create_cumulative_work_features(df: pd.DataFrame) -> pd.DataFrame:
    """일일 누적 조업 및 설비 연속 가동 피처를 생성합니다. (Strict Causality: shift(1) 강제)"""
    df = df.copy()

    # 1. 당일 누적 전력 사용량 (00시부터 t-1시까지 누적, 당일 에너지 소비 페이스 및 기저 부하 포착)
    df["일일누적전력_lag1"] = df.groupby("날짜")["전력_평균_실수"].transform(
        lambda s: s.shift(1).fillna(0.0).cumsum()
    )

    # 2. 당일 누적 생산 실적 (00시부터 t-1시까지 누적, 일일 생산 목표 진도율 포착)
    df["일일누적생산량_lag1"] = df.groupby("날짜")["생산량"].transform(
        lambda s: s.shift(1).fillna(0.0).cumsum()
    )

    # 3. 연속 조업 시간 (기계가 쉬지 않고 가동된 연속 시간, 설비 발열 및 모터 부하 포착)
    is_operating = (df["생산량"].shift(1) > 0).astype(int)
    consec = []
    cur = 0
    for val in is_operating:
        if val == 1:
            cur += 1
        else:
            cur = 0
        consec.append(cur)
    df["연속조업시간_lag1"] = consec

    return df


def build_feature_pipeline(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """전체 피처 엔지니어링 파이프라인을 순차 실행하여 특성 데이터프레임을 생성합니다."""
    if df is None:
        df = load_cleaned_data()

    df = df.copy()

    # 시간 인덱스 생성
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["날짜"].astype(str) + " " + df["시간"].astype(str) + ":00:00")

    print(">>> [피처 생성 1단계] 캘린더 및 주기성 피처 생성...")
    df = create_calendar_features(df)

    print(">>> [피처 생성 2단계] 조업 스케줄 및 피크 이벤트 피처 생성...")
    df = create_schedule_features(df)

    print(">>> [피처 생성 3단계] 시계열 지연(Autoregressive) 피처 생성 (Strict Causality)...")
    df = create_autoregressive_features(df)

    print(">>> [피처 생성 4단계] 생산 상태 및 동역학 피처 생성 (Lag 1 기반)...")
    df = create_production_features(df)

    print(">>> [피처 생성 5단계] 외기 열역학 및 공조 부하 피처 생성 (CDD, HDD, DI, 축열 EMA)...")
    df = create_weather_features(df)

    print(">>> [피처 생성 6단계] 일일 누적 조업 및 설비 연속 가동 피처 생성 (Strict Causality)...")
    df = create_cumulative_work_features(df)

    print(f"\n[피처 파이프라인 완료] 생성 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "전력_평균_실수"
) -> Tuple[pd.DataFrame, pd.Series]:
    """모델 학습용 피처셋(X)과 타겟(y)을 분리하며, 데이터 누수 변수를 원천 차단합니다."""
    drop_cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2021-06-30 23:00:00",
    val_end: str = "2021-08-15 23:00:00",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """팀 공통 기준에 따라 Train(1~6월), Validation(7~8/15), Test(8/16~9/14)로 분할합니다.

    Args:
        df: 피처가 포함된 데이터프레임
        train_end: Train 종료 시점 (기본값: '2021-06-30 23:00:00')
        val_end: Validation 종료 시점 (기본값: '2021-08-15 23:00:00')

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (df_train, df_val, df_test)
    """
    if "datetime" not in df.columns:
        dt = pd.to_datetime(df["날짜"].astype(str) + " " + df["시간"].astype(str) + ":00:00")
    else:
        dt = df["datetime"]

    train_mask = (dt <= train_end)
    val_mask = (dt > train_end) & (dt <= val_end)
    test_mask = (dt > val_end)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    print("\n=== [팀 공통 시계열 분할 현황] ===")
    print(f"Train (학습용):      {len(df_train)}행 ({df_train['날짜'].min()} ~ {df_train['날짜'].max()})")
    print(f"Validation (검증용): {len(df_val)}행 ({df_val['날짜'].min()} ~ {df_val['날짜'].max()})")
    print(f"Test (최종평가용):   {len(df_test)}행 ({df_test['날짜'].min()} ~ {df_test['날짜'].max()})")
    print(f"합계 무결성:        {len(df_train) + len(df_val) + len(df_test)} == {len(df)}행 (100% 일치)")

    return df_train, df_val, df_test


def save_features_dataset(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    encoding: str = "utf-8-sig"
) -> Path:
    """피처 엔지니어링이 완료된 데이터셋을 CSV로 저장합니다."""
    if output_path is None:
        output_path = LOCAL_DATA_DIR / "okm_features_2021.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"\n[저장 완료] 피처 데이터셋: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


if __name__ == "__main__":
    df_feat = build_feature_pipeline()
    save_features_dataset(df_feat)
    tr, va, te = split_train_val_test(df_feat)
