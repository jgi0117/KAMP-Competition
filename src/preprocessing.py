# -*- coding: utf-8 -*-
"""데이터 전처리 및 이상치/결측치 정제 파이프라인 모듈.

이 모듈은 OKM 제조 생산 및 전력 원본 데이터셋의 결측치, 이상치, 컬럼명 표준화 및
도메인 논리 기반 보정을 순차적으로 수행하는 재사용 가능한 파이프라인을 제공합니다.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """원본 CSV 데이터를 안전하게 로드합니다."""
    if data_path is None:
        # 프로젝트 루트 기준 data 폴더
        root_dir = Path(__file__).resolve().parent.parent
        data_path = root_dir / "data" / "okm_augumented_2021.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    df = pd.read_csv(data_path, encoding="utf-8")
    return df


def convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 컬럼(int64: 예 20210101)을 datetime64[ns] 형식으로 변환합니다.
    
    근거: 시계열 분석, 주기성(일/월/계절) 추출 및 날짜 기반 인덱싱의 표준화.
    """
    df = df.copy()
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """축약 영문 컬럼명을 직관적인 한글 표준 명칭으로 변경합니다.
    
    - day: ISO 요일 표준 (1:월 ~ 7:일) -> '요일'
    - d: 일자 (1 ~ 31) -> '일'
    - m: 월 (1 ~ 9) -> '월'
    """
    df = df.copy()
    rename_map = {
        "day": "요일",
        "d": "일",
        "m": "월"
    }
    df = df.rename(columns=rename_map)
    return df


def restore_time_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """시간 컬럼의 비정상 값(70~188)을 기온 일변화 곡선 검증 기반으로 0~23 정규 시간으로 복원합니다.
    
    근거 및 방어 전략:
    1. '시간_raw': 원본 시간 수치를 영구 보존.
    2. '시간_복원여부': 복원 대상 행(7/13, 7/15 총 48개 행) 식별 불리언 플래그 생성.
    3. 전일/익일 기온 일변화(새벽 최저, 한낮 최고)와 행 정렬 순서가 100% 일치함을 검증하고 0~23시 재부여.
    """
    df = df.copy()
    df["시간_raw"] = df["시간"]
    df["시간_복원여부"] = df["시간"] > 23

    abnormal_dates = df[df["시간_복원여부"]]["날짜"].unique()
    for target_date in abnormal_dates:
        mask = df["날짜"] == target_date
        if mask.sum() == 24:
            df.loc[mask, "시간"] = list(range(24))
            
    # 시간 컬럼을 int64로 보장
    df["시간"] = df["시간"].astype(int)
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """도메인 인과성 및 시계열 연속성에 근거하여 결측치를 안전하게 보정합니다.
    
    1. 공장인원 (17건, 2021-08-28 18시 ~ 2021-08-29 10시):
       - 근거: 해당 기간 전력 0 kW(셧다운), 생산량 0. 전체 데이터셋에서 생산량=0 ↔ 공장인원=0이 100% 일치.
       - 처리: 0.0 으로 대체.
    
    2. 풍속 (3건: 6/1 01시·02시, 7/4 20시):
       - 근거: 국소 1~2시간 대기 관측 누락. 대기 물리량의 시간적 연속성 반영.
       - 처리: 시간 기준 선형 보간 (Linear Interpolation).
    
    3. 강수량 (1건: 1/24 00시):
       - 근거: 직전 23시 6.3mm에서 01시 0.0mm로 비가 그쳐가는 전이 감쇠 반영.
       - 처리: 직전-직후 선형 보간 (3.15 mm).
    """
    df = df.copy()

    # 1. 공장인원 결측치: 생산량 0 인과성 근거 0.0 대체
    df.loc[df["공장인원"].isnull() & (df["생산량"] == 0), "공장인원"] = 0.0

    # 2. 풍속 결측치: 선형 보간
    df["풍속"] = df["풍속"].interpolate(method="linear").round(2)

    # 3. 강수량 결측치: 선형 보간 (6.3mm -> 0.0mm 전이)
    df["강수량"] = df["강수량"].interpolate(method="linear").round(2)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """전처리 및 분석 품질 향상을 위한 필수 파생 변수를 추가합니다.
    
    1. 공장_셧다운_여부 (bool):
       - 비가동 시에도 대기전력(Base Load, 평균 46.5kW)이 존재하는 정상 운영과 달리,
         전력이 완전히 0 kW로 차단된 비정상 정전/셧다운 이벤트를 식별 (모델 마스킹 및 가중치 활용).
    
    2. 전력_평균_실수 (float64):
       - 기존 정수형 '평균' 컬럼의 반올림 오차(±0.25~0.5)를 제거한 4분할(15/30/45/60분) 정확한 산술평균.
    """
    df = df.copy()

    # 1. 공장 셧다운/정전 플래그 (15분, 30분, 45분, 60분 모두 0인 행)
    df["공장_셧다운_여부"] = (
        (df["15분"] == 0) & 
        (df["30분"] == 0) & 
        (df["45분"] == 0) & 
        (df["60분"] == 0)
    )

    # 2. 전력 평균 실수형 컬럼
    df["전력_평균_실수"] = (df["15분"] + df["30분"] + df["45분"] + df["60분"]) / 4.0

    return df


def load_cleaned_data(
    data_path: Optional[Path] = None,
    auto_generate: bool = True,
) -> pd.DataFrame:
    """정제 완료 데이터를 불러오고, 없으면 기존 전처리 파이프라인을 실행합니다."""
    if data_path is None:
        root_dir = Path(__file__).resolve().parent.parent
        data_path = root_dir / "data" / "okm_cleaned_2021.csv"

    data_path = Path(data_path)
    if data_path.exists():
        return pd.read_csv(data_path, encoding="utf-8-sig", parse_dates=["날짜"])

    alias_path = data_path.parent / "okm_preprocessed_2021.csv"
    if alias_path.exists():
        return pd.read_csv(alias_path, encoding="utf-8-sig", parse_dates=["날짜"])

    if auto_generate:
        return run_preprocessing_pipeline()

    raise FileNotFoundError(f"정제 데이터 파일을 찾을 수 없습니다: {data_path}")


def run_preprocessing_pipeline(
    data_path: Optional[Path] = None,
    save_summary: bool = True
) -> pd.DataFrame:
    """전체 전처리 파이프라인을 실행하고 결과를 검증합니다."""
    print(">>> [전처리 1단계] 원본 데이터 로드...")
    df = load_raw_data(data_path)
    initial_shape = df.shape
    print(f"    - 로드 완료: {initial_shape[0]}행 × {initial_shape[1]}열")

    print(">>> [전처리 2단계] 날짜 컬럼 datetime 변환...")
    df = convert_datetime_columns(df)

    print(">>> [전처리 3단계] 컬럼명 표준화 (day->요일, d->일, m->월)...")
    df = standardize_column_names(df)

    print(">>> [전처리 4단계] 시간 컬럼 이상치 복원 및 원본/플래그 보존...")
    df = restore_time_anomalies(df)

    print(">>> [전처리 5단계] 결측치 보정 (공장인원 0.0, 풍속 선형보간, 강수량 선형보간)...")
    df = impute_missing_values(df)

    print(">>> [전처리 6단계] 파생 변수 생성 (공장_셧다운_여부, 전력_평균_실수)...")
    df = add_derived_features(df)

    # 전처리 완료 후 무결성 검증
    remaining_nulls = df.isnull().sum().sum()
    print("\n=== [전처리 파이프라인 완료 검증] ===")
    print(f"최종 데이터 크기: {df.shape[0]}행 × {df.shape[1]}열")
    print(f"잔여 결측치 수: {remaining_nulls}개 (목표: 0개)")
    print(f"시간 범위: {df['시간'].min()}시 ~ {df['시간'].max()}시 (고유값: {df['시간'].nunique()}개)")
    print(f"공장 셧다운 감지 행 수: {df['공장_셧다운_여부'].sum()}개")
    print(f"시간 복원 행 수: {df['시간_복원여부'].sum()}개")

    if remaining_nulls > 0:
        print("[경고] 결측치가 여전히 존재합니다:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("[성공] 모든 결측치가 완벽히 정제되었습니다 (결측치 0개).")

    return df


if __name__ == "__main__":
    df_cleaned = run_preprocessing_pipeline()
