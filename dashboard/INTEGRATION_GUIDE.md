# Dashboard 데이터·모델 통합 기록

대시보드 레이아웃·스타일·탐색 구조와 인터랙션은 `kjh_dashboard` 구현을 유지하고, 데이터·모델 지표는 `submission-clean`의 저장 산출물을 단일 진실 공급원으로 연결했다.

현재 데이터 흐름은 다음과 같다.

```text
results/model_comparison.csv ───────────────┐
neural/results/final/test_predictions.csv ──┤
neural/results/final/alert_settings.csv ─────┤
docs/report/evidence/*.csv ──────────────────┘
                         ↓
                  dashboard/data.py
                         ↓
             layouts.py + app.py callbacks
```

- `data.py`는 경로 해석, 필수 컬럼 검증, 요약 지표, 차트와 FN·FP 사례를 담당한다.
- Test 예측은 2021-08-16 00:00부터 2021-09-14 23:00까지 703시간이다.
- 재생은 저장 결과를 한 시간씩 보여 주며 실시간 추론이 아니다.
- 홈과 기여·점검 화면은 저장된 XGBoost·LightGBM의 네이티브 TreeSHAP을 168시간 lag에서 변수별로 합산하며 재생 시각마다 갱신한다.
- 보고서의 `grouped_permutation_importance.csv`는 전체 Test 구간의 전역 검증 근거로 별도 유지한다.
- `mock_data.py`는 과거 프로토타입 추적용으로만 남아 있으며 앱에서 import하지 않는다.

검증은 `python scripts/verify_dashboard.py`로 수행한다. 전체 제출물 재현은 `python scripts/reproduce_submission.py`를 사용한다.

아래 내용은 운영 스트리밍 전환 시 참고할 수 있도록 남긴 초기 설계 기록이며, 현재 저장 Test 산출물의 계약보다 우선하지 않는다.

## 부록 1. 초기 운영 연동 설계

| 파일 | 역할 | 실제 연동 시 변경 여부 |
| --- | --- | --- |
| `app.py` | 앱 초기화, 라우팅 및 14개 콜백 | 데이터 갱신·추론 콜백 추가 시 변경 |
| `layouts.py` | 4개 화면과 UI 컴포넌트 | KPI를 동적 컴포넌트로 바꿀 때 변경 |
| `data.py` | 저장 Test 산출물 로딩·검증·차트 | 운영 입력 전환 시 어댑터 교체 |
| `mock_data.py` | 과거 프로토타입 참고 코드 | 현재 앱에서 사용하지 않음 |
| `assets/styles.css` | 다크 산업용 관제 UI와 반응형 스타일 | 데이터 연동과 무관 |
| `design_system.md` | 색상·상태·차트·반응형 기준 | UI 변경 시 준수 |

초기 프로토타입의 데이터 흐름은 다음과 같았다.

```text
mock_data.py -> layouts.py의 초기 화면/차트
             -> app.py의 범위 선택·시점 선택 콜백
             -> 브라우저 session Store의 조치 기록
```

운영 연동 시에는 `mock_data.py`를 직접 확장하기보다 `data_provider.py` 같은 어댑터 모듈을 추가하고,
레이아웃과 콜백이 어댑터의 안정적인 반환 스키마만 사용하도록 구성하는 것을 권장한다.

## 2. 화면이 요구하는 데이터 계약

### 전력 시계열

`energy_frame()`과 `energy_figure()`가 사용하는 최소 스키마다.

| 컬럼 | 타입 | 의미 |
| --- | --- | --- |
| `timestamp` | timezone이 명시된 datetime | 15분 단위 관측/예측 시각 |
| `actual_kw` | float, nullable | 계측 전력. 미래 구간은 null |
| `predicted_kw` | float, nullable | 과거 또는 현재 시점의 모델 예측 |
| `future_kw` | float, nullable | 현재 이후 1시간 예측. 과거 구간은 null |

- 시간순 오름차순, 시각 중복 없음, 기본 간격 15분을 보장한다.
- 현재 UI는 미래 4개 시점, 즉 1시간 horizon을 전제로 한다.
- 학습 시 사용한 목표값의 단위와 화면 단위 `kW`가 같아야 한다. 다르면 어댑터에서 변환한다.
- 한국 운영 데이터라면 저장은 UTC, 화면 변환은 `Asia/Seoul`로 통일하는 것을 권장한다.
- 결측치는 0으로 치환하지 말고 null로 유지한 뒤 품질 상태를 별도로 표시한다.

### 예측 요약 KPI

현재 `layouts.py`의 `hero_kpi_card()`와 위험 배너에 값이 하드코딩되어 있다. 실제 연동 시 다음 값을 한 객체로 계산해 전달한다.

```python
{
    "as_of": "2026-10-07T14:30:00+09:00",
    "current_kw": 148.2,
    "forecast_max_kw": 176.8,
    "forecast_mean_kw": 158.4,
    "peak_at": "2026-10-07T15:15:00+09:00",
    "threshold_kw": 160.0,
    "excess_kw": 16.8,
    "risk_probability": 0.82,
    "risk_level": "danger"
}
```

`risk_level`은 최소 `normal`, `caution`, `danger` 중 하나로 제한한다. 임계값 산정 방식(고정 계약전력,
최근 7일 상위 5% 등)은 모델과 분리해 설정값으로 관리한다.

### 기여 요인/SHAP

현재 `CAUSES`와 `cause_figure()`가 기대하는 논리 스키마다.

| 컬럼 | 타입 | 의미 |
| --- | --- | --- |
| `timestamp` | datetime | 설명 대상 예측 시각 |
| `feature` | string | 사용자에게 표시할 변수명 |
| `contribution_kw` | float | 예측값을 높이거나 낮춘 기여량 |
| `feature_value` | optional | 해당 시점의 원래 변수값 |
| `rank` | integer | 절댓값 기준 우선순위 |

- 트리 모델은 SHAP 값을 사용할 수 있지만, 변환된 feature 이름을 현장 용어로 매핑해야 한다.
- 기여도는 인과관계가 아니므로 화면 문구도 계속 `기여 요인`으로 유지한다.
- 모델 출력이 표준화된 스케일이면 원래 단위 또는 예측 단위로 역변환한 뒤 표시한다.

### 모델 평가 지표

`model_page()`와 `model_comparison_figure()`의 실제 값으로 교체할 항목은 다음과 같다.

- 후보별 test RMSE
- 최종 모델의 RMSE, MAE, R²
- 피크 분류를 운영할 경우 Precision, Recall, F1, FP, FN
- 데이터셋 버전, 학습 기간, test 기간, 모델 artifact 버전

데이터 누수를 막기 위해 시간 순서 기반 train/validation/test 분할 결과만 표시한다. 화면의 현재
`Top-3 GBDT`, RMSE `9.34`, MAE `6.13`, R² `0.9730`은 보고서 기반 참고값이며,
최종 팀 모델의 재현 가능한 평가 산출물로 교체해야 한다. 분류 지표와 오류 히트맵은 전부 시뮬레이션이다.

### 조치 이력

현재 `dcc.Store(storage_type="session")`에 다음 키를 가진 dict 목록으로 저장한다.

```python
{"시각": str, "단계": str, "전력": str, "담당자": str, "조치": str, "상태": str}
```

운영 시에는 표시용 한글 키 대신 내부 영문 스키마와 DB 기본키를 사용하고, UI 직전에 표시 형식으로 변환한다.
현재 메모 입력값은 저장되지 않으며 완료 처리, 필터, CSV 내보내기도 연결되지 않았다.

## 3. 권장 연동 순서

1. 최종 학습 데이터의 시간대, 15분 grain, 목표 컬럼과 단위를 확정한다.
2. 모델 artifact와 전처리 pipeline을 함께 저장해 학습/추론 feature 순서를 고정한다.
3. `data_provider.py`에 데이터 로딩과 위 계약을 만족하는 반환 함수를 구현한다.
4. `mock_data.py`의 `energy_frame`, `CAUSES`, 모델 지표를 provider 호출로 교체한다.
5. `hero_kpi_card()`와 위험 배너를 고정 문자열이 아닌 provider 결과로 렌더링한다.
6. `dcc.Interval` 콜백에서 최신 데이터 갱신과 추론을 수행하되, 긴 추론은 별도 서비스나 캐시로 분리한다.
7. 데이터 없음, 지연, 스키마 오류, 추론 실패 상태를 정상 0값과 구분해 표시한다.
8. 실제 분류 검증 전까지 `SIMULATION` 표기와 가짜 분류 지표를 제거하지 않는다.

권장 provider 인터페이스 예시는 다음과 같다.

```python
def load_power_series(as_of, lookback_hours: int, horizon_steps: int = 4) -> pd.DataFrame: ...
def load_forecast_summary(as_of) -> dict: ...
def load_contributions(forecast_at, top_k: int = 5) -> pd.DataFrame: ...
def load_model_metrics() -> dict: ...
```

## 4. 모델 artifact와 설정

- 대용량 모델과 원천 데이터는 Git에 직접 커밋하지 않는다.
- 경로, DB 접속값, API 주소는 환경변수로 주입하고 `.env`는 커밋하지 않는다.
- 모델과 함께 전처리기, feature 목록, 학습 데이터 버전, 라이브러리 버전, 평가 JSON을 보관한다.
- 로드 시 artifact 버전과 입력 스키마를 검증하고 불일치하면 추론하지 않는다.
- 개발 모드에서는 mock provider, 운영 모드에서는 real provider를 선택할 수 있게 한다.

예시 환경변수 이름:

```text
DASHBOARD_DATA_MODE=mock
MODEL_ARTIFACT_PATH=artifacts/model.joblib
MODEL_METADATA_PATH=artifacts/model_metadata.json
POWER_DATA_PATH=data/inference_input.parquet
```

## 5. 완료 검증 체크리스트

- `python dashboard/app.py`로 4개 화면이 모두 열린다.
- 6/12/24시간 전환 후 실제값, 과거 예측, 미래 예측이 올바르게 분리된다.
- 현재 시각과 예측 horizon이 데이터 기준 시각과 일치한다.
- 차트 시점 선택 시 같은 시점의 SHAP/기여도가 표시된다.
- KPI, 위험 배너, 차트, 모델 성능표가 같은 모델 출력에서 파생된다.
- 결측·지연·중복·시간대 오류를 의도적으로 넣었을 때 오류 상태로 표시된다.
- 학습 데이터에 test 기간의 미래 정보가 포함되지 않았음을 확인한다.
- 새 환경에서 `environment.yml` 또는 `dashboard/requirements.txt`로 재현된다.
- 브라우저 폭 1280px와 1920px에서 주요 수치와 표가 잘리지 않는다.

## 6. 현재 알려진 경계

- UI의 기준 시각은 `REFERENCE_TIME = 2024-10-15 14:30`으로 고정되어 있다.
- 재생 버튼은 실제 센서를 읽지 않고 5초마다 표시 시각만 15분 증가시킨다.
- 홈 KPI, 위험 배너, 기여·점검 표, 모델 카드 대부분은 레이아웃 내부 고정 문자열이다.
- 피크 확률, 분류 지표, 오류 히트맵, 경보와 조치 이력은 시뮬레이션이다.
- 영구 저장소, 인증/권한, API 재시도, 캐시, 로깅과 모니터링은 아직 없다.
