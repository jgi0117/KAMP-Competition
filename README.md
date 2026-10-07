# OKM 전력 피크 예측: 기본 9개 변수 비교

이 브랜치는 KJH 정제 데이터만 사용해 LSTM, TCN, LSTM·TCN 가중 앙상블, XGBoost, LightGBM의 seed 42 결과를 비교합니다. XGBoost와 LightGBM만 새로 학습했습니다. 피처 엔지니어링 실험은 포함하지 않습니다.

## 바로 확인

```powershell
python scripts/build_base9_comparison.py
python dashboard/app.py
```

대시보드 실행 전 `python -m pip install -r dashboard/requirements.txt`가 필요합니다. 대시보드는 저장된 Test 703시간을 재생하며 실시간 예측을 수행하지 않습니다.

결과보고서의 시각화와 HWPX는 아래 순서로 다시 생성합니다. 대시보드 캡처는 실제 Dash 화면을 Playwright로 저장하며 Chrome이 없으면 Playwright Chromium을 사용합니다.

```powershell
python -m pip install -r requirements-report.txt
python -m playwright install chromium
python scripts/make_report_figures.py
python scripts/capture_dashboard.py
python scripts/fill_result_report.py
python scripts/verify_report.py
```

HWPX 작성본은 원본 양식의 장 구성과 필수 만족도 조사 항목을 유지합니다. 팀명·서명과 만족도 조사 완료 화면은 제출자가 채워야 합니다.

## 재학습

전처리를 확인하려면 `python scripts/verify_cleaned_data.py`를 실행합니다. 원본에서 다시 만든 6,168행×22열을 제공된 cleaned CSV와 전 열 비교합니다.

트리 모델은 Python 3.12에서 아래 패키지 버전으로 실행했습니다: xgboost 2.0.3, lightgbm 4.7.0, numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0. `python -m pip install -r requirements-tree.txt` 후 다음을 실행합니다.

```powershell
python scripts/train_base9_trees.py --models lightgbm xgboost --n-jobs 8
python scripts/build_base9_comparison.py
```

신경망 재학습 환경은 `lhs_cleaned/requirements.txt`를 사용합니다. 단계별 탐색은 아래 순서로 실행합니다. 각 단계의 완료 결과를 다음 단계가 읽습니다.

```powershell
foreach ($model in @('lstm', 'tcn')) {
  foreach ($stage in 1..4) {
    python lhs_cleaned/grid_search.py --model $model --stage $stage
  }
}
python lhs_cleaned/final_evaluate.py --selection seed42 --seeds 42 --models lstm tcn
```

신경망 재학습에는 상당한 시간이 필요하며 기존 seed 42 fold 결과와 최종 예측은 `lhs_cleaned/results/`에 보존되어 있습니다. 신경망 가중치 파일은 보존되지 않아 새 입력의 즉시 추론은 지원하지 않습니다.

## 제출 경로

| 경로 | 내용 |
|---|---|
| `data/okm_cleaned_2021.csv` | 학습에 사용한 KJH 정제 자료 |
| `data/okm_augumented_2021.csv` | 전처리 재현용 원본; 학습에 미사용 |
| `src/preprocessing.py` | KJH 데이터 정제 코드 |
| `model_search/` | LSTM·TCN·XGBoost·LightGBM별 탐색 범위 |
| `lhs_cleaned/grid_search.py`, `final_evaluate.py` | 신경망 탐색·최종 평가와 앙상블 |
| `scripts/train_base9_trees.py` | 두 트리 모델의 전체 그리드 탐색·평가 |
| `scripts/build_base9_comparison.py` | 다섯 후보 결과와 시각·정답 일치 검증 |
| `results/base9_tree/` | 전체 트리 그리드, 선택 모델, 예측, 지표 |
| `results/base9_comparison.csv` | 동일 Test 구간의 다섯 후보 비교 |
| `dashboard/` | 저장된 Test 예측 시각화 |
| `presentation/` | 원본 발표 PPT |
| `docs/` | 원본 HWPX 결과보고서 양식 |
| `report/` | 원본 양식을 유지한 HWPX 작성본과 삽입한 그래프·대시보드 스크린샷 |

비교 설계와 수치는 [BASE9_COMPARISON.md](BASE9_COMPARISON.md)를 참조하세요.
