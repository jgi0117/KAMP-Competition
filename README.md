# OKM 전력 피크 예측: 기본 9개 변수 비교

이 브랜치는 KJH 정제 데이터의 9개 공통 입력 변수로 LSTM, TCN, LSTM·TCN 가중 앙상블, XGBoost, LightGBM의 seed 42 결과를 비교합니다. XGBoost와 LightGBM은 이 브랜치에서 전체 그리드를 다시 탐색하고 학습했습니다.

## 바로 확인

```powershell
python scripts/build_base9_comparison.py
python dashboard/app.py
```

대시보드 실행 전 `python -m pip install -r dashboard/requirements.txt`가 필요합니다. 대시보드는 저장된 Test 703시간을 재생하며 실시간 예측을 수행하지 않습니다.

결과보고서의 근거표, EDA·모델 평가 시각화 13개, HWPX와 검증을 한 명령으로 실행합니다. 저장된 대시보드 화면을 갱신할 때는 Playwright Chromium을 설치하고 `--capture-dashboard`를 붙입니다.

```powershell
python -m pip install -r requirements-submission.txt
python scripts/reproduce_submission.py
```

트리 모델의 108개 조합×3개 fold 학습까지 다시 실행하려면 `python scripts/reproduce_submission.py --train-trees --n-jobs 8`을 사용합니다. 대시보드 화면을 갱신하려면 `python -m playwright install chromium` 후 `--capture-dashboard`를 추가합니다. 신경망의 4단계 탐색과 평가 명령은 아래 재학습 절에 있습니다.

제출용 본문은 [심사기준별 상세내용](report/심사기준별_상세내용.md)에서 바로 검토할 수 있으며, [HWPX 작성본](report/OKM_경진대회_결과보고서_작성본.hwpx)에 같은 내용과 그림이 들어 있습니다. 그리드 탐색의 후보 수·범위·3개 fold 결과, 최종 후보 선택 기준과 수치, 오경보·미탐지 및 변수 영향 분석을 심사표의 6개 장에 맞춰 기록했습니다.

HWPX 작성본은 원본 양식의 장 구성과 필수 만족도 조사 항목을 유지합니다. 팀명·서명과 만족도 조사 완료 화면은 제출자가 채워야 합니다.

## 재학습

전처리를 확인하려면 `python scripts/verify_cleaned_data.py`를 실행합니다. 원본에서 다시 만든 6,168행×22열을 제공된 cleaned CSV와 전 열 비교합니다.

트리 모델은 Python 3.12에서 아래 패키지 버전으로 실행했습니다: xgboost 2.0.3, lightgbm 4.7.0, numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0. `python -m pip install -r requirements-tree.txt` 후 다음을 실행합니다.

```powershell
python scripts/train_base9_trees.py --models lightgbm xgboost --n-jobs 8
python scripts/build_base9_comparison.py
```

신경망 재학습 환경은 `neural/requirements.txt`를 사용합니다. 단계별 탐색은 아래 순서로 실행합니다. 각 단계의 완료 결과를 다음 단계가 읽습니다.

```powershell
foreach ($model in @('lstm', 'tcn')) {
  foreach ($stage in 1..4) {
    python neural/grid_search.py --model $model --stage $stage
  }
}
python neural/final_evaluate.py --models lstm tcn
```

신경망 재학습에는 상당한 시간이 필요합니다. seed 42 fold 결과와 최종 예측은 `neural/results/`에 보관되어 있으며, 재학습 명령은 `.keras` 모델 파일을 생성합니다.

## 제출 경로

| 경로 | 내용 |
|---|---|
| `data/okm_cleaned_2021.csv` | 학습에 사용한 KJH 정제 자료 |
| `data/okm_augumented_2021.csv` | 전처리 재현용 원본 |
| `src/preprocessing.py` | KJH 데이터 정제 코드 |
| `notebooks/kjh/` | KJH 대시보드 브랜치의 데이터 이해·전처리 EDA 노트북을 현재 자료 경로에 맞춰 정리 |
| `model_search/` | LSTM·TCN·XGBoost·LightGBM별 탐색 범위 |
| `neural/core/` | 9개 입력의 시간순 분할·평가와 LSTM·TCN 구현 |
| `neural/grid_search.py`, `neural/final_evaluate.py` | 신경망 탐색·최종 평가와 앙상블 |
| `scripts/train_base9_trees.py` | 두 트리 모델의 전체 그리드 탐색·평가 |
| `scripts/build_base9_comparison.py` | 다섯 후보 결과와 시각·정답 일치 검증 |
| `results/base9_tree/` | 전체 트리 그리드, 선택 모델, 예측, 지표 |
| `results/base9_comparison.csv` | 동일 Test 구간의 다섯 후보 비교 |
| `dashboard/` | 저장된 Test 예측 시각화 |
| `presentation/` | 전력 피크 예측 발표 PPT와 EDA·전처리 발표 PPT |
| `docs/` | 원본 HWPX 결과보고서 양식 |
| `report/` | 원본 양식을 유지한 HWPX 작성본과 삽입한 그래프·대시보드 스크린샷 |
| `scripts/reproduce_submission.py` | 정제 검증부터 보고서 파일 검증까지 실행 |

비교 설계와 수치는 [BASE9_COMPARISON.md](BASE9_COMPARISON.md)를 참조하세요.
