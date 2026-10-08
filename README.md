# OKM 전력 피크 예측: 5개 모델 성능 비교

이 브랜치는 KJH 정제 데이터의 9개 공통 입력 변수로 LSTM, TCN, LSTM·TCN 가중 앙상블, XGBoost, LightGBM의 seed 42 결과를 비교합니다. XGBoost와 LightGBM은 이 브랜치에서 전체 그리드를 다시 탐색하고 학습했습니다.

## 바로 확인

| 목적 | 실행 명령 | 주요 산출물 |
|---|---|---|
| 대시보드 확인 | `python dashboard/app.py` | `http://127.0.0.1:8050` |
| 5개 모델 비교 갱신 | `python scripts/build_model_comparison.py` | `results/model_comparison.csv` |
| 트리 모델 재학습 | `python scripts/train_tree_models.py --models lightgbm xgboost --n-jobs 8` | `results/tree_models/` |
| 보고서 일괄 재생성 | `python scripts/reproduce_submission.py` | `docs/report/`의 근거표·그림·HWPX |
| 재학습·화면 캡처 포함 전체 재현 | `python scripts/reproduce_submission.py --train-trees --n-jobs 8 --capture-dashboard` | 모델·대시보드 캡처·최종 보고서 |

최초 실행 전 루트에서 `python -m pip install -r requirements.txt`로 공통 환경을 설치합니다. 대시보드는 저장된 Test 703시간을 재생하며 실시간 예측을 수행하지 않습니다. 실행 후 `http://127.0.0.1:8050`을 엽니다.

결과보고서의 근거표, EDA·모델 평가·대시보드 그림 14개, 편집 가능한 모델별 Grid Search·성능·재현성 표 6개, HWPX 생성과 검증을 한 명령으로 실행합니다. 저장된 대시보드 화면 네 장을 갱신할 때는 Playwright Chromium을 설치하고 `--capture-dashboard`를 붙입니다.

```powershell
python -m pip install -r requirements.txt
python scripts/reproduce_submission.py
```

`--train-trees`는 완료된 tree grid-search CSV를 재사용하고 최종 모델을 다시 학습합니다.
108개 후보 × 3개 시간순 fold를 처음부터 다시 탐색하려면
`python scripts/reproduce_submission.py --train-trees --force-tree-search --n-jobs 8`을 사용합니다.

대시보드 화면을 갱신하려면 `python -m playwright install chromium` 후 `--capture-dashboard`를 추가합니다. 신경망의 4단계 탐색과 평가 명령은 아래 재학습 절에 있습니다.

제출용 본문은 [심사기준별 상세내용](docs/report/심사기준별_상세내용.md)에서 바로 검토할 수 있으며, [HWPX 작성본](docs/report/OKM_경진대회_결과보고서_작성본.hwpx)에 같은 내용과 그림이 들어 있습니다. 그리드 탐색의 후보 수·범위·3개 fold 결과, 최종 후보 선택 기준과 수치, 오경보·미탐지 및 변수 영향 분석을 심사표의 6개 장에 맞춰 기록했습니다.

HWPX 작성본은 원본 양식의 장 구성과 필수 만족도 조사 항목을 유지합니다. 팀명·서명과 만족도 조사 완료 화면은 제출자가 채워야 합니다.

## 재학습

전처리를 확인하려면 `python scripts/verify_cleaned_data.py`를 실행합니다. 원본에서 다시 만든 6,168행×22열을 제공된 cleaned CSV와 전 열 비교합니다.

트리 모델은 Python 3.12에서 아래 패키지 버전으로 실행했습니다: xgboost 2.0.3, lightgbm 4.7.0, numpy 2.4.3, pandas 3.0.1, scikit-learn 1.8.0. 루트의 `requirements.txt`를 설치한 후 다음을 실행합니다.

```powershell
python scripts/train_tree_models.py --models lightgbm xgboost --n-jobs 8
python scripts/build_model_comparison.py
```

신경망 재학습도 같은 `requirements.txt` 환경을 사용합니다. 단계별 탐색은 아래 순서로 실행하며, 각 단계의 완료 결과를 다음 단계가 읽습니다.

```powershell
python -m pip install -r requirements.txt

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
| `scripts/train_tree_models.py` | 두 트리 모델의 전체 그리드 탐색·평가 |
| `scripts/build_model_comparison.py` | 5개 모델 결과와 시각·정답 일치 검증 |
| `results/tree_models/` | 전체 트리 그리드, 선택 모델, 예측, 지표 |
| `results/model_comparison.csv` | 동일 Test 구간의 5개 모델 비교 |
| `dashboard/` | 저장된 Test 예측 시각화 |
| `docs/templates/` | 원본 HWPX 결과보고서 양식 |
| `docs/report/` | HWPX 작성본, 상세 본문, 근거표, 그래프·대시보드 캡처 |
| `docs/presentation/` | 전력 피크 예측 발표 PPT와 EDA·전처리 발표 PPT |
| `requirements.txt` | 신경망·트리·대시보드·보고서의 통합 실행 환경 |
| `scripts/reproduce_submission.py` | 정제 검증부터 보고서 파일 검증까지 실행 |

비교 설계와 수치는 [모델 비교 문서](docs/model-comparison.md)를 참조하세요.

## 대시보드 캡처

```powershell
python -m pip install -r requirements.txt
python -m playwright install chromium
python scripts/capture_dashboard.py
```

캡처는 `docs/report/figures/`의 `03_dashboard.png`, `12_dashboard_causes.png`, `13_dashboard_model.png`, `14_dashboard_actions.png`에 저장됩니다.

## 소스코드 제출 ZIP

```powershell
python scripts/package_source_submission.py
```

`outputs/OKM_소스코드_제출.zip`에 raw·정제 데이터, 전처리부터 대시보드까지의 코드, 저장된 모델 결과, Test 703시간의 다섯 모델 통합 예측 CSV, 공통 `requirements.txt`, 실행 README와 SHA-256 목록을 묶습니다. ZIP의 `README.md` 순서대로 압축 해제 위치에서 재현할 수 있습니다.
