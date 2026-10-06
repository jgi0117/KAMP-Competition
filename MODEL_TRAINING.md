# 모델 학습 및 비교 실행 안내

이 코드는 과거 168시간의 전처리 완료 데이터로 다음 1시간의 평균 전력 사용량(`전력_평균_실수`)을 예측합니다. 동일한 데이터와 분할 조건으로 XGBoost, LightGBM, TimesFM 3.0, Chronos-2, Moirai 2.0을 한 번에 실행하고 RMSE, MAE, R²를 비교합니다. LSTM과 TCN은 이 브랜치에 포함하지 않습니다.

## 공통 데이터 조건

- KJH 전처리 코드인 `src/preprocessing.py`를 사용합니다.
- 입력 Feature는 `kamp_models/schema.py`의 고정된 20개 열과 순서를 모든 모델이 공유합니다.
- 각 표본은 직전 168시간을 입력으로, 바로 다음 1시간의 `전력_평균_실수`를 정답으로 사용합니다.
- seed는 모든 모델에서 42로 고정합니다.
- 셧다운 시점은 입력 이력에는 남기고, 해당 시점을 정답으로 갖는 표본은 학습과 평가에서 제외합니다.

`data/okm_augumented_2021.csv`에 KJH 전처리를 적용한 결과는 `data/okm_cleaned_2021.csv`와 바이트 단위까지 같습니다. 두 파일의 SHA-256은 모두 `ce8142c4d88d4b627f1893304a3c07de328f5b6cce5bda7fdefe20397096c698`입니다.

## LHS 시간 분할

`src/data_pipeline.py`의 분할 상수를 kgj 모델도 직접 사용합니다. 따라서 lhs와 이 브랜치는 모델 구현만 다르고 데이터 조건은 같습니다.

최종 학습과 평가는 다음과 같이 분리합니다.

| 용도 | Target 기간 | 표본 수 |
|---|---|---:|
| train | 2021-01-08 00:00 ~ 2021-08-01 23:00 | 4,944 |
| validation / early stopping | 2021-08-02 00:00 ~ 2021-08-15 23:00 | 336 |
| test | 2021-08-16 00:00 ~ 2021-09-14 23:00 | 703 |

전체 6,000개 window 중 셧다운 Target 17개를 제외하여 5,983개를 사용합니다.

하이퍼파라미터 탐색은 다음 세 개의 expanding fold를 사용합니다. 각 평가 구간 직전 14일은 early stopping 전용이며 평가 점수 계산에 섞지 않습니다.

| Fold | train | early stopping | 평가 | 표본 수(train / early stopping / 평가) |
|---|---|---|---|---:|
| 1 | 2021-05-01 23:00 이전 | 2021-05-02 ~ 05-15 | 2021-05-16 ~ 06-15 | 2,736 / 336 / 744 |
| 2 | 2021-06-01 23:00 이전 | 2021-06-02 ~ 06-15 | 2021-06-16 ~ 07-15 | 3,480 / 336 / 720 |
| 3 | 2021-07-01 23:00 이전 | 2021-07-02 ~ 07-15 | 2021-07-16 ~ 08-15 | 4,200 / 336 / 744 |

임시 7:2:1 분할이 다시 필요하면 `prepare_data.py --split-strategy ratio`를 사용할 수 있습니다. 기본값은 `lhs`입니다.

## 실행

Anaconda 환경을 만들거나 갱신한 뒤 실행합니다.

```powershell
conda env update --name kamp-competition --file environment.yml
conda activate kamp-competition
```

원본 CSV부터 전처리, window 생성, 전체 모델 비교까지 한 번에 실행합니다.

```powershell
python run_models.py --data data/okm_augumented_2021.csv --output outputs/experiment_01
```

전처리와 공통 분할 파일만 만들 수도 있습니다.

```powershell
python prepare_data.py --input data/okm_augumented_2021.csv `
  --cleaned-reference data/okm_cleaned_2021.csv `
  --output outputs/prepared/model_input_lhs.npz
```

연결 상태만 빠르게 확인하려면 트리 모델을 smoke 조건으로 실행합니다.

```powershell
python run_models.py --data outputs/prepared/model_input_lhs.npz `
  --models xgboost lightgbm --quick --output outputs/smoke
```

## 학습과 fine tuning

- XGBoost와 LightGBM은 문서의 전체 grid를 세 개의 LHS fold에서 평가하고 평균 RMSE가 가장 낮은 조합을 선택합니다.
- TimesFM 3.0, Chronos-2, Moirai 2.0은 zero-shot과 fine-tuning 결과를 모두 생성합니다.
- 사전학습 모델의 fine-tuning 범위는 F0~F7, 학습률은 `1e-6`, `1e-5`, `1e-4`이며 각 조합을 동일한 세 fold에서 비교합니다.
- 유효 batch size는 32, 최대 optimizer step은 1,000, 검증 주기는 100 step, early stopping patience는 3입니다.
- 최종 비교 지표는 RMSE, MAE, R²만 사용합니다.

모델별 구현은 `kamp_models/models/`의 별도 Python 파일에 있고 `kamp_models/runner.py`가 순서대로 실행해 하나의 비교표를 만듭니다.

## 결과 파일

- `comparison.csv`: 모델별 RMSE, MAE, R²와 실행 시간
- `search_results_xgboost.csv`, `search_results_lightgbm.csv`: fold별 grid 탐색 결과
- `finetune_search_results_<model>.csv`: 사전학습 모델의 F0~F7, 학습률, fold 결과
- `predictions/<model>.csv`: 정답과 예측값
- `models/`: 최종 모델 또는 fine-tuned checkpoint
- `run_metadata.json`: 실행 환경, 데이터 크기, 장치, fine-tuning 조건

GPU를 사용할 수 있으면 자동으로 사용하고, 해당 라이브러리의 GPU 빌드나 장치가 없으면 그 모델만 CPU로 실행합니다.

## 터미널 진행 상태

실행을 시작하면 선택된 장치, CUDA 연결 여부, GPU 이름, CUDA runtime과 VRAM이 먼저 출력됩니다. 학습 중에는 현재 모델 번호, tree grid의 후보·fold 진행률, foundation 모델의 F0~F7·학습률·fold 번호, optimizer step, 검증 RMSE와 경과 시간이 표시됩니다. 각 모델이 끝날 때 실제 사용 장치와 최종 RMSE도 출력됩니다.
