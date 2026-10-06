# 모델 학습 및 비교 실행 안내

이 코드는 과거 168시간의 전처리 완료 데이터로 다음 1시간의 평균 전력사용량을 예측합니다. 동일 데이터로 XGBoost, LightGBM, TimesFM 3.0, Chronos-2, Moirai 2.0을 한 번에 실행하고 RMSE, MAE, R²를 비교합니다. LSTM은 포함하지 않습니다.

## 입력 데이터

입력은 NumPy `npz` 파일 한 개입니다. 시간순으로 분리된 다음 배열이 필요합니다.

| 키 | 필수 형태 | 설명 |
|---|---|---|
| `X_train`, `X_val`, `X_test` | `[표본, 특성]` 또는 `[표본, 168, 특성]` | 트리 모델 입력 |
| `y_train`, `y_val`, `y_test` | `[표본, 1]` | 각 기준 시점 직후 1시간의 평균 전력사용량 |
| `target_history_train`, `target_history_val`, `target_history_test` | `[표본, 168]` | 사전학습 모델의 전력사용량 이력 |

`X_*`가 `[표본, 168]`이면 그 배열을 전력 이력으로 자동 사용합니다. `X_*`가 `[표본, 168, 특성]`이면 `model_config.toml`의 `target_feature_index` 열을 전력 이력으로 사용합니다. 이 두 형태가 아니면 `target_history_*`를 별도로 넣어야 합니다.

모든 배열은 시간순이어야 하며 결측값과 무한값이 없어야 합니다. 윈도우 생성, 다음 1시간 평균 산출, 스케일 복원, 날짜 분할은 전처리 단계에서 끝내야 합니다. 평가값은 전달된 원 단위 배열에서 계산합니다.

## 실행

Anaconda Prompt에서 다음 명령을 실행합니다.

```powershell
conda activate kamp-competition
python run_models.py --data data/preprocessed.npz --output outputs/experiment_01
```

전체 탐색 전에 코드와 데이터 연결만 확인하려면 다음처럼 실행합니다. `--quick`은 트리 모델의 단일 조합과 사전학습 모델의 F0·첫 학습률·1 optimizer step만 실행합니다.

```powershell
python run_models.py --data data/preprocessed.npz --output outputs/smoke --quick
```

특정 모델만 실행할 수도 있습니다.

```powershell
python run_models.py --data data/preprocessed.npz --models xgboost lightgbm --output outputs/trees
```

모델 구현은 `kamp_models/models/` 아래의 `xgboost.py`, `lightgbm.py`, `timesfm3.py`, `chronos2.py`, `moirai2.py`로 분리되어 있습니다. `kamp_models/runner.py`가 설정을 읽고 이 모델들을 순서대로 실행해 하나의 비교표로 합칩니다. 공통 시계열 fold와 미세조정 루프는 `foundation_common.py`, F0~F7 동결 규칙은 `finetuning.py`에 있습니다.

## 학습 방식

- XGBoost와 LightGBM은 문서에 제시된 108개 조합을 각각 3개 `TimeSeriesSplit` fold로 평가합니다. 각 예측 시차는 독립 회귀기로 학습하며, 검증 RMSE가 가장 낮은 조합을 전체 train+validation 데이터로 다시 학습합니다.
- TimesFM 3.0, Chronos-2, Moirai 2.0은 모두 zero-shot 결과와 미세조정 결과를 생성합니다.
- 미세조정 탐색은 모델마다 `F0~F7 × 학습률 [1e-6, 1e-5, 1e-4] × 3개 expanding fold`, 총 72회입니다. 각 실험은 원본 사전학습 체크포인트에서 독립적으로 시작합니다.
- F0은 출력부만 학습합니다. F1/F2는 마지막 1/2개 Transformer 블록과 출력부를 학습합니다. F3~F6은 문서의 모델별 블록 수를 적용하고 F7은 입력부를 포함한 전체 파라미터를 학습합니다.
- 유효 배치는 32, 최대 optimizer step은 1,000, 검증 주기는 100 step, 조기 종료 patience는 3회입니다. 물리 배치는 기본 1이고 gradient accumulation 32회로 유효 배치를 맞춥니다.
- fold 평균 검증 RMSE가 가장 낮은 범위와 학습률을 선택합니다. fold별 최적 step의 중앙값으로 전체 train+validation 구간을 고정 seed 42에서 다시 학습합니다.
- 모든 모델의 최종 비교 지표는 시험 구간의 다음 1시간 평균 전력사용량 예측에 대한 RMSE, MAE, R²입니다.

기본 Anaconda 환경에 설치된 PyTorch가 CPU 빌드라면 전체 탐색 시간이 매우 깁니다. CUDA PyTorch 환경에서는 `device = "auto"`가 GPU를 자동 선택합니다. GPU 메모리가 허용되면 `train_batch_size`를 2, 4처럼 늘리되 32의 약수로 두면 됩니다.

## 결과 파일

- `comparison.csv`: 트리 모델과 사전학습 모델의 zero-shot(`*_zs`)·미세조정(`*_ft`) 결과, RMSE, MAE, R², 학습·추론 시간
- `search_results_xgboost.csv`, `search_results_lightgbm.csv`: 모든 fold 탐색 결과
- `finetune_search_results_<model>.csv`: F0~F7, 학습률, fold별 RMSE, MAE, R², 최적 step과 파라미터 수
- `predictions/<model>.csv`: 표본과 예측 시차별 정답·예측값
- `models/*.joblib`: 최종 트리 모델
- `models/timesfm3`, `models/moirai2`: 원본 체크포인트에 덮어쓸 미세조정 파라미터
- `models/chronos2`: seed 42 최종 Chronos 체크포인트와 선택 설정
- `run_metadata.json`: 실행 환경과 데이터 크기

사전학습 모델 패키지나 모델 가중치 다운로드가 실패해도 다른 모델은 계속 실행되며, 실패 원인은 `comparison.csv`의 `error` 열에 기록됩니다.
