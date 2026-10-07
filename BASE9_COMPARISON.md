# 기본 9개 변수 모델 비교

이 브랜치는 KJH 정제 데이터 `data/okm_cleaned_2021.csv`를 사용합니다. 피처 엔지니어링 데이터는 사용하지 않습니다. 원본 CSV도 `data/okm_augumented_2021.csv`에 함께 보관합니다.

## 공통 기준

- 입력 변수: 과거 전력 평균, 생산량, 기온, 풍속, 습도, 강수량, 시간 sin/cos, 주말 여부 (총 9개)
- seed: 42
- 검증: 시간순 확장 fold 3개 (2021-05-16~06-15, 06-16~07-15, 07-16~08-15)
- 각 fold의 평가 시작 전 14일은 학습에서 제외
- Test: 2021-08-16~09-14, 셧다운 17시간을 제외한 703시간
- 피크 기준: 177 kW
- 최종 설정: 검증 평균 RMSE 최저의 2% 이내 후보 중 피크 구간 MAE가 가장 낮은 설정
- 경보 확률: 검증 잔차의 표준편차로 계산하고, 검증 F1이 가장 높은 경보 기준을 적용

LSTM과 TCN 결과는 `lhs_cleaned/results/`에 가져온 기존 실험입니다. LSTM은 168시간, TCN은 24시간 입력을 사용합니다. 가중 앙상블은 두 모델의 검증 MSE 역수로 가중치를 정하며 별도 그리드 서치를 하지 않습니다. XGBoost와 LightGBM만 이 브랜치에서 다시 탐색하고 학습합니다. 두 트리 모델의 입력 창은 168시간입니다. 따라서 대상 데이터·변수·fold·Test·seed·선택 규칙은 맞추되, 모델별 입력 길이는 기존 딥러닝 최종 설정을 존중합니다.

기존 `outputs/xgboost_lightgbm_summary/`의 수치는 이전 20개 변수 실험입니다. 이 브랜치의 9개 변수 비교에는 `results/base9_tree/`와 `results/base9_comparison.csv`를 사용합니다.

## 파일

- `lhs_cleaned/grid_search.py`, `lhs_cleaned/final_evaluate.py`, `lhs_cleaned/src/models/`: 기존 딥러닝 학습·평가 코드
- `lhs_cleaned/results/grid_search/`, `lhs_cleaned/results/final/`: 기존 seed 42 LSTM·TCN·앙상블 결과
- `scripts/train_base9_trees.py`: XGBoost·LightGBM 전체 108조합 × 3 fold 탐색, 재학습, Test 평가
- `results/base9_tree/grid_search/`: 모든 트리 후보의 fold별 지표와 후보별 집계
- `results/base9_tree/test_metrics.csv`: 새 트리 모델의 Test 회귀·피크 경보 지표
- `results/base9_tree/predictions/`, `results/base9_tree/models/`: 시간별 예측과 학습된 트리 모델
- `scripts/build_base9_comparison.py`: 두 트리 탐색이 108조합 × 3 fold 모두 완료됐는지와 다섯 모델의 Test 시점·정답 일치를 확인하고 비교표 생성
- `results/base9_comparison.csv`: 다섯 모델의 동일 Test 구간 지표
- `presentation/`: lhs 발표 PPT 원본
- `dashboard/`: kjh 대시보드 프로토타입 (실제 모델과 아직 연결되지 않은 모의 데이터 화면)

## 트리 모델 실행

```powershell
python scripts/train_base9_trees.py --models lightgbm xgboost --n-jobs 8
python scripts/build_base9_comparison.py
```

검색 CSV는 fold마다 저장되며, 같은 명령을 다시 실행하면 완료된 fold는 건너뜁니다. 학습에는 `xgboost`, `lightgbm`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `joblib`가 필요합니다.
