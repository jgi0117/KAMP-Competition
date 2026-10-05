# 제조 생산데이터 기반 전력사용량 예측 모델 및 평가 지표 선정

2026년 10월 5일

## 1. 모델 후보 3개

세 모델을 **동일한 예측 구간, 입력 정보, 시간순 검증 구간**에서 모두 학습·비교한 뒤 최종 모델을 선정한다. 아래 논문 수치는 각 연구의 자료와 실험 조건에서 나온 결과이므로 모델 간 순위를 미리 정하는 데 사용하지 않는다.

| 모델 | 전력사용량 예측 논문에서의 사용 이력 | 논문 평가 지표 | 후보 선정 근거 |
|---|---|---|---|
| **XGBoost** | 철강공장 시간별 전력사용량 예측 연구에서 과거 24시간의 전력값을 입력해 시험했다. 연구가 보고한 XGBoost 결과는 RMSE 54.833 kWh, MAE 30.302 kWh, MAPE 74.57%다. [1] | RMSE, MAE, MAPE [1] | 생산·기상·달력·과거 전력처럼 형태가 다른 입력을 한 모델에 결합하고, 비선형 조건을 학습할 수 있다. |
| **LightGBM** | 여섯 지역의 시간별 전력부하를 대상으로 XGBoost와 순환신경망 등을 함께 비교한 연구가 있다. 스페인 1년 자료에서 LightGBM의 보고값은 nRMSE 2.31%, MAE 397.8, R² 0.98이다. [2] | nRMSE, MAE, R² [2] | XGBoost와 동일한 입력 조건에서 부스팅 방식의 차이를 비교할 수 있다. |
| **LSTM** | 같은 철강공장 연구에서 순차적인 과거 전력값을 입력해 시험했다. 연구가 보고한 LSTM 결과는 RMSE 49.826 kWh, MAE 28.139 kWh, MAPE 66.80%다. [1] | RMSE, MAE, MAPE [1] | 생산 및 전력 변화의 시간적 순서를 직접 학습하는 방식이 트리 모델보다 유리한지 검증할 수 있다. |

## 2. 사전학습 모델 3개와 전력사용량 예측 적용 이력

아래 세 모델도 동일한 평가 조건에서 비교할 수 있다. **전력 가격 예측이나 발전량 예측을 전력사용량 예측 이력으로 세지 않았다.** 적용 이력의 대상이 제조공장인지, 건물·고객·전력망인지 구분했다.

| 모델 | 2026년 기준 모델 특성 | 실제 전력사용량 예측 적용 이력 | 선정 근거 |
|---|---|---|---|
| **TimesFM 3.0** (2026년 8월 발표) | Google Research의 다변량 사전학습 모델. 과거 변수와 미래에 알려진 변수를 입력하고 점·분위수 예측을 생성한다. [4] | **있음 — 고객 전력소비 벤치마크.** 공식 TimesFM 3.0 GIFT-Eval 결과에 `electricity/W/short`, `electricity/D/short` 실험이 기록돼 있다. GIFT-Eval 논문이 정의한 Electricity 자료의 원천은 실제 고객 전력소비 기록이다. 제조공장 개별 자료의 검증은 아니다. [5, 6] | 생산계획 등 미래에 알려진 변수의 효과를 비교할 수 있다. 공개 가중치는 비상업·비운영 용도로 제한된다. [4, 7] |
| **Chronos-2** (2025년 10월 발표) | 단변량·다변량·공변량 포함 예측을 지원한다. [8] | **있음 — 실제 전력망 부하.** 2026년 연구가 ISO New England와 ENTSO-E 전력부하에 Chronos-2를 적용해 제로샷·미세조정 예측을 비교했다. 제로샷 성능은 해당 연구의 전용 학습 모델보다 낮았고, 미세조정 후 단기 예측은 개선됐다. [9] | 생산·기상·달력 변수의 활용과 사전학습 모델의 적응 효과를 함께 비교할 수 있다. |
| **Moirai 2.0** (가중치 2025년 8월 공개) | 분위수 예측을 제공한다. 원 논문은 변수를 독립된 단변량 시계열로 처리한다고 명시한다. [10, 11] | **있음 — 실제 전력망 부하.** 2026년 제로샷 비교 연구가 2020~2024년 ERCOT 시간별 전력부하에 Moirai-2를 적용했다. 512시간 입력·24시간 예측 조건에서 MASE 약 0.33을 보고했다. [12] | 전력 이력만 사용하는 사전학습 예측의 비교 기준이 된다. 생산·기상 변수의 공동 효과를 직접 모델링하는 후보는 아니다. |

TimesFM 3.0의 적용 근거는 **공식 공개 실험 결과와 GIFT-Eval 논문**이다. Chronos-2와 Moirai 2.0의 적용 근거는 **실제 부하 자료를 사용한 별도 연구 논문**이다. 세 적용 대상 모두 제조공장 전력사용량과 완전히 같지는 않으므로 최종 선정은 이 과제의 동일 조건 비교 결과로 결정한다.

### 사전학습 모델 라이선스

| 모델 | 공개 가중치 라이선스 | 코드 라이선스 및 사용 범위 |
|---|---|---|
| **TimesFM 3.0** | **TimesFM Non-Commercial License v1.0** [7] | 저장소 코드는 **Apache-2.0** [13]. 3.0 가중치는 비상업·비운영 사용으로 제한되며, 상업·운영 목적은 별도 허가가 필요하다. [7] |
| **Chronos-2** | **Apache-2.0** [14] | `chronos-forecasting` 코드도 **Apache-2.0** [15]. 라이선스 조건을 지키면 상업적 이용이 가능하다. |
| **Moirai 2.0-R-small** | **CC BY-NC 4.0** [16] | `uni2ts` 코드는 **Apache-2.0** [17]. 공개 가중치의 상업적 이용은 허용되지 않는다. [16] |

## 3. 평가 지표 선정

| 목적 | 지표 | 선택 근거 |
|---|---|---|
| 전체 구간의 평균 오차 | **MAE** = 절대오차의 평균 | 철강공장 [1], 지역 전력부하 [2], 조선소 최대수요전력 [3] 연구가 공통으로 사용한다. 원래 전력 단위의 평균 오차를 비교할 수 있다. |
| 큰 예측오차 | **RMSE** = 평균 제곱오차의 제곱근 | 철강공장 [1]과 조선소 피크 [3] 연구가 사용한다. 큰 오차에 더 큰 가중치를 준다. |
| 높은 전력사용량 구간 | **피크 구간 MAE** | 전체 MAE가 낮아도 높은 부하 구간의 오차는 클 수 있으므로 분리해 보고한다. 조선소 연구는 전체 오차와 상위 위험일 탐지를 함께 평가했다. [3] |

**비교 시 고정할 사항:** 예측 시작 시점과 예측 길이, 시간순 훈련·검증·시험 구간을 여섯 모델에 동일하게 적용한다. 여섯 모델 모두 전력 이력만 쓰는 공통 실험을 수행하고, 외생 변수를 받는 모델은 생산·기상·달력 변수를 추가한 확장 실험도 수행한다. 피크 구간의 기준값은 시험 구간을 보기 전에 훈련 구간에서 정한다. 전체 MAE·RMSE와 피크 구간 MAE를 함께 확인해 최종 모델을 선정한다. MAPE는 [1]에서, nRMSE·R²는 [2]에서 사용한 지표로 기록하되 공통 비교 지표로는 MAE와 RMSE를 사용한다.

## 4. 참고 문헌 및 공식 실험 자료

[1] “Comparative evaluation of several models for forecasting hourly electricity use in a steel plant,” *Scientific Reports*, 2026. https://doi.org/10.1038/s41598-026-43868-z

[2] “Forecasting load consumption: a comprehensive evaluation of deep learning and machine learning techniques,” *Electric Power Systems Research*, 2025. https://doi.org/10.1016/j.epsr.2025.111834

[3] K. Lee, J. Ahn, J. Lee, “Day-ahead industrial peak demand forecasting: A shipyard benchmark of deep learning forecasters and stacking ensembles,” *International Journal of Electrical Power & Energy Systems*, 2026. https://doi.org/10.1016/j.ijepes.2026.111933

[4] Google Research, “TimesFM-3: A zero-shot foundation model for multivariate forecasting,” 2026-08-31. https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/

[5] Google Research, TimesFM 3.0 GIFT-Eval 공식 결과표. https://raw.githubusercontent.com/google-research/timesfm/master/timesfm3-usage/benchmarks/gift_eval/all_results.csv

[6] T. Aksu et al., “GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation,” arXiv:2410.10393, 2024. https://arxiv.org/abs/2410.10393

[7] Google Research, TimesFM 3.0 공개 가중치 라이선스. https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE

[8] A. F. Ansari et al., “Chronos-2: From Univariate to Universal Forecasting,” arXiv:2510.15821, 2025. https://arxiv.org/abs/2510.15821

[9] V. Pendyala et al., “Assessing Covariate-Informed Grid Load Forecasting with a Time-Series Foundation Model,” arXiv:2609.06656, 2026. https://arxiv.org/abs/2609.06656

[10] C. Liu et al., “Moirai 2.0: When Less Is More for Time Series Forecasting,” arXiv:2511.11698, 2025. https://arxiv.org/abs/2511.11698

[11] Salesforce AI Research, Uni2TS 공식 저장소. https://github.com/SalesforceAIResearch/uni2ts

[12] L. Simeone, “Time Series Foundation Models for Energy Load Forecasting on Consumer Hardware: A Multi-Dimensional Zero-Shot Benchmark,” arXiv:2602.10848, 2026. https://arxiv.org/abs/2602.10848

[13] Google Research, TimesFM 코드 라이선스. https://github.com/google-research/timesfm/blob/master/LICENSE

[14] Amazon, Chronos-2 공식 모델 카드. https://huggingface.co/amazon/chronos-2

[15] Amazon Science, Chronos 코드 라이선스. https://github.com/amazon-science/chronos-forecasting/blob/main/LICENSE

[16] Salesforce AI Research, Moirai 2.0-R-small 공식 모델 카드. https://huggingface.co/Salesforce/moirai-2.0-R-small

[17] Salesforce AI Research, Uni2TS 코드 라이선스. https://github.com/SalesforceAIResearch/uni2ts/blob/main/LICENSE.txt
