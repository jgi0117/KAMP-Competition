# 딥러닝 모델 후보군 검토 – LSTM·TCN 기반 Hybrid 모델

> 원본 PDF: [딥러닝모델후보군검토_LSTMTCN기반Hybrid모델.pdf](./딥러닝모델후보군검토_LSTMTCN기반Hybrid모델.pdf)

## 1. 기본 모델

이번 프로젝트에서는 먼저 아래 두 모델을 기본 모델로 사용한다.

- **LSTM**
- **TCN**

두 모델을 동일한 데이터 분할, 동일한 입력 Feature, 동일한 평가 지표 기준으로 비교한 뒤, 성능이 확인되면 논문 기반 Hybrid 모델을 추가로 적용한다.

Hybrid 후보는 구현 난이도와 프로젝트 적용 가능성을 고려하여 아래 2개 모델로 선정한다.

- IEEE EECR 2023 기반 **LSTM-TCN Weighted Ensemble**
- IET 2026 기반 **Causation-Guided TCN-LSTM**

---

## 2. 후보 1 – IEEE EECR 2023

### 논문 정보

**논문명**

> *Integrated Forecasting Models Based on LSTM and TCN for Short-Term Electricity Load Forecasting*

**발표**

- 2023 9th International Conference on Electrical Engineering, Control and Robotics (EECR)
- IEEE
- DOI: `10.1109/EECR56827.2023.10149951`

논문은 단기 전력부하 예측 정확도를 높이기 위해 **LSTM과 TCN을 각각 학습한 후 두 모델의 예측값을 결합하는 방식**을 제안한다.

### 모델 구조

기본적인 구조는 다음과 같다.

```
     동일한 시계열 데이터
            │
       ┌────┴────┐
       ↓         ↓
     LSTM       TCN
       ↓         ↓
   예측값 1   예측값 2
       └────┬────┘
            ↓
      가중치 기반 결합
            ↓
      최종 전력 예측값
```

LSTM과 TCN을 하나의 네트워크 안에 직접 연결하는 방식은 아니다.
먼저 두 모델을 **독립적으로 학습**하고 각각 예측값을 만든 뒤, 두 결과를 최종적으로 결합한다.

### 핵심 아이디어

단순히

```
(LSTM 예측 + TCN 예측) / 2
```

처럼 평균을 내는 것이 아니라, 각 모델의 예측오차를 기준으로 가중치를 결정한다.

논문에서는 **inverse squared error ratio**를 이용해 두 예측값을 결합한다. 즉 검증 과정에서 오차가 작은 모델의 결과에 더 높은 비중을 주는 방식이다.

개념적으로 보면 다음과 같다.

```
LSTM Validation Error
        ↓
    LSTM Weight

TCN Validation Error
        ↓
    TCN Weight

LSTM Prediction × Weight
          +
TCN Prediction × Weight
          ↓
   Final Prediction
```

### LSTM과 TCN을 같이 사용하는 이유

두 모델은 시계열을 학습하는 방법이 다르다.

#### LSTM

이전 시점의 정보를 내부 상태에 유지하면서 순차적으로 학습한다.

주요 장점:

- 시간 순서 학습
- 과거 정보 유지
- 장기적인 의존관계 학습

#### TCN

Causal Convolution과 Dilated Convolution을 사용해 여러 시간범위의 패턴을 학습한다.

주요 장점:

- 병렬 학습 가능
- 비교적 긴 과거 범위 처리
- 국소적인 변화 패턴 탐지

따라서 두 모델의 예측결과를 결합하면 한 모델의 약점을 다른 모델이 보완할 수 있다는 것이 이 논문의 접근이다.

실험에서도 결합 LSTM-TCN 모델이 단일 LSTM, 단일 TCN 및 비교 네트워크보다 더 낮은 예측오차를 보였다고 보고한다.

### 우리 프로젝트 적용 방법

우리 프로젝트에서는 먼저

```
Model 1 : LSTM
Model 2 : TCN
```

을 각각 학습한다.

그 후 Validation 성능을 이용해 두 모델의 Weight를 계산한다.

예를 들어

```
LSTM RMSE = 15
TCN RMSE  = 12
```

라면 TCN 쪽에 더 높은 Weight를 부여한다.

최종적으로

```
Final Prediction
=
LSTM Prediction × W1
+
TCN Prediction × W2
```

형태로 예측한다.

### 우리 프로젝트에서의 장점

가장 큰 장점은 **구현 부담이 작다는 점**이다.

어차피 이번 프로젝트에서

- LSTM
- TCN

을 각각 구현할 예정이므로 새로운 복잡한 신경망을 다시 개발할 필요가 없다.
기존 두 모델의 예측값을 결합하는 단계만 추가하면 된다.

따라서 실험 구조도 명확하다.

```
LSTM
vs
TCN
vs
LSTM-TCN Weighted Ensemble
```

세 모델의 성능을 동일한 Test Set에서 직접 비교할 수 있다.

### 적용 우선순위

**높음**

이유:

- 구현 난이도 낮음
- 기존 LSTM/TCN 코드 재사용 가능
- 추가 학습 비용이 크지 않음
- 단기 전력수요 예측 논문에서 직접 검증
- 단일 모델 대비 Ensemble 효과 확인 가능

---

## 3. 후보 2 – IET 2026

### 논문 정보

**논문명**

> *A Causation-Guided Data-Driven Model for Electricity Consumption Forecasting of Industrial Chains*

**게재**

- IET Generation, Transmission & Distribution
- Volume 20, Issue 1
- First Published: 24 April 2026
- DOI: `10.1049/gtd2.70302`

이 논문은 일반적인 건물 전력이나 가정 전력이 아니라 **산업체 전력소비 예측**을 대상으로 한다.

실제 중국의 전기차 산업체 데이터를 이용해 모델을 검증했다는 점에서 이번 제조업 경진대회와 연관성이 높다.

### 전체 연구 구조

이 논문은 단순히 TCN과 LSTM만 결합하지 않는다.
먼저 산업체 전력사용량에 영향을 주는 변수들의 관계를 분석하고, 그 결과를 모델 입력에 반영한다.

전체 흐름은 다음과 같다.

```
산업 전력 데이터
+
외부 영향요인
      ↓
인과관계 / 상관관계 분석
      ↓
중요한 변수 선정
      ↓
변수별 Weight 적용
      ↓
Time Lag 반영
      ↓
TCN-LSTM
      ↓
전력소비 예측
```

논문에서는 산업 내부 관계를 분석하기 위해 CCM과 PCM을 사용하고, 외부 요인의 관련성은 Grey Relational Analysis로 정량화한다. 이렇게 얻은 인과관계와 상관관계를 TCN-LSTM 입력의 Weight와 시간 Window에 반영한다.

---

## 4. 핵심 모델 – TCN → LSTM

우리 프로젝트에서 가장 참고할 부분은 **TCN-LSTM 구조**이다.

구조는 다음과 같다.

```
  과거 시계열 입력
        ↓
       TCN
        ↓
 시계열 Feature 추출
        ↓
       LSTM
        ↓
  시간 의존성 학습
        ↓
  전력사용량 예측
```

즉 IEEE 모델처럼 마지막 예측값을 결합하는 방식이 아니라,

> **TCN의 출력이 LSTM의 입력으로 연결되는 하나의 Hybrid Network**

이다.

논문은 TCN의 시간 순서 제약과 LSTM의 장기 의존성 학습 능력을 함께 활용하기 위해 이 구조를 사용한다.

### TCN의 역할

TCN이 먼저 원본 시계열에서 시간적인 특징을 추출한다.

예를 들어

```
t-168
...
t-24
...
t-3
t-2
t-1
```

과 같은 과거 데이터가 입력되면 TCN의 Dilated Convolution을 통해 여러 시간범위의 특징을 추출한다.

따라서

- 직전 시간의 변화
- 단기 전력 증가
- 하루 주기
- 장기 패턴

등을 먼저 압축된 Feature 형태로 만들 수 있다.

### LSTM의 역할

TCN에서 만들어진 Temporal Feature를 LSTM에 입력한다.

LSTM은 이를 순차적으로 처리하면서

> 과거의 어떤 패턴이 현재 전력수요에 지속적으로 영향을 주는가

를 학습한다.

따라서 역할을 단순화하면

```
TCN
=
시간 패턴 추출

LSTM
=
시간 패턴 사이의 장기 관계 학습
```

으로 볼 수 있다.

---

## 5. Time Lag 개념

IET 논문에서 특히 참고할 만한 부분은 **변수마다 영향이 나타나는 시간이 다를 수 있다는 점**이다.

예를 들어

```
생산량 변화
    ↓
즉시 전력 증가

기온 상승
    ↓
몇 시간 뒤 냉방부하 증가

과거 전력
    ↓
다음 시간 전력에 직접 영향
```

처럼 변수마다 전력사용량에 영향을 주는 시간이 다를 수 있다.

논문은 인과관계를 분석한 뒤 해당 **causal time lag를 모델의 입력 Window에 반영**한다.

---

## 6. 우리 프로젝트와의 연결

우리 데이터에는 다음과 같은 변수가 존재한다.

```
전력사용량
생산량
시간
요일
기온
습도
풍속
강수량
```

또한 EDA에서 이미

```
Lag 1h   → 강한 상관
Lag 168h → 강한 주간 반복성
```

이 확인되었다.

따라서 TCN이 여러 시간대의 패턴을 먼저 추출하고, LSTM이 이 패턴의 장기 관계를 다시 학습하는 구조를 적용할 근거가 있다.

특히 IET 논문은 산업 전력소비 자체를 대상으로 했다는 점에서 이번 프로젝트와의 연결성이 높다. 논문의 Case Study 역시 실제 전기차 산업체 데이터를 이용해 검증됐다.

---

## 7. 논문 전체 구조를 그대로 구현할 필요는 없음

IET 논문에는 다음과 같은 추가 기법도 포함되어 있다.

- CCM
- PCM
- Grey Relational Analysis
- Wavelet decomposition
- Causal Weight
- Causal Time Lag
- TCN-LSTM

이를 이번 프로젝트에 모두 그대로 구현하면 오히려 복잡도가 지나치게 커질 수 있다.

따라서 1차적으로 가져올 핵심은 아래 정도로 한정한다.

> **TCN → LSTM Hybrid 구조**

그리고 추가적으로 성능 개선 여지가 있다면

> **Feature별 Time Lag 반영**

정도까지 검토한다.

즉 처음부터 논문 전체를 재현하는 것이 아니라 **우리 데이터에 필요한 핵심 구조만 차용**한다.

---

## 8. 두 후보 모델 비교

| 구분 | IEEE EECR 2023 | IET 2026 |
| --- | --- | --- |
| 모델 | LSTM-TCN Weighted Ensemble | TCN-LSTM Hybrid |
| 결합 방식 | 예측결과 결합 | 네트워크 내부 연결 |
| 구조 | LSTM / TCN 병렬 | TCN → LSTM |
| 난이도 | 낮음 | 중간 |
| 기존 코드 활용 | 매우 쉬움 | 가능 |
| 추가 학습 | 거의 없음 | 필요 |
| 핵심 장점 | Ensemble 효과 | 시간 Feature + 장기 관계 동시 학습 |
| 논문 분야 | Short-Term Load Forecasting | Industrial Electricity Forecasting |
| 우리 데이터 연관성 | 높음 | **매우 높음** |

---

## 9. 최종 실험 모델

현재 모델 후보는 다음 4개로 정리한다.

```
① LSTM

② TCN

③ IEEE EECR 2023
   LSTM-TCN Weighted Ensemble

④ IET 2026
   TCN-LSTM Hybrid
```

---

## 10. 실험 진행 순서

우선 단일 모델부터 충분히 튜닝한다.

```
STEP 1

LSTM
vs
TCN
```

↓

각 모델의 최적 Hyperparameter 선정

↓

```
STEP 2

LSTM
TCN
LSTM-TCN Weighted Ensemble
```

↓

Ensemble 성능 확인

↓

```
STEP 3

TCN-LSTM Hybrid
```

↓

최종 비교

```
LSTM
vs
TCN
vs
Weighted Ensemble
vs
TCN-LSTM Hybrid
```

---

## 11. 후보 선정 이유

이번 모델 선정의 기준은 단순히 모델 종류를 늘리는 것이 아니다.
기본 모델인 LSTM과 TCN을 먼저 충분히 비교하고, 두 모델을 결합했을 때 실제 성능이 개선되는지를 검증한다.

### IEEE EECR 2023

> 이미 구현한 LSTM과 TCN을 그대로 활용하면서 가장 적은 비용으로 Ensemble 효과를 검증할 수 있다.

### IET 2026

> 제조·산업 전력사용량을 대상으로 TCN과 LSTM을 하나의 네트워크로 결합한 연구이기 때문에 이번 경진대회 주제와 직접적인 연결성이 높다.

따라서 현재 모델링 전략은 다음과 같이 정리한다.

> **LSTM과 TCN을 기본 모델로 구축하고 각각의 최적 성능을 확보한 뒤, IEEE EECR 2023의 Weighted Ensemble과 IET 2026의 TCN-LSTM Hybrid를 추가하여 단일 모델 대비 성능 개선 여부를 검증한다.**

### 논문 원문

- [IEEE Xplore – Integrated Forecasting Models Based on LSTM and TCN for Short-Term Electricity Load Forecasting](https://doi.org/10.1109/EECR56827.2023.10149951)
- [Wiley/IET – A Causation-Guided Data-Driven Model for Electricity Consumption Forecasting of Industrial Chains](https://doi.org/10.1049/gtd2.70302)
