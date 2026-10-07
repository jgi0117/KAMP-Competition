# 재현 방법 — 전처리부터 결과 생성까지 한 번에 실행

> 요구사항 6번(코드 및 재현성) 대응 문서입니다. `lhs` 브랜치 기준.
> 명령 하나(`python run_all.py`)로 **전처리 → 피처 생성 → 학습 → 추론·평가 → 오류분석 → 결과 요약**까지 자동 실행됩니다.

## 1. 실행 흐름

```
원본 okm_augumented_2021.csv
   │  1단계  정현님 전처리 (src/team_kjh/preprocessing.py)
   ▼        시간 이상치 복원, 결측 보정, 셧다운 표시, 전력_평균_실수
정제 okm_cleaned_2021.csv
   │  2단계  정현님 피처 생성 (src/team_kjh/features.py)
   ▼        달력·조업 일정, 과거 전력·생산 파생, 기상 파생 (64열)
피처 okm_features2_2021.csv
   │  3단계  (선택) 1차 그리드 서치 315회 (grid_search.py)
   │  4단계  최종 학습·추론·평가 (final_evaluate.py)
   │         검증 fold 3개로 σ·경보 기준·앙상블 가중치 → Test 8/16~9/14 평가
   │         베이스라인 2개 + LSTM·TCN·Weighted Ensemble·TCN-LSTM 비교
   │  5단계  영향요인·오류분석 (analysis/error_analysis.py)
   ▼  6단계  결과 요약
results/reproduce/  REPORT.md · run_manifest.json · final/ · data/
```

어느 단계 파일을 갖고 있든 그 단계부터 시작할 수 있습니다.

## 2. 환경 준비 (처음 한 번)

```bash
git clone https://github.com/jgi0117/KAMP-Competition.git
cd KAMP-Competition
git checkout lhs
python -m pip install -r requirements.txt     # Python 3.11 권장
```

`requirements.txt` 는 실제 실험에 쓴 버전으로 고정되어 있습니다 (TensorFlow 2.21, pandas 3.0, scikit-learn 1.9 등).

## 3. 데이터 넣기

데이터는 저장소에 올리지 않습니다. 팀 공유 드라이브에서 받아 아래 위치에 둡니다.

| 가진 파일 | 넣을 위치 | 실행 명령 |
| --- | --- | --- |
| 피처 `okm_features2_2021.csv` (가장 간단) | `data/processed/` | `python run_all.py` |
| 정제 `okm_cleaned_2021.csv` | `data/processed/` | `python run_all.py --cleaned data/processed/okm_cleaned_2021.csv` |
| 원본 `okm_augumented_2021.csv` | `data/raw/` | `python run_all.py --raw data/raw/okm_augumented_2021.csv` |

실행하면 데이터 지문(MD5)을 확인해서 **팀이 실험에 쓴 데이터와 같은지** 화면과 `run_manifest.json` 에 남깁니다.
정제 파일부터 실행하면 정현님 피처 코드가 팀 실험 파일과 **64개 컬럼 모두 같은 값**을 만들어 냅니다 (확인 완료).

## 4. 실행 옵션

| 옵션 | 뜻 | 걸리는 시간 |
| --- | --- | --- |
| (없음) | 저장소의 튜닝 결과(`results/grid_search/`)에서 최종 설정을 읽어 학습·평가·분석 | CPU 약 2.5시간 · Colab T4 약 30분 |
| `--tune full` | 1차 그리드 서치 315회부터 다시 실행 | Colab T4 약 5~6시간 (CPU 비권장) |
| `--quick` | 동작 확인용 (모든 학습 epoch 1번, 수치는 의미 없음) | CPU 약 10~15분 |
| `--out <폴더>` | 결과 폴더 (기본 `results/reproduce`) | |

시간 대부분은 LSTM(입력 168시간, 2층) 학습입니다.

## 5. 결과 확인

| 파일 | 내용 |
| --- | --- |
| `REPORT.md` | 최종 Test 비교표 (베이스라인 2 + 딥러닝 4) 와 오류분석 요약 |
| `run_manifest.json` | 실행 시각, 라이브러리 버전, git 커밋, 데이터 지문, 단계별 소요 시간, 주요 결과 |
| `final/test_summary.csv` | 최종 비교표 원본 |
| `final/test_predictions.csv` | Test 시간별 실제·예측·이상 확률 |
| `final/analysis_*/figures/` | 영향 변수, FN·FP 히트맵, 조건별 놓침률, Test 타임라인 그림 |
| `run_all.log` | 실행 로그 전체 |

팀 실험 결과(`results/final/`)와 비교하면 같은 값이 나와야 합니다. 다만 CPU 와 GPU, 운영체제에 따라 부동소수점 계산 순서가 달라 **소수점 아래 값이 조금 다를 수 있습니다.** 팀 결과는 로컬 CPU(Windows 11, Python 3.11.9)에서 만들었습니다.

## 6. Colab GPU 로 실행하기

`notebooks/colab_grid_search.ipynb` 를 Colab 에 올리고 1~4번 칸(Drive 연결, GPU 확인, 코드 받기)을 실행한 뒤 새 칸에서:

```python
%cd {CODE_DIR}
!pip -q install -r requirements.txt
!python run_all.py --features "{DRIVE_ROOT}/data/okm_features2_2021.csv" --out "{DRIVE_ROOT}/results/reproduce"
```

결과가 Drive 에 저장되어 세션이 끊겨도 남습니다.

## 7. 재현성을 위해 지킨 것

- **seed 42 고정** (`keras.utils.set_random_seed`), 학습·검증·Test 구간을 날짜로 고정
- **시간 순서 유지**: 미래 데이터로 학습하지 않음. Scaler 는 학습 구간으로만 fit
- **설정 선택도 코드로 고정**: 그리드 서치 결과 CSV 에서 "RMSE 최저 대비 2% 이내 중 피크 MAE 최소" 규칙으로 자동 선택
- **이상 기준·경보 기준도 코드로 계산**: 177 kW 는 `peak_threshold_from_data()`, 경보 확률은 검증 F1 최대값
- **데이터 지문 확인**과 **실행 기록(run_manifest.json)** 자동 저장
- 라이브러리 버전 고정 (`requirements.txt`)
