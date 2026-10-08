"""Evidence-backed Korean prose for the six competition evaluation sections."""

from pathlib import Path
import json

import pandas as pd


def build_chapters(root: Path, scores: pd.DataFrame):
    evidence = root / "docs/report/evidence"
    audit = json.loads((evidence / "data_diagnostics.json").read_text(encoding="utf-8"))
    eda = json.loads((evidence / "eda_summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(evidence / "grid_search_summary.csv", encoding="utf-8-sig")
    importance = pd.read_csv(evidence / "grouped_permutation_importance.csv", encoding="utf-8-sig")
    interaction = pd.read_csv(evidence / "production_time_interaction.csv", encoding="utf-8-sig")
    errors = pd.read_csv(evidence / "test_error_by_hour.csv", encoding="utf-8-sig").set_index("hour")
    misses = pd.read_csv(evidence / "test_missed_peaks.csv", encoding="utf-8-sig")
    settings = pd.read_csv(root / "neural/results/final/alert_settings.csv", encoding="utf-8-sig").set_index("model")
    weights = pd.read_csv(root / "neural/results/final/ensemble_weights.csv", encoding="utf-8-sig").set_index("model")
    e = scores.loc["ensemble"]
    base = audit["baselines"]["lag168"]
    recent = audit["baselines"]["lag1"]
    gain = (base["rmse"] - e.rmse) / base["rmse"]

    def stage(model, number):
        return grid.loc[grid.model.eq(model) & grid.stage.astype(str).eq(str(number))].iloc[0]

    def tree(model):
        return grid.loc[grid.model.eq(model)].iloc[-1]

    def impact(model, feature):
        return float(importance.loc[importance.model.eq(model) & importance.feature.eq(feature),
                                    "rmse_increase_mean"].iloc[0])

    def group(morning, high):
        return interaction.loc[interaction.morning.eq(morning) & interaction.high_production.eq(high)].iloc[0]

    low = group(False, False)
    active = group(True, True)
    miss = misses.iloc[0]
    return [
        [
            "◦ 1.1 제조 데이터와 예측 과제",
            "- 관측 단위는 공장 전체의 시간별 제조공정 기록이다. 2021년 1월 1일~9월 14일 총 6,168시간에 대해 생산량, 기상, 인력 및 15·30·45·60분 전력 계측을 시간 키로 연결한다. 네 전력 계측의 실수 평균을 다음 시간의 예측 목표(kW)로 정의하였다.",
            "- 일별 평균 전력과 생산량을 함께 그려 1~9월 조업 수준의 변동을 살폈다. 생산량이 낮은 날에도 전력 소비가 이어지는 구간과 8월 말 셧다운 구간을 구분하여 해석했다.",
            "- 과거 전력은 설비 부하의 연속성, 생산량은 작업 강도, 기온·풍속·습도·강수량은 외부 조건, 시간 주기와 주말 구분은 조업 일정을 설명한다. 공장 단위 집계에서 생산 활동과 전력 사용 패턴을 함께 파악해 피크 사전경보에 연결하였다.",
            "◦ 1.2 결측·중복·시간 이상 진단과 정제",
            f"- 원본 {audit['raw_rows']:,}행·{audit['raw_columns']}열에서 풍속 {audit['raw_missing']['풍속']}건, 강수량 {audit['raw_missing']['강수량']}건, 공장인원 {audit['raw_missing']['공장인원']}건의 결측과 시간 표기 이상 {audit['restored_hours']}건을 확인하였다. 완전 중복 행과 중복 시각은 각각 {audit['duplicate_raw_rows']}건이었다.",
            f"- 시간 표기를 날짜·요일과 연속성에 맞춰 복원하고 원래 값과 복원 여부를 함께 기록했다. 공장인원 결측은 0으로, 풍속·강수량은 시간 순서의 선형보간으로 보정하였다. 네 계측값으로 전력 실수 평균을 재계산하고 공장 셧다운 {audit['shutdown_hours']}시간을 표기했다. 정제 후 {audit['clean_rows']:,}행·{audit['clean_columns']}열의 결측은 {audit['clean_missing']}건이다.",
            f"- 원본 결측 위치와 셧다운 전후 전력 추이를 확인했다. 셧다운을 제외한 생산량 0인 {eda['idle_hours']:,}시간의 평균 전력은 {eda['idle_mean_kw']:.1f} kW였다. 따라서 생산 중단과 공장 셧다운의 전력 상태를 구분해 진단했다.",
            "◦ 1.3 입력과 피크 불균형",
            "- 공통 입력은 과거 전력_평균_실수, 생산량, 기온, 풍속, 습도, 강수량, 시간_sin, 시간_cos, 주말여부의 9개다. 시간_sin·cos와 주말여부는 날짜·시간에서 동일한 규칙으로 생성한다. 모든 모델에 동일한 변수 정의와 시각 경계를 적용하였다.",
            f"- 요일×시간 평균 전력에서는 평일 11시 {eda['weekday_11_kw']:.1f} kW에서 12시 {eda['weekday_12_kw']:.1f} kW로 낮아지는 조업 리듬이 나타났다. 생산량과 전력의 산점도에서는 무생산 대기 부하와 높은 생산량의 고부하를 함께 확인했다. 전력의 1시간·168시간 시차 상관은 각각 {eda['lag_correlations']['1']:.3f}·{eda['lag_correlations']['168']:.3f}으로, 짧은 연속성과 주간 반복성이 입력 길이 비교의 근거가 되었다.",
            f"- 학습 기간의 높은 전력 구간을 기준으로 피크를 177 kW 이상으로 고정했다. 최종 Test의 유효한 {audit['test_hours']}시간 중 피크는 {audit['test_peaks']}시간({audit['test_peak_rate']:.1%})으로 소수 사례다. 따라서 전체 전력 오차와 함께 피크 F1·재현율·오경보를 별도로 평가하였다.",
            "◦ 1.4 시계열 검증 설계",
            "- 검증은 5월 16일~6월 15일, 6월 16일~7월 15일, 7월 16일~8월 15일의 확장형 3개 fold로 구성했다. 각 검증 시작 직전 14일은 조기 종료 판단 구간으로 두고 이전 기간으로 학습했다. 최종 Test는 8월 16일~9월 14일에 고정하여 탐색과 분리하였다.",
        ],
        [
            "◦ 2.1 기준선과 공통 선택 규칙",
            f"- 같은 Test 703시간에서 전주 같은 시각의 전력을 예측값으로 쓰는 기준선의 RMSE는 {base['rmse']:.2f} kW, MAE는 {base['mae']:.2f} kW, 피크 F1은 {base['f1']:.3f}이다. 직전 시간 값을 쓰는 기준선은 RMSE {recent['rmse']:.2f} kW, F1 {recent['f1']:.3f}이다. 두 기준선도 검증 잔차로 확률을 계산하고 검증 구간에서 경보 기준을 선택한 뒤 동일 Test에서 평가했다.",
            "- seed 42와 3개 시간순 fold를 공통으로 적용했다. 각 후보의 검증 RMSE 평균 최솟값에서 2% 이내인 설정을 모은 다음 피크 시각 MAE가 가장 낮은 설정을 채택했다. 동률에서는 전체 RMSE가 낮은 설정을 우선한다. 경보 확률의 기준값도 검증 자료의 F1으로 정했다.",
            "◦ 2.2 LSTM·TCN의 4단계 탐색",
            "- LSTM은 입력 길이 → 은닉 유닛·층 → 학습률·dropout → 배치 크기의 4단계로 탐색했다. 단계별 전체 후보값과 최종 선택값은 표 1에 한눈에 비교하도록 정리했다. 중복 설정을 합쳐 22개 고유 후보·66개 fold 결과를 기록했다.",
            f"- LSTM의 단계별 선택 검증 RMSE는 {stage('lstm',1).validation_rmse:.2f} → {stage('lstm',2).validation_rmse:.2f} → {stage('lstm',3).validation_rmse:.2f} → {stage('lstm',4).validation_rmse:.2f} kW였다. 최종 입력 168시간, 유닛 128, 2층, dropout 0.3, 학습률 0.001, 배치 16을 선택했다. 최종 설정의 검증 피크 MAE는 {stage('lstm',4).validation_peak_mae:.2f} kW다.",
            "- TCN도 입력 길이 → 필터·커널·팽창 패턴 → 학습률·dropout → 배치 크기 순으로 탐색했다. 표 2에 파라미터별 후보값, 34개 고유 후보·102개 fold, 최종 설정을 따로 정리해 선택 과정을 빠르게 파악할 수 있게 했다.",
            f"- TCN의 단계별 선택 검증 RMSE는 {stage('tcn',1).validation_rmse:.2f} → {stage('tcn',2).validation_rmse:.2f} → {stage('tcn',3).validation_rmse:.2f} → {stage('tcn',4).validation_rmse:.2f} kW였다. 입력 24시간, 필터 128, 커널 2, 팽창 auto+1, dropout 0.3, 학습률 0.001, 배치 32를 선택했다.",
            "◦ 2.3 트리 모델의 전체 조합 탐색",
            "- XGBoost는 max_depth 3·5·7, 학습률 0.01·0.05·0.1, 트리 수 200·500·1000, min_child_weight 1·5, colsample_bytree 0.8·1.0의 108개 조합을 모두 평가했다. 각 조합에 3개 fold를 적용한 324개 fold 결과와 선택값을 표 3에 정리했다.",
            f"- XGBoost 최저 검증 RMSE는 {tree('xgboost').lowest_rmse:.2f} kW였고 2% 범위에 {int(tree('xgboost').rmse_2pct_candidates)}개 후보가 남았다. 피크 MAE {tree('xgboost').validation_peak_mae:.2f} kW인 28번 후보(max_depth 3, 학습률 0.1, 트리 200, min_child_weight 5, 열 표본 1.0)를 선택했다. 선택 후보의 검증 RMSE는 {tree('xgboost').validation_rmse:.2f} kW다.",
            "- LightGBM은 num_leaves 7·15·31, 학습률 0.01·0.05·0.1, 트리 수 200·500·1000, min_child_samples 20·50, colsample_bytree 0.8·1.0의 108개 조합×3 fold를 평가했으며 전체 범위와 선택값은 표 4에 정리했다.",
            f"- LightGBM 최저 검증 RMSE는 {tree('lightgbm').lowest_rmse:.2f} kW였고 2% 범위에 {int(tree('lightgbm').rmse_2pct_candidates)}개 후보가 남았다. 피크 MAE {tree('lightgbm').validation_peak_mae:.2f} kW인 55번 후보(잎 15, 학습률 0.05, 트리 500, 최소 자식 표본 50, 열 표본 0.8)를 선택했다. 선택 후보의 검증 RMSE는 {tree('lightgbm').validation_rmse:.2f} kW다.",
            "◦ 2.4 앙상블과 최종 Test 평가",
            f"- 검증 MSE의 역수로 LSTM {weights.loc['lstm','weight']:.3f}, TCN {weights.loc['tcn','weight']:.3f}의 가중치를 정했다. 앙상블 검증 RMSE는 {settings.loc['ensemble','val_rmse']:.2f} kW, 검증 경보 F1은 {settings.loc['ensemble','val_alert_f1']:.3f}이었다. 검증 잔차 표준편차 {settings.loc['ensemble','sigma']:.2f} kW를 이용해 177 kW 초과 확률을 계산하고 경보 기준 {settings.loc['ensemble','alert_cutoff']:.2f}를 선택했다.",
            f"- Test RMSE/F1: LSTM {scores.loc['lstm','rmse']:.2f}/{scores.loc['lstm','alert_f1']:.3f}, TCN {scores.loc['tcn','rmse']:.2f}/{scores.loc['tcn','alert_f1']:.3f}, 앙상블 {e.rmse:.2f}/{e.alert_f1:.3f}, XGBoost {scores.loc['xgboost','rmse']:.2f}/{scores.loc['xgboost','alert_f1']:.3f}, LightGBM {scores.loc['lightgbm','rmse']:.2f}/{scores.loc['lightgbm','alert_f1']:.3f}. 앙상블이 RMSE와 경보 F1 모두 최고였다.",
            f"- Test 피크 49시간의 미탐지/오경보는 LSTM {int(scores.loc['lstm','fn'])}/{int(scores.loc['lstm','fp'])}건, TCN {int(scores.loc['tcn','fn'])}/{int(scores.loc['tcn','fp'])}건, 앙상블 {int(e.fn)}/{int(e.fp)}건, XGBoost {int(scores.loc['xgboost','fn'])}/{int(scores.loc['xgboost','fp'])}건, LightGBM {int(scores.loc['lightgbm','fn'])}/{int(scores.loc['lightgbm','fp'])}건이다. 경보 재현율과 오경보 부담을 함께 비교해 최종 후보를 결정했다.",
            f"- 앙상블의 Test MAE는 {e.mae:.2f} kW, R²는 {e.r2:.3f}, 피크 재현율은 {e.alert_recall:.1%}, 정밀도는 {e.alert_precision:.1%}다. 기준선 대비 RMSE를 {gain:.1%} 낮춰 전력 규모와 피크 사전경보를 함께 개선했다.",
        ],
        [
            "◦ 3.1 변수 영향과 공정 해석",
            f"- 보존된 트리 모델에서 각 변수의 과거 168시간 묶음을 다섯 번 섞어 Test RMSE 변화를 측정했다. 과거 전력을 섞으면 XGBoost +{impact('xgboost','전력_평균_실수'):.2f} kW, LightGBM +{impact('lightgbm','전력_평균_실수'):.2f} kW로 오차가 크게 늘었다. 공정의 연속 부하가 가장 강한 예측 단서라는 결과다.",
            f"- 시간_cos의 RMSE 증가량은 XGBoost +{impact('xgboost','시간_cos'):.2f} kW, LightGBM +{impact('lightgbm','시간_cos'):.2f} kW였고 XGBoost의 생산량은 +{impact('xgboost','생산량'):.2f} kW였다. 시간 주기와 생산 강도를 함께 확인해야 피크 대응이 정교해진다.",
            "◦ 3.2 생산 강도와 조업 시간의 상호작용",
            f"- Test에서 직전 시간 생산량 1,000 이하·오전 07~11시 외 조건의 피크율은 {int(low.peaks)}/{int(low.hours)}시간({low.peak_rate:.1%})이었다. 직전 생산량 1,000 초과와 오전 07~11시가 함께 나타난 조건은 {int(active.peaks)}/{int(active.hours)}시간({active.peak_rate:.1%})이었다. 이 조합을 조업 계획 확인의 우선순위로 삼는다.",
            "- 생산량과 시간의 결합은 개별 변수만 볼 때보다 운전 상황을 더 구체적으로 설명한다. 해당 교차 집계는 발생 조건의 연관성을 나타내며, 현장 조치는 작업 계획과 설비 상태 확인을 거쳐 결정한다.",
            "◦ 3.3 미탐지와 오경보의 집중 조건",
            f"- 최종 Test 피크 49시간에서 앙상블은 48시간을 경보하고 1시간을 놓쳤다. 미탐지 시각은 {miss.datetime}, 실측 {miss.actual:.1f} kW, 예측 {miss.pred_ensemble_seed42:.1f} kW, 피크 확률 {miss.prob_ensemble_seed42:.3f}이었다. 확률이 검증 기준 0.20 바로 아래에 있었고 직전 생산량은 {miss.previous_production:,.0f}이었다.",
            f"- Test 오경보는 {int(e.fp)}건이다. 시간대별로 09시·11시·13시에 각각 {int(errors.loc[9,'false_positive'])}건, 16시에 {int(errors.loc[16,'false_positive'])}건이 모였다. 경보 이후 최근 계측값과 생산계획을 함께 확인하는 절차가 필요한 구간이다.",
            "- 검증 2,208시간과 Test 703시간을 합친 오류 진단표에서는 피크 225시간 중 미탐지 57건, 오경보 151건이었다. 미탐지는 08시 11건·10시 10건·16시 9건에, 오경보는 13시 29건·09시 25건에 집중됐다. 이 시간대를 우선 모니터링 대상으로 정했다.",
        ],
        [
            "◦ 4.1 현장 의사결정 흐름",
            f"- 시간별 전력·생산·기상 데이터를 정제한 후 대시보드에 예상 전력, 177 kW 피크 확률, 관리상한(UCL)과 관리하한(LCL)을 함께 표시한다. 관리한계에는 Test 정보를 쓰지 않고 검증 구간 앙상블 잔차에서 구한 σ={settings.loc['ensemble','sigma']:.2f} kW를 고정해 적용한다.",
            f"- 시각 t의 관리상한은 예측값(t)+3σ, 관리하한은 max(0, 예측값(t)−3σ)로 계산한다. 현재 관리폭은 예측값 중심으로 ±{3 * settings.loc['ensemble','sigma']:.2f} kW이다. 정상 오차가 근사적으로 정규분포를 따른다면 약 99.7%가 이 구간에 들어오므로, 일시적 잡음보다 의미 있는 이탈을 선별하는 기준이 된다.",
            "- 실측이 UCL을 넘으면 예상보다 큰 부하 급증으로 보고, LCL 아래면 비정상 설비 정지·계측 오류·생산계획 변경 가능성을 점검한다. 관리한계 내부에서도 177 kW 초과 확률이 0.20 이상이면 수요 피크 경보를 별도로 유지해, 공정 이상 감지와 최대수요 관리를 서로 보완한다.",
            "- UCL 이탈 시에는 ① 최근 계측값과 센서 상태 확인, ② 당시 생산량·작업 지시 대조, ③ 부하가 큰 작업의 시작 시점 분산, ④ 품질·안전에 영향이 없는 비필수 설비의 가동 시간 조정 순으로 대응한다. 연속 2회 이상 이탈은 단발성 경보보다 높은 우선순위로 확인한다.",
            "- 관제 화면은 저장된 Test 703시간을 한 시간씩 재생한다. 재생 시각 t까지 실측을 표시하고 t+1의 앙상블 예측을 한 시간 앞서 보여 준다. 6·12·24시간 보기에서 예측선, 3σ 관리상·하한, 177 kW 피크 기준을 함께 읽어 다음 시간의 점검 필요성을 판단한다.",
            "◦ 4.2 조치 우선순위와 운영 기록",
            "- 오전 조업 시작과 직전 시간 생산량 1,000 초과가 겹치는 시각을 우선 확인한다. 오경보가 잦은 09·11·13시에는 예측값만으로 설비를 조정하기보다 작업 계획 및 현장 계측 확인을 함께 수행한다. 생산 안전과 품질 조건을 만족하는 조정안만 작업자가 승인한다.",
            f"- 모델 성능 화면은 같은 Test 703시간에서 다섯 후보의 RMSE를 비교하고 최종 앙상블의 F1 {e.alert_f1:.3f}, 재현율 {e.alert_recall:.1%}, 미탐지 {int(e.fn)}건, 오경보 {int(e.fp)}건을 함께 제시한다. 담당자는 평균 전력 오차와 경보 부담을 동시에 확인해 최종 모델 선정 근거를 검토한다.",
            "- 변수 기여도 화면은 저장된 XGBoost·LightGBM의 직전 168시간 입력에 대한 TreeSHAP을 각 변수별로 합산해 해당 시각 예측을 높이거나 낮춘 방향을 보여 준다. 두 트리 후보의 평균 기여도와 상위 3개 변수를 함께 표시해 공정·시간 조건을 점검한다. 설명값의 대상 모델과 기준 시각을 화면에 명시했다.",
            "- 오류 사례 검토 화면은 Test의 피크 49시간, 정탐 48건, 미탐지 1건, 오경보 85건을 표시한다. 미탐지·오경보 시각과 실측 전력을 목록에서 확인하고 입력값·경보 기준·직전 생산량을 체크리스트로 검토한다. 관제 화면의 조치 등록 창에서는 선택한 조치와 담당자를 현재 브라우저 세션의 목록에 추가해 사후 점검 항목을 정리한다.",
            f"- 평가 구간에서 앙상블은 피크 49시간 중 48시간을 포착하고 전주 같은 시각 기준선 대비 RMSE를 {gain:.1%} 낮췄다. 현장 실증에서는 피크 재현율, 불필요 확인 건수, 작업 변경 횟수, 실제 최대수요 전력 변화를 함께 관리한다.",
        ],
        [
            "◦ 5.1 제조공정 특성을 반영한 결합 방식",
            "- 시계열의 장기 패턴을 보는 LSTM(168시간)과 단기 변화를 보는 TCN(24시간)을 검증 오차의 역수로 결합했다. 두 관점의 가중 평균은 LSTM 단독보다 Test RMSE를 0.50 kW, TCN 단독보다 1.39 kW 낮췄다.",
            "- 셧다운 표기와 시간 복원 이력을 전처리 단계에 남겨 공정 중단·재가동을 구분했고, 직전 생산량과 오전 조업 시간의 교차 조건을 사전 점검 대상으로 연결했다. 이는 모델 예측을 실제 작업 순서에 맞추는 설계다.",
            "◦ 5.2 희소 피크에 맞춘 선택과 불확실성 표현",
            f"- Test 피크율 {audit['test_peak_rate']:.1%}에서 전체 평균 오차만 낮추는 대신 검증 RMSE 2% 범위의 후보 중 피크 MAE를 비교했다. 최종 경보는 검증 잔차 표준편차 {settings.loc['ensemble','sigma']:.2f} kW를 반영한 피크 확률과 F1 기준값 0.20으로 판단한다.",
            f"- 최종 앙상블의 Test F1 {e.alert_f1:.3f}은 전주 같은 시각 기준선 {base['f1']:.3f}보다 {round(e.alert_f1, 3)-round(base['f1'], 3):.3f}, 직전 시간 기준선 {recent['f1']:.3f}보다 {round(e.alert_f1, 3)-round(recent['f1'], 3):.3f} 높다. 각 차이는 표시한 세 자리 F1의 차이다. 미탐지 1건과 오경보 85건을 함께 공개해 재현율 중심 경보의 운영 비용을 판단할 수 있게 했다.",
        ],
        [
            "◦ 6.1 실행 환경과 파일 구성",
            "- data/의 원본·정제 CSV, src/preprocessing.py의 정제 규칙, model_search/의 모델별 탐색 범위, neural/core/의 신경망 구현·평가, scripts/의 학습·비교·보고서 자동화, results/의 모델 결과, docs/report·docs/presentation의 최종 문서를 역할별로 배치했다.",
            "- Python 통합 환경은 루트 requirements.txt 하나에 명시했다. seed 42, 데이터 해시, 9개 공통 입력변수 목록, 피크 기준, 5개 모델의 후보 선택 규칙은 결과 manifest에 기록했다.",
            "◦ 6.2 전처리부터 제출물 생성까지",
            "- 정제 검증 스크립트는 원본 6,168행에서 정제 결과를 다시 만들고 제공 CSV의 모든 열과 비교한다. 트리 모델 학습 스크립트는 XGBoost와 LightGBM의 108개 조합×3 fold 결과와 선택 모델을 생성한다.",
            "- 5개 모델 비교 검증 스크립트는 Test 시각·실측값 일치와 트리 전체 탐색 완료를 검사한다. 이어서 근거 생성, 변수 중요도, EDA, 도표, HWPX 생성·검증 순으로 실행한다. 대시보드 캡처 스크립트는 관제·모델 성능·변수 기여도·오류 검토의 실제 Dash 화면 네 장을 촬영해 보고서 그림으로 저장한다.",
            "- 표 6은 목적별 실행 명령과 입력·산출물을 한 흐름으로 정리한다. python scripts/reproduce_submission.py는 정제 검증, 결과 비교, 근거표·시각화·HWPX 생성과 검증을 연속 실행한다. --train-trees는 두 트리의 탐색·재학습을, --capture-dashboard는 실제 화면 캡처 갱신을 포함한다. 후보 설정, fold 결과 CSV, Test 예측, 모델 파일을 함께 보관해 선택 과정을 추적할 수 있다.",
        ],
    ]
