# OKM Test 결과 대시보드

`pip install -r dashboard/requirements.txt` 후 저장소 루트에서 `python dashboard/app.py`로 실행합니다.

2021년 홀드아웃 Test 703시간에 저장된 예측을 표시합니다. 최적 후보인 LSTM·TCN 가중 앙상블과 다른 네 후보의 RMSE를 비교합니다. 신경망 가중치가 보존되지 않았으므로 이 화면은 신규 데이터 추론 또는 실시간 운영 대시보드가 아닙니다. 피크 기준 177 kW와 경보 확률 기준값은 저장된 검증 결과에서 읽습니다.
