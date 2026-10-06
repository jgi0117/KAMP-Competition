# -*- coding: utf-8 -*-
"""전력사용량 예측 모델링 및 벤치마크 모듈.

이 모듈은 데이터 누수(Data Leakage), 과적합(Overfitting), 과소적합(Underfitting)을
원천 방어하면서 다각적 머신러닝/딥러닝 모델 군(선형, 트리, 앙상블, 신경망)을 학습하고
팀 공통 시계열 분할 기준(Train, Validation, Test)에 따라 성능을 정밀 평가합니다.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 프로젝트 루트 경로 등록
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from src.features import build_feature_pipeline, get_feature_target_split, split_train_val_test
    from src.config import LOCAL_DATA_DIR, REPORTS_DIR
except ImportError:
    from features import build_feature_pipeline, get_feature_target_split, split_train_val_test
    LOCAL_DATA_DIR = _ROOT_DIR / "data"
    REPORTS_DIR = _ROOT_DIR / "reports"


class DeepPowerMLP(nn.Module):
    """공장 전력 시계열 특성 학습용 심층 잔차 신경망 (Deep Residual MLP)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.out = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.ln1(self.fc1(x))))
        x = self.drop2(self.act2(self.ln2(self.fc2(x))))
        return self.out(x).squeeze(-1)


def train_deep_mlp(
    X_tr_sc: np.ndarray,
    y_tr: np.ndarray,
    X_va_sc: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.003,
    seed: int = 42
) -> DeepPowerMLP:
    """PyTorch MLP 모델을 조기 종료 체크포인트 방식으로 학습합니다."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DeepPowerMLP(in_dim=X_tr_sc.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    ds_tr = TensorDataset(torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    best_va_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            p_va = model(torch.tensor(X_va_sc, dtype=torch.float32)).numpy()
            va_rmse = np.sqrt(mean_squared_error(y_va, p_va))
            if va_rmse < best_va_loss:
                best_va_loss = va_rmse
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return model


def get_model_zoo() -> Dict[str, Dict[str, Any]]:
    """다양한 가설과 정규화 전략을 가진 후보 모델 딕셔너리를 반환합니다."""
    return {
        "Ridge (L2 Linear)": {
            "model": Ridge(alpha=50.0),
            "input_type": "scaled",
            "desc": "선형 결합 기반 기본 베이스라인 (비선형 및 임계치 부하 과소적합 검증용)"
        },
        "ElasticNet": {
            "model": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "input_type": "scaled",
            "desc": "L1+L2 규제 기반 선형 회귀 (변수 선택 및 다중공선성 제어)"
        },
        "RandomForest": {
            "model": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
            "input_type": "imputed",
            "desc": "배깅 기반 앙상블 트리 (분산 감소, 외삽 및 계절 편향 과적합 점검용)"
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.035,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=1.5,
                random_state=42
            ),
            "input_type": "raw",
            "desc": "결측치 네이티브 수용 히스토그램 기반 GBDT (빠르고 안정적인 일반화)"
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                n_estimators=350,
                learning_rate=0.03,
                num_leaves=25,
                max_depth=5,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.5,
                reg_lambda=3.0,
                random_state=42,
                verbose=-1
            ),
            "input_type": "raw",
            "desc": "리프 중심(Leaf-wise) 고속 트리 부스팅 (비선형 상호작용 극대화)"
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=42,
                verbosity=0
            ),
            "input_type": "raw",
            "desc": "2차 테일러 전개 가중치 정규화 기반 Depth-wise GBDT"
        },
        "CatBoost": {
            "model": CatBoostRegressor(
                iterations=500,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=5.0,
                subsample=0.85,
                random_seed=42,
                verbose=0
            ),
            "input_type": "imputed",
            "desc": "대칭 트리(Oblivious Trees) 기반 부스팅 (과적합 원천 억제 및 최고 일반화)"
        }
    }


def evaluate_all_models(
    df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """모든 후보 모델 및 최적 앙상블을 훈련/평가하고 벤치마크 결과표를 생성합니다."""
    if df is None:
        df = build_feature_pipeline()

    df_tr, df_va, df_te = split_train_val_test(df)

    X_tr, y_tr = get_feature_target_split(df_tr)
    X_va, y_va = get_feature_target_split(df_va)
    X_te, y_te = get_feature_target_split(df_te)

    # 1. 결측치 보정 (Median Imputation: Train 기준으로만 Fit하여 누수 방지)
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
    X_va_imp = pd.DataFrame(imputer.transform(X_va), columns=X_va.columns)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns)

    # 2. 표준 정규화 (StandardScaler: Train 기준으로만 Fit하여 누수 방지)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_imp)
    X_va_sc = scaler.transform(X_va_imp)
    X_te_sc = scaler.transform(X_te_imp)

    zoo = get_model_zoo()
    results = []
    predictions = {}

    print(f"\n>>> [모델 벤치마크 시작] 총 {len(zoo) + 2}개 모델 학습 및 검증 진행 중...")

    for name, config in zoo.items():
        model = config["model"]
        itype = config["input_type"]
        t0 = time.time()

        if itype == "scaled":
            X_t, X_v, X_e = X_tr_sc, X_va_sc, X_te_sc
        elif itype == "imputed":
            X_t, X_v, X_e = X_tr_imp, X_va_imp, X_te_imp
        else:
            X_t, X_v, X_e = X_tr, X_va, X_te

        model.fit(X_t, y_tr)
        p_tr = model.predict(X_t)
        p_va = model.predict(X_v)
        p_te = model.predict(X_e)
        fit_time = time.time() - t0

        predictions[name] = {"va": p_va, "te": p_te}

        v_rmse = np.sqrt(mean_squared_error(y_va, p_va))
        v_mae = mean_absolute_error(y_va, p_va)
        v_r2 = r2_score(y_va, p_va)

        t_rmse = np.sqrt(mean_squared_error(y_te, p_te))
        t_mae = mean_absolute_error(y_te, p_te)
        t_r2 = r2_score(y_te, p_te)

        tr_rmse = np.sqrt(mean_squared_error(y_tr, p_tr))
        tr_r2 = r2_score(y_tr, p_tr)

        results.append({
            "Model": name,
            "Family": "Linear" if "Ridge" in name or "Elastic" in name else "Tree/GBDT",
            "Train_RMSE": tr_rmse,
            "Train_R2": tr_r2,
            "Val_RMSE": v_rmse,
            "Val_MAE": v_mae,
            "Val_R2": v_r2,
            "Test_RMSE": t_rmse,
            "Test_MAE": t_mae,
            "Test_R2": t_r2,
            "Fit_Time_s": fit_time,
            "Description": config["desc"]
        })
        print(f"    - {name:<22s} 완료 | Val RMSE: {v_rmse:.3f} kW | Test RMSE: {t_rmse:.3f} kW ({fit_time:.2f}s)")

    # 3. PyTorch Deep Residual MLP 모델 추가
    t0_dl = time.time()
    mlp_model = train_deep_mlp(X_tr_sc, y_tr.values, X_va_sc, y_va.values, epochs=40, lr=0.003)
    mlp_model.eval()
    with torch.no_grad():
        p_va_dl = mlp_model(torch.tensor(X_va_sc, dtype=torch.float32)).numpy()
        p_te_dl = mlp_model(torch.tensor(X_te_sc, dtype=torch.float32)).numpy()
        p_tr_dl = mlp_model(torch.tensor(X_tr_sc, dtype=torch.float32)).numpy()
    fit_time_dl = time.time() - t0_dl

    predictions["DeepPowerMLP (PyTorch)"] = {"va": p_va_dl, "te": p_te_dl}

    results.append({
        "Model": "DeepPowerMLP (PyTorch)",
        "Family": "Deep Learning",
        "Train_RMSE": np.sqrt(mean_squared_error(y_tr, p_tr_dl)),
        "Train_R2": r2_score(y_tr, p_tr_dl),
        "Val_RMSE": np.sqrt(mean_squared_error(y_va, p_va_dl)),
        "Val_MAE": mean_absolute_error(y_va, p_va_dl),
        "Val_R2": r2_score(y_va, p_va_dl),
        "Test_RMSE": np.sqrt(mean_squared_error(y_te, p_te_dl)),
        "Test_MAE": mean_absolute_error(y_te, p_te_dl),
        "Test_R2": r2_score(y_te, p_te_dl),
        "Fit_Time_s": fit_time_dl,
        "Description": "LayerNorm 및 SiLU 활성화 함수 적용 심층 잔차 신경망"
    })
    print(f"    - DeepPowerMLP (PyTorch) 완료 | Val RMSE: {np.sqrt(mean_squared_error(y_va, p_va_dl)):.3f} kW | Test RMSE: {np.sqrt(mean_squared_error(y_te, p_te_dl)):.3f} kW ({fit_time_dl:.2f}s)")

    # 4. Tri-GBDT 균형 앙상블 (HistGBM 0.4 + LightGBM 0.3 + CatBoost 0.3)
    p_va_ens = (
        0.40 * predictions["HistGradientBoosting"]["va"] +
        0.30 * predictions["LightGBM"]["va"] +
        0.30 * predictions["CatBoost"]["va"]
    )
    p_te_ens = (
        0.40 * predictions["HistGradientBoosting"]["te"] +
        0.30 * predictions["LightGBM"]["te"] +
        0.30 * predictions["CatBoost"]["te"]
    )
    p_tr_ens = (
        0.40 * zoo["HistGradientBoosting"]["model"].predict(X_tr) +
        0.30 * zoo["LightGBM"]["model"].predict(X_tr) +
        0.30 * zoo["CatBoost"]["model"].predict(X_tr_imp)
    )

    predictions["Tri-GBDT Ensemble"] = {"va": p_va_ens, "te": p_te_ens}

    results.append({
        "Model": "Tri-GBDT Ensemble (기존)",
        "Family": "Ensemble Blend",
        "Train_RMSE": np.sqrt(mean_squared_error(y_tr, p_tr_ens)),
        "Train_R2": r2_score(y_tr, p_tr_ens),
        "Val_RMSE": np.sqrt(mean_squared_error(y_va, p_va_ens)),
        "Val_MAE": mean_absolute_error(y_va, p_va_ens),
        "Val_R2": r2_score(y_va, p_va_ens),
        "Test_RMSE": np.sqrt(mean_squared_error(y_te, p_te_ens)),
        "Test_MAE": mean_absolute_error(y_te, p_te_ens),
        "Test_R2": r2_score(y_te, p_te_ens),
        "Fit_Time_s": 0.0,
        "Description": "HistGBM(0.4) + LightGBM(0.3) + CatBoost(0.3) 이종 트리 블렌딩"
    })
    print(f"    - Tri-GBDT Ensemble 완료 | Val RMSE: {np.sqrt(mean_squared_error(y_va, p_va_ens)):.3f} kW | Test RMSE: {np.sqrt(mean_squared_error(y_te, p_te_ens)):.3f} kW")

    # 5. Top-3 GBDT 정예 앙상블 (XGBoost 0.40 + CatBoost 0.30 + LightGBM 0.30)
    p_va_top3 = (
        0.40 * predictions["XGBoost"]["va"] +
        0.30 * predictions["CatBoost"]["va"] +
        0.30 * predictions["LightGBM"]["va"]
    )
    p_te_top3 = (
        0.40 * predictions["XGBoost"]["te"] +
        0.30 * predictions["CatBoost"]["te"] +
        0.30 * predictions["LightGBM"]["te"]
    )
    p_tr_top3 = (
        0.40 * zoo["XGBoost"]["model"].predict(X_tr) +
        0.30 * zoo["CatBoost"]["model"].predict(X_tr_imp) +
        0.30 * zoo["LightGBM"]["model"].predict(X_tr)
    )

    predictions["Top-3 GBDT Ensemble"] = {"va": p_va_top3, "te": p_te_top3}

    results.append({
        "Model": "Top-3 GBDT Ensemble (권장)",
        "Family": "Ensemble Blend",
        "Train_RMSE": np.sqrt(mean_squared_error(y_tr, p_tr_top3)),
        "Train_R2": r2_score(y_tr, p_tr_top3),
        "Val_RMSE": np.sqrt(mean_squared_error(y_va, p_va_top3)),
        "Val_MAE": mean_absolute_error(y_va, p_va_top3),
        "Val_R2": r2_score(y_va, p_va_top3),
        "Test_RMSE": np.sqrt(mean_squared_error(y_te, p_te_top3)),
        "Test_MAE": mean_absolute_error(y_te, p_te_top3),
        "Test_R2": r2_score(y_te, p_te_top3),
        "Fit_Time_s": 0.0,
        "Description": "XGBoost(0.40) + CatBoost(0.30) + LightGBM(0.30) 최우수 정예 블렌딩"
    })
    print(f"    - Top-3 GBDT Ensemble 완료 | Val RMSE: {np.sqrt(mean_squared_error(y_va, p_va_top3)):.3f} kW | Test RMSE: {np.sqrt(mean_squared_error(y_te, p_te_top3)):.3f} kW")

    res_df = pd.DataFrame(results).sort_values("Test_RMSE")

    # 예측값 데이터프레임 생성
    pred_te_df = df_te[["datetime", "날짜", "시간"]].copy()
    pred_te_df["실제값"] = y_te.values
    for mname, pdict in predictions.items():
        pred_te_df[f"예측_{mname}"] = pdict["te"]

    return res_df, predictions, pred_te_df


if __name__ == "__main__":
    benchmark_df, preds, pred_test_df = evaluate_all_models()
    print("\n" + "="*80)
    print("                     [전력 예측 다중 모델 종합 벤치마크 결과]                     ")
    print("="*80)
    cols_show = ["Model", "Family", "Val_RMSE", "Val_R2", "Test_RMSE", "Test_MAE", "Test_R2", "Fit_Time_s"]
    print(benchmark_df[cols_show].to_string(index=False))

    # 테스트셋 저장 (팀원 공유 및 분석용)
    out_pred = LOCAL_DATA_DIR / "okm_model_predictions_test.csv"
    pred_test_df.to_csv(out_pred, index=False, encoding="utf-8-sig")
    print(f"\n[저장 완료] 테스트셋 모델별 예측 결과 파일: {out_pred}")
