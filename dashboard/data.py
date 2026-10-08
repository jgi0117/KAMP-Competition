"""Validated held-out Test evidence and local tree explanations."""

from functools import lru_cache
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PEAK_KW = 177.0
CONTROL_SIGMA = 3.0
TREE_LOOKBACK = 168


def load_evidence():
    comparison = pd.read_csv(ROOT / "results/model_comparison.csv", encoding="utf-8-sig")
    expected = {"lstm", "tcn", "ensemble", "xgboost", "lightgbm"}
    if set(comparison.model) != expected:
        raise ValueError(f"Expected five model results, found {set(comparison.model)}")
    if not comparison.seed.eq(42).all() or not comparison.n_features.eq(9).all():
        raise ValueError("Comparison must use seed 42 and nine input variables")

    predictions = pd.read_csv(ROOT / "neural/results/final/test_predictions.csv")
    predictions["datetime"] = pd.to_datetime(predictions.datetime)
    predictions = predictions.sort_values("datetime").reset_index(drop=True)
    if len(predictions) != 703 or predictions.datetime.duplicated().any():
        raise ValueError("Expected 703 unique held-out Test hours")
    if not predictions.datetime.diff().dropna().le(pd.Timedelta(hours=18)).all():
        raise ValueError("Unexpected gap in Test predictions")

    settings = pd.read_csv(ROOT / "neural/results/final/alert_settings.csv", encoding="utf-8-sig").set_index("model")
    weights = pd.read_csv(ROOT / "neural/results/final/ensemble_weights.csv", encoding="utf-8-sig").set_index("model")
    cutoff = float(settings.loc["ensemble", "alert_cutoff"])
    residual_sigma = float(settings.loc["ensemble", "sigma"])
    predictions["predicted"] = predictions["pred_ensemble_seed42"]
    predictions["probability"] = predictions["prob_ensemble_seed42"]
    predictions["control_upper"] = predictions.predicted + CONTROL_SIGMA * residual_sigma
    predictions["control_lower"] = np.maximum(
        0.0, predictions.predicted - CONTROL_SIGMA * residual_sigma
    )
    predictions["control_violation"] = (
        (predictions.actual > predictions.control_upper)
        | (predictions.actual < predictions.control_lower)
    )
    predictions.attrs["residual_sigma"] = residual_sigma
    predictions["actual_peak"] = predictions.actual >= PEAK_KW
    predictions["alert"] = predictions.probability >= cutoff
    predictions["outcome"] = np.select(
        [predictions.actual_peak & ~predictions.alert,
         ~predictions.actual_peak & predictions.alert,
         predictions.actual_peak & predictions.alert],
        ["FN", "FP", "TP"], default="TN",
    )
    winner = comparison.loc[comparison.rmse.idxmin()]
    if winner.model != "ensemble":
        raise ValueError(f"Dashboard expects ensemble winner; found {winner.model}")
    return comparison, predictions, cutoff, weights.weight.to_dict()


# Shared visual tokens from the existing dashboard design system.
CHARCOAL = "#0C1822"
TEXT = "#E7EEF3"
BLUE = "#4C8DFF"
TEAL = "#21C7BA"
AMBER = "#E5AA3A"
RED = "#F0525F"
MUTED = "#91A3AE"
GRID = "rgba(145, 163, 174, 0.14)"


COMPARISON, TEST, ALERT_CUTOFF, WEIGHTS = load_evidence()
BEST = COMPARISON.set_index("model").loc["ensemble"]
RESIDUAL_SIGMA = float(TEST.attrs["residual_sigma"])
REPLAY_START_INDEX = 24
REFERENCE_TIME = TEST.iloc[REPLAY_START_INDEX].datetime
TEST_START = TEST.datetime.min()
TEST_END = TEST.datetime.max()


def _base_layout(fig: go.Figure, height: int, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=margin or {"l": 45, "r": 20, "t": 18, "b": 38},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Pretendard, SUIT, Noto Sans KR, Malgun Gothic, sans-serif", "color": TEXT, "size": 14},
        hoverlabel={"bgcolor": CHARCOAL, "bordercolor": TEAL, "font_color": TEXT, "font_size": 14},
        legend={"orientation": "h", "y": 1.12, "x": 0},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, tickfont={"color": MUTED}, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, tickfont={"color": MUTED}, zeroline=False)
    return fig


def replay_timestamp(step: int = 0) -> pd.Timestamp:
    available = len(TEST) - REPLAY_START_INDEX
    index = REPLAY_START_INDEX + (max(0, int(step or 0)) % available)
    return pd.Timestamp(TEST.iloc[index].datetime)


def energy_frame(as_of: pd.Timestamp | None = None, hours: int = 24) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of or REFERENCE_TIME)
    start = as_of - pd.Timedelta(hours=hours)
    return TEST.loc[TEST.datetime.between(start, as_of)].copy()


def energy_figure(hours: int = 6, replay_step: int = 0, as_of=None) -> go.Figure:
    as_of = pd.Timestamp(as_of) if as_of is not None else replay_timestamp(replay_step)
    frame = energy_frame(as_of, hours)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame.datetime, y=frame.actual, name="실측 전력", mode="lines+markers",
        line={"color": TEXT, "width": 2.4}, marker={"size": 4},
    ))
    fig.add_trace(go.Scatter(
        x=frame.datetime, y=frame.predicted, name="앙상블 예측", mode="lines",
        line={"color": TEAL, "width": 2.2},
    ))
    fig.add_trace(go.Scatter(
        x=frame.datetime, y=frame.control_lower, name="3σ 관리하한", mode="lines",
        line={"color": "rgba(76,141,255,0.55)", "width": 1, "dash": "dot"},
    ))
    fig.add_trace(go.Scatter(
        x=frame.datetime, y=frame.control_upper, name="3σ 관리상한", mode="lines",
        line={"color": "rgba(76,141,255,0.75)", "width": 1, "dash": "dot"},
        fill="tonexty", fillcolor="rgba(76,141,255,0.08)",
    ))
    alerts = frame.loc[frame.alert]
    fig.add_trace(go.Scatter(
        x=alerts.datetime, y=alerts.actual, name="피크 경보", mode="markers",
        marker={"color": RED, "size": 8, "symbol": "diamond"},
    ))
    misses = frame.loc[frame.outcome.eq("FN")]
    fig.add_trace(go.Scatter(
        x=misses.datetime, y=misses.actual, name="미탐지", mode="markers",
        marker={"color": AMBER, "size": 13, "symbol": "x"},
    ))
    fig.add_hline(
        y=PEAK_KW, line_dash="dash", line_color=RED, line_width=1.3,
        annotation_text=f"피크 기준 {PEAK_KW:.0f} kW",
        annotation_font={"color": "#FF7C85", "size": 12},
    )
    fig.add_vline(
        x=as_of, line_color="#7C929F", line_width=1, line_dash="dot",
    )
    fig.update_yaxes(title="전력 (kW)", rangemode="tozero")
    fig.update_xaxes(tickformat="%H:%M\n%m/%d")
    fig.update_layout(clickmode="event+select", hovermode="x unified")
    return _base_layout(fig, 470)


def _importance() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "docs/report/evidence/grouped_permutation_importance.csv",
        encoding="utf-8-sig",
    )
    required = {"model", "feature", "rmse_increase_mean", "rmse_increase_std"}
    if not required.issubset(frame.columns):
        raise ValueError("Permutation-importance evidence has an unexpected schema")
    return frame


@lru_cache(maxsize=1)
def _tree_explanation_assets():
    """Load the exact tree-model inputs and saved fitted models once."""
    from neural.core import data_pipeline as dp

    manifest = json.loads((ROOT / "results/tree_models/manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["feature_columns"])
    if features != list(dp.FEATURE_COLUMNS) or int(manifest["lookback"]) != TREE_LOOKBACK:
        raise ValueError("Tree explanation contract differs from the saved model manifest")
    history = dp.load_data(ROOT / "data/okm_cleaned_2021.csv")
    models = {
        name: joblib.load(ROOT / f"results/tree_models/models/{name}_seed42.joblib")
        for name in ("xgboost", "lightgbm")
    }
    return history, features, models


def _explanation_timestamp(value=None) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp(REFERENCE_TIME)
    if isinstance(value, pd.Timestamp):
        timestamp = value
    else:
        text = str(value).replace("T", " ")
        if len(text) <= 11 and "/" in text:
            text = f"{TEST_START.year}/{text}"
        timestamp = pd.Timestamp(text)
    if not TEST.datetime.eq(timestamp).any():
        raise ValueError(f"Selected explanation time is outside the held-out Test set: {timestamp}")
    return timestamp


@lru_cache(maxsize=256)
def local_tree_importance(selected_time=None) -> pd.DataFrame:
    """Return grouped native TreeSHAP contributions for one Test timestamp.

    Each tree model consumes 168 hourly lags × nine features. Native per-cell
    contributions are summed over the 168 lags so the nine displayed feature
    values remain additive to the model prediction together with the bias.
    """
    import xgboost as xgb

    timestamp = _explanation_timestamp(selected_time)
    history, features, models = _tree_explanation_assets()
    positions = np.flatnonzero(history.datetime.to_numpy() == np.datetime64(timestamp))
    if len(positions) != 1 or positions[0] < TREE_LOOKBACK:
        raise ValueError(f"Cannot build a {TREE_LOOKBACK}-hour window for {timestamp}")
    position = int(positions[0])
    window = history.iloc[position - TREE_LOOKBACK:position][features].to_numpy(dtype=np.float32)
    flattened = np.ascontiguousarray(window.reshape(1, -1))

    xgboost_model = models["xgboost"]
    lightgbm_model = models["lightgbm"]
    contributions = {
        "xgboost": xgboost_model.get_booster().predict(
            xgb.DMatrix(flattened), pred_contribs=True
        )[0],
        "lightgbm": np.asarray(lightgbm_model.predict(flattened, pred_contrib=True))[0],
    }
    predictions = {
        "xgboost": float(xgboost_model.predict(flattened)[0]),
        "lightgbm": float(lightgbm_model.predict(flattened)[0]),
    }
    rows = []
    for model, values in contributions.items():
        feature_values, bias = values[:-1], float(values[-1])
        grouped = feature_values.reshape(TREE_LOOKBACK, len(features)).sum(axis=0)
        if not np.isclose(float(grouped.sum() + bias), predictions[model], atol=1e-3):
            raise ValueError(f"{model} TreeSHAP contributions do not reconcile to its prediction")
        rows.extend(
            {"model": model, "feature": feature, "contribution_kw": float(contribution),
             "latest_value": float(window[-1, index]), "prediction_kw": predictions[model],
             "bias_kw": bias, "timestamp": timestamp}
            for index, (feature, contribution) in enumerate(zip(features, grouped))
        )
    return pd.DataFrame(rows)


def cause_figure(height: int = 330, selected_time: str | None = None) -> go.Figure:
    timestamp = _explanation_timestamp(selected_time)
    frame = local_tree_importance(timestamp).pivot(
        index="feature", columns="model", values="contribution_kw"
    ).fillna(0)
    frame["magnitude"] = frame[["xgboost", "lightgbm"]].abs().mean(axis=1)
    frame = frame.sort_values("magnitude").tail(9)
    fig = go.Figure()
    for model, color in (("xgboost", BLUE), ("lightgbm", TEAL)):
        fig.add_trace(go.Bar(
            x=frame.get(model, pd.Series(0, index=frame.index)), y=frame.index,
            name=model, orientation="h", marker_color=color,
            hovertemplate="%{y}<br>예측 기여 %{x:+.2f} kW<extra>" + model + "</extra>",
        ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="해당 시점 예측 기여 (kW)", zeroline=True, zerolinecolor=MUTED)
    fig.update_layout(barmode="group")
    return _base_layout(fig, height, {"l": 125, "r": 20, "t": 30, "b": 45})


def importance_figure(selected_time=None) -> go.Figure:
    frame = local_tree_importance(_explanation_timestamp(selected_time)).groupby(
        "feature", as_index=False
    ).agg(importance=("contribution_kw", "mean"))
    frame["magnitude"] = frame.importance.abs()
    frame = frame.sort_values("magnitude").tail(9)
    fig = go.Figure(go.Bar(
        x=frame.importance, y=frame.feature, orientation="h",
        marker_color=[RED if value > 0 else BLUE for value in frame.importance],
        text=[f"{value:+.2f}" for value in frame.importance],
        textposition=["inside" if abs(value) >= 1 else "outside" for value in frame.importance],
        insidetextanchor="end",
        cliponaxis=False,
    ))
    fig.update_xaxes(title="두 트리 모델 평균 예측 기여 (kW)", zeroline=True, zerolinecolor=MUTED)
    return _base_layout(fig, 330, {"l": 125, "r": 55, "t": 18, "b": 45})


def feature_rows(selected_time=None) -> list[tuple[str, ...]]:
    frame = local_tree_importance(_explanation_timestamp(selected_time)).groupby(
        "feature", as_index=False
    ).agg(importance=("contribution_kw", "mean"))
    frame["magnitude"] = frame.importance.abs()
    frame = frame.sort_values("magnitude", ascending=False).head(3)
    return [
        (
            str(rank), row.feature, f"{row.importance:+.2f} kW", "TreeSHAP 합산",
            "예측 증가" if row.importance >= 0 else "예측 감소", "계산 완료",
            "danger" if rank == 1 else "caution",
        )
        for rank, row in enumerate(frame.itertuples(), start=1)
    ]


def model_comparison_figure() -> go.Figure:
    frame = COMPARISON.sort_values("rmse", ascending=False)
    labels = {"ensemble": "LSTM+TCN", "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM", "tcn": "TCN"}
    fig = go.Figure(go.Bar(
        x=frame.rmse, y=[labels.get(model, model) for model in frame.model],
        orientation="h",
        marker_color=[TEAL if model == "ensemble" else BLUE for model in frame.model],
        text=[f"{value:.2f}" for value in frame.rmse], textposition="auto",
        customdata=np.c_[frame.mae, frame.r2],
        hovertemplate="RMSE %{x:.2f} kW<br>MAE %{customdata[0]:.2f} kW<br>R² %{customdata[1]:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title="Test RMSE (kW)", rangemode="tozero")
    return _base_layout(fig, 390, {"l": 95, "r": 20, "t": 20, "b": 45})


def error_heatmap_figure() -> go.Figure:
    frame = pd.read_csv(ROOT / "docs/report/evidence/test_error_by_hour.csv", encoding="utf-8-sig")
    z = [frame.false_positive.to_list(), frame.false_negative.to_list()]
    fig = go.Figure(go.Heatmap(
        z=z, x=frame.hour, y=["오경보(FP)", "미탐지(FN)"],
        colorscale=[[0, "#10212D"], [0.5, AMBER], [1, RED]],
        text=z, texttemplate="%{text}", colorbar={"title": "건"},
        hovertemplate="%{y}<br>%{x}시: %{z}건<extra></extra>",
    ))
    fig.update_xaxes(title="시간대", dtick=1)
    return _base_layout(fig, 290, {"l": 95, "r": 35, "t": 20, "b": 45})


SUMMARY = {
    "model": "LSTM + TCN",
    "rmse": float(BEST.rmse), "mae": float(BEST.mae), "r2": float(BEST.r2),
    "alert_f1": float(BEST.alert_f1),
    "alert_precision": float(BEST.alert_precision),
    "alert_recall": float(BEST.alert_recall),
    "fn": int(BEST.fn), "fp": int(BEST.fp),
    "peak_count": int(TEST.actual_peak.sum()), "alert_count": int(TEST.alert.sum()),
    "max_actual": float(TEST.actual.max()), "max_predicted": float(TEST.predicted.max()),
    "peak_kw": PEAK_KW, "cutoff": ALERT_CUTOFF,
    "test_start": TEST_START, "test_end": TEST_END, "test_hours": len(TEST),
}

FEATURE_ROWS = feature_rows()


def _action_rows() -> list[dict]:
    review = TEST.loc[TEST.outcome.isin(["FN", "FP"])].copy()
    review = review.sort_values(["outcome", "probability"], ascending=[True, False]).head(8)
    labels = {"FN": "미탐지", "FP": "오경보"}
    return [
        {"시각": row.datetime.strftime("%m/%d %H:%M"), "단계": labels[row.outcome],
         "전력": f"{row.actual:.1f} kW", "담당자": "미지정",
         "조치": "사후 모델 검토", "상태": "검토 대상"}
        for row in review.itertuples()
    ]


ACTION_ROWS = _action_rows()
