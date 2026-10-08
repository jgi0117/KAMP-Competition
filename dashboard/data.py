"""Validated held-out Test evidence for the five base-nine models."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]
PEAK_KW = 177.0
CONTROL_SIGMA = 3.0


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


def energy_figure(hours: int = 6, replay_step: int = 0) -> go.Figure:
    as_of = replay_timestamp(replay_step)
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


def cause_figure(height: int = 330, selected_time: str | None = None) -> go.Figure:
    frame = _importance().pivot(index="feature", columns="model", values="rmse_increase_mean").fillna(0)
    frame["mean"] = frame.mean(axis=1)
    frame = frame.sort_values("mean").tail(9)
    fig = go.Figure()
    for model, color in (("xgboost", BLUE), ("lightgbm", TEAL)):
        fig.add_trace(go.Bar(
            x=frame.get(model, pd.Series(0, index=frame.index)), y=frame.index,
            name=model, orientation="h", marker_color=color,
            hovertemplate="%{y}<br>RMSE 증가 %{x:.2f} kW<extra>" + model + "</extra>",
        ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="순열 교란 시 RMSE 증가 (kW)")
    fig.update_layout(barmode="group")
    if selected_time:
        fig.add_annotation(
            x=1, y=1.12, xref="paper", yref="paper",
            text=f"선택 시점 {selected_time} · 전역 중요도 참고",
            showarrow=False, font={"color": TEAL, "size": 12},
        )
    return _base_layout(fig, height, {"l": 125, "r": 20, "t": 30, "b": 45})


def importance_figure() -> go.Figure:
    frame = _importance().groupby("feature", as_index=False).agg(
        importance=("rmse_increase_mean", "mean")
    ).sort_values("importance").tail(9)
    fig = go.Figure(go.Bar(
        x=frame.importance, y=frame.feature, orientation="h",
        marker_color=[RED if value > 10 else TEAL for value in frame.importance],
        text=[f"{value:.2f}" for value in frame.importance], textposition="auto",
    ))
    fig.update_xaxes(title="평균 RMSE 증가 (kW)")
    return _base_layout(fig, 330, {"l": 125, "r": 20, "t": 18, "b": 45})


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

_FEATURE_SUMMARY = _importance().groupby("feature", as_index=False).agg(
    importance=("rmse_increase_mean", "mean")
).sort_values("importance", ascending=False).head(3)
FEATURE_ROWS = [
    (
        str(rank),
        row.feature,
        f"{row.importance:.2f} kW",
        "순열 교란",
        "RMSE 증가" if row.importance >= 0 else "RMSE 감소",
        "검증 완료",
        "danger" if rank == 1 else "caution",
    )
    for rank, row in enumerate(_FEATURE_SUMMARY.itertuples(), start=1)
]


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
