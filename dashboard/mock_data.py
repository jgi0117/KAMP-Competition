"""더미 데이터와 Plotly 차트를 생성하는 모듈.

실제 모델 파이프라인이 연결되기 전 레이아웃 검증을 위한 값만 담는다.
회귀 모델 비교 수치는 프로젝트 보고서의 테스트 결과를 사용한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


CHARCOAL = "#0C1822"
TEXT = "#E7EEF3"
BLUE = "#4C8DFF"
TEAL = "#21C7BA"
AMBER = "#E5AA3A"
RED = "#F0525F"
MUTED = "#91A3AE"
GRID = "rgba(145, 163, 174, 0.14)"

REFERENCE_TIME = pd.Timestamp("2024-10-15 14:30")


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
    fig.update_xaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, tickfont={"color": MUTED}, title_font={"color": MUTED}, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, linecolor=GRID, tickfont={"color": MUTED}, title_font={"color": MUTED}, zeroline=False)
    return fig


def energy_frame() -> pd.DataFrame:
    timestamps = pd.date_range(end=REFERENCE_TIME + pd.Timedelta(hours=1), periods=101, freq="15min")
    x = np.arange(len(timestamps))
    daily = 108 + 43 * np.sin((x - 11) / 96 * 2 * np.pi)
    work_cycle = 12 * np.sin(x / 13) + 5 * np.cos(x / 5)
    predicted = daily + work_cycle
    actual = predicted + 5 * np.sin(x / 3.7) - 3 * np.cos(x / 7)

    current_idx = len(timestamps) - 5
    transition_start = current_idx - 8
    actual[transition_start : current_idx + 1] = np.linspace(actual[transition_start], 148.2, 9)
    predicted[transition_start : current_idx + 1] = np.linspace(predicted[transition_start], 151.0, 9)
    actual[current_idx + 1 :] = np.nan
    future = predicted.copy()
    future[: current_idx + 1] = np.nan
    future[current_idx + 1 :] = np.array([158.7, 166.1, 176.8, 172.9])
    predicted[current_idx + 1 :] = np.nan

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "actual_kw": actual,
            "predicted_kw": predicted,
            "future_kw": future,
        }
    )


def energy_figure(hours: int = 6) -> go.Figure:
    frame = energy_frame()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame["actual_kw"], name="실제 전력",
            mode="lines", line={"color": TEXT, "width": 2.4},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame["predicted_kw"], name="예측 전력",
            mode="lines", line={"color": TEAL, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["timestamp"], y=frame["future_kw"], name="예측(미래)",
            mode="lines+markers", line={"color": TEAL, "width": 2.6, "dash": "dot"},
            marker={"size": 6, "line": {"color": CHARCOAL, "width": 1}},
        )
    )
    fig.add_hrect(y0=145, y1=160, fillcolor=AMBER, opacity=0.045, line_width=0)
    fig.add_hrect(y0=160, y1=230, fillcolor=RED, opacity=0.035, line_width=0)
    fig.add_hline(
        y=160,
        line_dash="dash",
        line_color=RED,
        line_width=1.4,
        annotation_text="위험 기준 160.0 kW",
        annotation_font={"color": "#FF7C85", "size": 12},
    )
    fig.add_vrect(
        x0=REFERENCE_TIME, x1=REFERENCE_TIME + pd.Timedelta(hours=1),
        fillcolor=TEAL, opacity=0.045, line_width=0,
        annotation_text="예측 구간",
        annotation_position="top right",
        annotation_font={"color": MUTED, "size": 11},
    )
    fig.add_vline(
        x=REFERENCE_TIME,
        line_color="#7C929F",
        line_width=1,
        line_dash="dot",
        annotation_text="현재 14:30",
        annotation_position="top left",
        annotation_font={"color": MUTED, "size": 11},
    )
    fig.add_annotation(
        x=REFERENCE_TIME + pd.Timedelta(minutes=45),
        y=176.8,
        text="예상 피크 15:15 · 176.8 kW",
        showarrow=True,
        arrowhead=0,
        arrowcolor=TEAL,
        ax=-92,
        ay=-34,
        bgcolor="#10252B",
        bordercolor=TEAL,
        borderwidth=1,
        borderpad=5,
        font={"color": TEXT, "size": 12},
    )
    fig.update_yaxes(title="전력 (kW)", range=[0, 230])
    fig.update_xaxes(
        tickformat="%H:%M\n%m/%d",
        range=[REFERENCE_TIME - pd.Timedelta(hours=hours), REFERENCE_TIME + pd.Timedelta(hours=1)],
    )
    fig.update_layout(
        clickmode="event+select",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.09, "x": 0, "font": {"size": 12}},
    )
    return _base_layout(fig, 470)


CAUSES = pd.DataFrame(
    {
        "공정조건": ["3시간 조업 모멘텀", "점심 반등 예상량", "기온 축열", "생산량 추세", "습도"],
        "기여도": [28.4, 22.1, 16.8, -12.5, -6.3],
    }
)


def cause_figure(height: int = 330, selected_time: str | None = None) -> go.Figure:
    frame = CAUSES.sort_values("기여도")
    top_three = set(CAUSES.reindex(CAUSES["기여도"].abs().sort_values(ascending=False).index).head(3)["공정조건"])
    colors = [
        (RED if value > 0 else BLUE) if condition in top_three else "#526879"
        for condition, value in zip(frame["공정조건"], frame["기여도"])
    ]
    fig = go.Figure(
        go.Bar(
            x=frame["기여도"], y=frame["공정조건"], orientation="h",
            marker_color=colors,
            text=[f"{value:+.1f}" for value in frame["기여도"]],
            textposition="inside",
            insidetextfont={"color": TEXT},
        )
    )
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="기여 전력 (kW)", range=[-40, 40])
    if selected_time:
        fig.add_annotation(
            x=1,
            y=1.12,
            xref="paper",
            yref="paper",
            text=f"선택 시점 {selected_time}",
            showarrow=False,
            font={"color": TEAL, "size": 13},
            xanchor="right",
        )
    return _base_layout(fig, height, {"l": 132, "r": 28, "t": 15, "b": 42})


def importance_figure() -> go.Figure:
    frame = pd.DataFrame(
        {
            "변수": CAUSES["공정조건"],
            "중요도": [0.36, 0.27, 0.18, 0.11, 0.08],
        }
    ).sort_values("중요도")
    fig = px.bar(frame, x="중요도", y="변수", orientation="h")
    fig.update_traces(marker_color=BLUE, text=frame["중요도"], textposition="outside")
    fig.update_xaxes(range=[0, 0.42], title="평균 |SHAP|")
    return _base_layout(fig, 230, {"l": 125, "r": 35, "t": 8, "b": 38})


def model_comparison_figure() -> go.Figure:
    frame = pd.DataFrame(
        {
            "모델": ["Top-3 GBDT", "CatBoost", "XGBoost", "Tri-GBDT", "LightGBM", "Ridge"],
            "RMSE": [9.34, 9.51, 9.56, 9.57, 10.15, 13.94],
        }
    ).sort_values("RMSE", ascending=False)
    colors = [TEAL if model == "Top-3 GBDT" else BLUE if value < 10 else "#9AA9BE" for model, value in zip(frame["모델"], frame["RMSE"])]
    fig = go.Figure(
        go.Bar(
            x=frame["RMSE"], y=frame["모델"], orientation="h",
            marker_color=colors, text=[f"{value:.2f}" for value in frame["RMSE"]],
            textposition="outside",
        )
    )
    fig.update_xaxes(title="Test RMSE (kW)", range=[0, 16])
    return _base_layout(fig, 280, {"l": 120, "r": 40, "t": 8, "b": 42})


def error_heatmap_figure() -> go.Figure:
    rng = np.random.default_rng(42)
    base = rng.integers(0, 7, size=(4, 24)).astype(float)
    base[2:, 12:18] += np.array([5, 7, 8, 11, 9, 6])
    fig = go.Figure(
        go.Heatmap(
            z=base,
            x=list(range(24)),
            y=["낮음", "중간", "높음", "피크 근접"],
            colorscale=[[0, "#0B1822"], [0.45, "#853943"], [1, "#F0525F"]],
            colorbar={"title": "오류 건수", "thickness": 12},
            hovertemplate="%{x}시 · %{y}<br>오류 %{z:.0f}건<extra></extra>",
        )
    )
    fig.update_xaxes(title=None, dtick=3)
    fig.update_yaxes(title="전력 구간")
    return _base_layout(fig, 250, {"l": 78, "r": 55, "t": 8, "b": 35})


ACTION_ROWS = [
    {"시각": "10/15 14:30", "단계": "위험", "전력": "176.8 kW", "담당자": "김현수", "조치": "관리자 호출", "상태": "처리 중"},
    {"시각": "10/15 11:15", "단계": "주의", "전력": "162.4 kW", "담당자": "이지민", "조치": "설비 상태 점검", "상태": "완료"},
    {"시각": "10/14 16:45", "단계": "위험", "전력": "181.2 kW", "담당자": "박서준", "조치": "가동 연기", "상태": "완료"},
    {"시각": "10/14 13:00", "단계": "주의", "전력": "159.8 kW", "담당자": "미지정", "조치": "-", "상태": "미확인"},
]
