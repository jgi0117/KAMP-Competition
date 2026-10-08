"""Read-only dashboard for the held-out 2021 Test period."""

from datetime import date

from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go

from data import CONTROL_SIGMA, PEAK_KW, load_evidence


comparison, test, cutoff, weights = load_evidence()
best = comparison.set_index("model").loc["ensemble"]
residual_sigma = float(test.attrs["residual_sigma"])
control_margin = CONTROL_SIGMA * residual_sigma
control_violations = int(test.control_violation.sum())
app = Dash(__name__)
server = app.server


def card(label, value, detail=""):
    return html.Div([html.Div(label, className="card-label"),
                     html.Div(value, className="card-value"),
                     html.Div(detail, className="card-detail")], className="card")


app.layout = html.Main([
    html.Header([
        html.Div("OKM · 전력 피크 예측", className="eyebrow"),
        html.H1("5개 모델 성능 비교 · 9개 공통 입력변수"),
        html.P("2021년 홀드아웃 Test 703시간 재생 · 실시간 설비 연결 없음"),
    ]),
    html.Section([
        card("최적 후보", "LSTM + TCN", "검증 세트에서 결정한 가중 앙상블"),
        card("Test RMSE", f"{best.rmse:.2f} kW", "5개 후보 중 최저"),
        card("3σ 관리한계", f"±{control_margin:.1f} kW",
             f"검증 잔차 σ={residual_sigma:.2f} · 이탈 {control_violations}건"),
        card("피크 재현율", f"{best.alert_recall:.1%}", f"177 kW 이상 · 놓친 피크 {int(best.fn)}건"),
        card("피크 F1", f"{best.alert_f1:.3f}", f"오경보 {int(best.fp)}건"),
    ], className="cards"),
    html.Section([
        html.Div([
            html.H2("실측과 예측"),
            html.P("음영은 예측값 ±3σ 관리구간입니다. 주황 마름모는 3σ 이탈, 빨간 점은 피크 경보, 노란 ×는 놓친 피크입니다."),
            dcc.DatePickerRange(id="dates", min_date_allowed=test.datetime.min().date(),
                                max_date_allowed=test.datetime.max().date(),
                                start_date=test.datetime.min().date(),
                                end_date=test.datetime.max().date(), display_format="YYYY-MM-DD"),
            dcc.Graph(id="timeline"),
        ], className="panel wide"),
        html.Div([
            html.H2("후보별 Test 오차"),
            dcc.Graph(id="ranking", figure={
                "data": [go.Bar(x=comparison.sort_values("rmse").model,
                                y=comparison.sort_values("rmse").rmse,
                                marker_color=["#60d9c5" if x == "ensemble" else "#67839d"
                                              for x in comparison.sort_values("rmse").model])],
                "layout": go.Layout(yaxis_title="RMSE (kW)", margin=dict(l=55, r=20, t=20, b=50),
                                    paper_bgcolor="#162333", plot_bgcolor="#162333", font_color="#e9f0f4"),
            }),
        ], className="panel"),
        html.Div([
            html.H2("판정 기준"),
            html.P(f"관리상한=예측+3σ, 관리하한=max(0, 예측−3σ)입니다. σ는 Test가 아닌 검증 잔차에서 산출한 {residual_sigma:.2f} kW를 고정해 사용합니다."),
            html.P(f"실측 {PEAK_KW:g} kW 이상의 피크 경보(확률 기준 {cutoff:g})는 3σ 이상 감지와 병행합니다."),
            html.P(f"앙상블 가중치: LSTM {weights['lstm']:.3f}, TCN {weights['tcn']:.3f}"),
            html.P("모든 값은 저장된 예측 파일에서 읽습니다. 모델 추론과 실시간 예측은 제공하지 않습니다."),
        ], className="panel"),
    ], className="grid"),
], className="container")


@app.callback(Output("timeline", "figure"), Input("dates", "start_date"), Input("dates", "end_date"))
def update_timeline(start_date, end_date):
    visible = test.loc[test.datetime.dt.date.between(
        date.fromisoformat(start_date), date.fromisoformat(end_date))]
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=visible.datetime, y=visible.control_lower, name="3σ 관리하한 (LCL)",
        mode="lines", line=dict(color="rgba(96,217,197,0.55)", width=1, dash="dot"),
        hovertemplate="LCL %{y:.1f} kW<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=visible.datetime, y=visible.control_upper, name="3σ 관리상한 (UCL)",
        mode="lines", line=dict(color="rgba(96,217,197,0.8)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(96,217,197,0.12)",
        hovertemplate="UCL %{y:.1f} kW<extra></extra>",
    ))
    figure.add_trace(go.Scatter(x=visible.datetime, y=visible.actual, name="실측", mode="lines",
                                line=dict(color="#e9f0f4", width=2)))
    figure.add_trace(go.Scatter(x=visible.datetime, y=visible.predicted, name="앙상블 예측", mode="lines",
                                line=dict(color="#60d9c5", width=2)))
    violations = visible.loc[visible.control_violation]
    figure.add_trace(go.Scatter(
        x=violations.datetime, y=violations.actual, name="3σ 이탈", mode="markers",
        marker=dict(color="#ff9f43", size=9, symbol="diamond", line=dict(color="#fff", width=1)),
    ))
    alerts = visible.loc[visible.alert]
    figure.add_trace(go.Scatter(x=alerts.datetime, y=alerts.actual, name="경보", mode="markers",
                                marker=dict(color="#ff6d6d", size=7)))
    misses = visible.loc[visible.outcome.eq("FN")]
    figure.add_trace(go.Scatter(x=misses.datetime, y=misses.actual, name="놓친 피크", mode="markers",
                                marker=dict(color="#ffd166", size=13, symbol="x")))
    figure.add_hline(y=PEAK_KW, line_dash="dash", line_color="#ff6d6d")
    figure.update_layout(paper_bgcolor="#162333", plot_bgcolor="#162333", font_color="#e9f0f4",
                         margin=dict(l=55, r=20, t=25, b=40), yaxis_title="전력 (kW)",
                         legend=dict(orientation="h", y=1.18), hovermode="x unified")
    return figure


if __name__ == "__main__":
    app.run(debug=False)
