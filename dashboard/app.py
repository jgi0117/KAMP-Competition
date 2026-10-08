"""AI 전력 피크 조기경보 시스템 Dash 레이아웃 프로토타입."""

from __future__ import annotations

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from layouts import action_table, page_for, sidebar, topbar
from data import ACTION_ROWS, SUMMARY, cause_figure, energy_figure, replay_timestamp


app = Dash(__name__, suppress_callback_exceptions=True, title="AI 전력 피크 조기경보 시스템")
server = app.server

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Interval(id="simulation-interval", interval=5_000, n_intervals=0, disabled=False),
        dcc.Store(id="viewport-width-store"),
        dcc.Store(id="home-range-store", data=6),
        dcc.Store(id="selected-time-store", storage_type="session"),
        dcc.Store(id="action-records-store", data=ACTION_ROWS, storage_type="session"),
        dcc.Store(id="alarm-muted-store", data=True, storage_type="session"),
        html.Div(id="alarm-audio-sink", hidden=True),
        html.Aside(id="sidebar", children=sidebar("/"), className="sidebar"),
        html.Div([topbar(), html.Div(id="page-container", children=page_for("/"))], id="main-shell", className="main-shell"),
    ],
    id="app-shell",
    className="app-shell",
)


@app.callback(
    Output("sidebar", "children"),
    Output("page-container", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str | None):
    pathname = pathname or "/"
    return sidebar(pathname), page_for(pathname)


@app.callback(
    Output("sidebar", "className"),
    Output("main-shell", "className"),
    Output("sidebar-toggle", "children"),
    Output("sidebar-toggle", "title"),
    Input("sidebar-toggle", "n_clicks"),
    Input("viewport-width-store", "data"),
    State("sidebar", "className"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks: int, viewport_width: int | None, current_class: str):
    if ctx.triggered_id == "viewport-width-store" or not n_clicks:
        if (viewport_width or 1920) <= 1440:
            return "sidebar collapsed", "main-shell sidebar-collapsed", "›", "사이드바 펼치기"
        return "sidebar expanded", "main-shell sidebar-expanded", "‹", "사이드바 접기"
    classes = current_class or "sidebar"
    is_expanded = "expanded" in classes
    is_collapsed = "collapsed" in classes or (not is_expanded and (viewport_width or 1920) <= 1440)
    if is_collapsed:
        return "sidebar expanded", "main-shell sidebar-expanded", "‹", "사이드바 접기"
    return "sidebar collapsed", "main-shell sidebar-collapsed", "›", "사이드바 펼치기"


@app.callback(Output("simulation-time", "children"), Input("simulation-interval", "n_intervals"))
def update_simulation_time(n_intervals: int):
    timestamp = replay_timestamp(n_intervals)
    return f"Test 재생 {timestamp:%m/%d %H:%M}"


app.clientside_callback(
    "function(pathname) { return window.innerWidth || 1920; }",
    Output("viewport-width-store", "data"),
    Input("url", "pathname"),
)


@app.callback(
    Output("simulation-interval", "disabled"),
    Output("play-status-label", "children"),
    Output("play-control-icon", "children"),
    Output("pause-button", "className"),
    Output("pause-button", "title"),
    Input("pause-button", "n_clicks"),
    State("simulation-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_simulation(n_clicks: int, is_paused: bool):
    will_pause = not bool(is_paused)
    if will_pause:
        return True, "일시정지", "▶", "control-button paused-control", "시뮬레이션 재생"
    return False, "실행 중", "Ⅱ", "control-button active-control", "시뮬레이션 일시정지"


@app.callback(
    Output("alarm-muted-store", "data"),
    Output("alarm-status-label", "children"),
    Output("mute-button", "className"),
    Output("mute-button", "title"),
    Input("mute-button", "n_clicks"),
    State("alarm-muted-store", "data"),
    prevent_initial_call=True,
)
def toggle_alarm(n_clicks: int, is_muted: bool):
    next_muted = not bool(is_muted)
    if next_muted:
        return True, "음소거", "control-button muted-control", "경보음 켜기"
    return False, "경보음 켜짐", "control-button alarm-control", "경보음 끄기"


app.clientside_callback(
    """
    function(isMuted) {
        if (isMuted !== false) return '';
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            const audio = new AudioContext();
            [0, 0.22, 0.44].forEach(function(delay) {
                const oscillator = audio.createOscillator();
                const gain = audio.createGain();
                oscillator.type = 'sine';
                oscillator.frequency.value = 880;
                gain.gain.setValueAtTime(0.0001, audio.currentTime + delay);
                gain.gain.exponentialRampToValueAtTime(0.14, audio.currentTime + delay + 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + delay + 0.13);
                oscillator.connect(gain).connect(audio.destination);
                oscillator.start(audio.currentTime + delay);
                oscillator.stop(audio.currentTime + delay + 0.14);
            });
        } catch (error) {}
        return 'played';
    }
    """,
    Output("alarm-audio-sink", "children"),
    Input("alarm-muted-store", "data"),
)


@app.callback(
    Output("energy-chart", "figure"),
    Output("home-range-store", "data"),
    Output("range-6h", "className"),
    Output("range-12h", "className"),
    Output("range-24h", "className"),
    Input("range-6h", "n_clicks"),
    Input("range-12h", "n_clicks"),
    Input("range-24h", "n_clicks"),
    Input("simulation-interval", "n_intervals"),
    State("home-range-store", "data"),
)
def update_energy_range(n6: int, n12: int, n24: int, n_intervals: int, current_hours: int):
    selected = {"range-6h": 6, "range-12h": 12, "range-24h": 24}.get(ctx.triggered_id, current_hours or 6)
    classes = ["filter-chip active" if selected == value else "filter-chip" for value in (6, 12, 24)]
    return energy_figure(selected, n_intervals), selected, *classes


@app.callback(
    Output("home-cause-chart", "figure"),
    Output("home-cause-title", "children"),
    Output("home-analysis-time", "children"),
    Output("home-cause-insight", "children"),
    Output("selected-time-store", "data"),
    Input("energy-chart", "clickData"),
    prevent_initial_call=True,
)
def update_home_cause(click_data: dict | None):
    if not click_data or not click_data.get("points"):
        return no_update, no_update, no_update, no_update, no_update
    raw_time = str(click_data["points"][0].get("x", ""))
    try:
        timestamp = raw_time.replace("T", " ")[:16]
        label = timestamp[5:16]
    except (TypeError, IndexError):
        label = raw_time
    return (
        cause_figure(470, label),
        f"{label} 모델 변수 중요도 참고",
        f"{label} 선택 · Test 전역 기준",
        "시점별 SHAP은 저장되지 않아 XGBoost·LightGBM의 Test permutation importance를 표시합니다.",
        label,
    )


@app.callback(Output("cause-detail-chart", "figure"), Input("selected-time-store", "data"))
def update_cause_detail(selected_time: str | None):
    return cause_figure(360, selected_time) if selected_time else cause_figure(360)


@app.callback(
    Output("action-drawer", "className"),
    Input("open-action-drawer", "n_clicks"),
    Input("close-action-drawer", "n_clicks"),
    Input("drawer-backdrop", "n_clicks"),
    Input("cancel-action-drawer", "n_clicks"),
    Input("save-action-drawer", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_action_drawer(open_clicks: int, close_clicks: int, backdrop_clicks: int, cancel_clicks: int, save_clicks: int):
    return "action-drawer open" if ctx.triggered_id == "open-action-drawer" else "action-drawer"


@app.callback(
    Output("save-action-drawer", "disabled"),
    Output("drawer-validation", "children"),
    Output("drawer-validation", "className"),
    Input("drawer-action-types", "value"),
)
def validate_drawer(actions: list[str] | None):
    if actions:
        return False, f"{len(actions)}개 조치가 선택되었습니다.", "drawer-validation ready"
    return True, "조치를 하나 이상 선택해야 저장할 수 있습니다.", "drawer-validation"


@app.callback(
    Output("action-records-store", "data"),
    Input("save-action-drawer", "n_clicks"),
    State("drawer-action-types", "value"),
    State("drawer-assignee", "value"),
    State("action-records-store", "data"),
    prevent_initial_call=True,
)
def save_action_record(n_clicks: int, actions: list[str] | None, assignee: str | None, records: list[dict] | None):
    if not actions:
        return no_update
    new_record = {
        "시각": f"{replay_timestamp(0):%m/%d %H:%M}",
        "단계": "Test 검토",
        "전력": f"{SUMMARY['max_predicted']:.1f} kW",
        "담당자": assignee or "미지정",
        "조치": ", ".join(actions),
        "상태": "처리 중",
    }
    return [new_record, *(records or ACTION_ROWS)]


@app.callback(Output("action-table-container", "children"), Input("action-records-store", "data"))
def refresh_action_table(records: list[dict] | None):
    return action_table(records)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
