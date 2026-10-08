"""목업을 Dash 컴포넌트로 옮긴 화면 레이아웃."""

from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

from data import (
    ACTION_ROWS,
    FEATURE_ROWS,
    SUMMARY,
    cause_figure,
    energy_figure,
    error_heatmap_figure,
    feature_rows,
    importance_figure,
    model_comparison_figure,
)


NAV_ITEMS = [
    ("/", "activity", "Test 피크 관제"),
    ("/causes", "search", "기여·점검"),
    ("/model", "chart", "모델 성능"),
    ("/actions", "clipboard", "조치 관리"),
]


ICON_PATHS = {
    "activity": "M3 12h4l2-7 4 14 2-7h6",
    "search": "M11 4a7 7 0 1 0 0 14a7 7 0 0 0 0-14m5 12l5 5",
    "chart": "M4 19V9m6 10V5m6 14v-7m5 7H2",
    "clipboard": "M9 5h6m-7-2h8v4H8zm-2 2H5v16h14V5h-2M8 12h8M8 16h8",
    "settings": "M12 8a4 4 0 1 0 0 8a4 4 0 0 0 0-8m0-5v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4m0-12.8L17 7M7 17l-1.4 1.4",
    "bolt": "M13 2 5 14h7l-1 8 8-12h-7z",
    "clock": "M12 3a9 9 0 1 0 0 18a9 9 0 0 0 0-18m0 5v5l3 2",
    "threshold": "M4 18h16M7 15l3-3 3 2 4-6",
    "alert": "M12 3 2.5 20h19zM12 9v5m0 3h.01",
    "target": "M12 4a8 8 0 1 0 8 8M12 8a4 4 0 1 0 4 4M12 12l8-8",
    "model": "M4 6h16v12H4zM8 10h2v4H8zm6-2h2v6h-2z",
    "check": "M5 12l4 4L19 6",
}


def ui_icon(name: str, class_name: str = ""):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        'stroke="black" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{ICON_PATHS[name]}"/></svg>'
    )
    mask = f'url("data:image/svg+xml,{quote(svg)}")'
    return html.Span(
        className=f"ui-icon {class_name}".strip(),
        style={"WebkitMaskImage": mask, "maskImage": mask},
        **{"aria-hidden": "true"},
    )


def sidebar(pathname: str):
    links = []
    for href, icon_name, label in NAV_ITEMS:
        active = pathname == href or (href != "/" and pathname.startswith(href))
        links.append(
            dcc.Link(
                [html.Span(ui_icon(icon_name), className="nav-icon"), html.Span(label, className="nav-label")],
                href=href,
                title=label,
                className=f"nav-link{' active' if active else ''}",
            )
        )
    return [
        html.Button("‹", id="sidebar-toggle", className="sidebar-toggle", title="사이드바 접기"),
        html.Div(links, className="nav-stack"),
        html.Div(
            [
                html.Div([html.Span(ui_icon("settings"), className="nav-icon"), html.Span("설정", className="nav-label")], className="nav-link nav-static"),
                html.Div("Prototype  v1.0.0", className="version-label"),
            ],
            className="sidebar-footer",
        ),
    ]


def topbar():
    return html.Header(
        [
            html.Div(
                [
                    html.Span("POWER CONTROL", className="system-kicker"),
                    html.H1("AI 전력 피크 조기경보 시스템", className="app-title"),
                ],
                className="system-identity",
            ),
            html.Div(
                [
                    html.Span("TEST REPLAY", className="badge badge-teal"),
                    html.Span([html.I(className="status-led normal"), "검증 데이터 로드 완료"], className="top-chip status-chip"),
                    html.Span("1시간 데이터 간격", className="top-chip"),
                    html.Span("Test 재생 준비", id="simulation-time", className="top-chip time-chip"),
                    html.Button(
                        [html.Span("Ⅱ", id="play-control-icon", className="control-icon"), html.Span("실행 중", id="play-status-label")],
                        id="pause-button",
                        className="control-button active-control",
                        title="시뮬레이션 일시정지",
                    ),
                    html.Button(
                        [html.Span("⌁", className="control-icon"), html.Span("음소거", id="alarm-status-label")],
                        id="mute-button",
                        className="control-button muted-control",
                        title="경보음 켜기",
                    ),
                ],
                className="top-actions",
            ),
        ],
        className="topbar",
    )


def graph_card(title: str, figure, class_name: str = "", graph_id: str | None = None):
    graph_props = {"figure": figure, "config": {"displayModeBar": False}, "className": "plot"}
    if graph_id:
        graph_props["id"] = graph_id
    return html.Section(
        [
            html.H3(title, className="section-title"),
            dcc.Loading(
                dcc.Graph(**graph_props),
                custom_spinner=html.Div(className="chart-skeleton"),
                delay_show=120,
            ),
        ],
        className=f"panel {class_name}".strip(),
    )


def kpi_card(label: str, value: str, icon_name: str, tone: str = "blue", note: str | None = None, class_name: str = ""):
    return html.Div(
        [
            html.Div(ui_icon(icon_name), className=f"kpi-icon {tone}"),
            html.Div(
                [
                    html.Div(label, className="kpi-label"),
                    html.Div(value, className=f"kpi-value {tone}"),
                    html.Div(note, className="kpi-note") if note else None,
                ]
            ),
        ],
        className=f"kpi-card {class_name}".strip(),
    )


def industrial_metric_rail(items: list[tuple[str, str, str, str, str]], class_name: str = ""):
    return html.Section(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(label, className="metric-label"),
                        html.Div([html.Strong(value), html.Span(unit)], className="metric-value"),
                        html.Div(note, className="metric-note"),
                    ],
                    className=f"metric-cell {tone}",
                )
                for label, value, unit, note, tone in items
            ],
            className="metric-rail-grid",
        ),
        className=f"metric-rail {class_name}".strip(),
    )


def control_page_header(kicker: str, title: str, statuses: list[tuple[str, str]]):
    return html.Header(
        [
            html.Div([html.Span(kicker, className="control-page-kicker"), html.H2(title, className="page-title")]),
            html.Div(
                [html.Span([html.I(className=f"status-led {tone}"), text], className="control-status") for text, tone in statuses],
                className="control-status-group",
            ),
        ],
        className="control-page-header",
    )


def hero_kpi_card():
    return industrial_metric_rail([
        ("Test 최대 예측", f"{SUMMARY['max_predicted']:.1f}", "kW", "앙상블 저장 예측", "forecast"),
        ("피크 재현율", f"{SUMMARY['alert_recall']:.1%}", "", f"실제 피크 {SUMMARY['peak_count']}시간", "danger"),
        ("Test 최대 실측", f"{SUMMARY['max_actual']:.1f}", "kW", "703시간 홀드아웃", "actual"),
        ("Test RMSE", f"{SUMMARY['rmse']:.2f}", "kW", "LSTM + TCN 앙상블", "forecast"),
        ("피크 기준", f"{SUMMARY['peak_kw']:.0f}", "kW", "Test 이전 자료로 고정", "threshold"),
    ], "kpi-updated")


def status_pill(text: str, tone: str):
    return html.Span(text, className=f"status-pill {tone}")


def feature_table_rows(rows):
    return [
        html.Div(
            [
                html.Span(rank), html.Span(condition),
                html.Strong(current, className="danger-text"),
                html.Span(reference),
                html.Span(("↑  " if "증가" in direction else "↓  ") + direction, className="danger-text"),
                status_pill(status, tone),
            ],
            className="data-row",
        )
        for rank, condition, current, reference, direction, status, tone in rows
    ]


def simulation_note(text: str):
    return html.Div([html.Span("i", className="info-dot"), html.Span(text)], className="simulation-note")


def home_page():
    return html.Main(
        [
            dcc.Store(id={"type": "page-replay-time", "page": "home"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.Span("TEST RESULT", className="alarm-code"), html.Strong("피크 경보 검증 결과")], className="alarm-title"),
                            html.Div(
                                [
                                    html.Span([html.Small("Test 기간"), html.B(f"{SUMMARY['test_start']:%m/%d}–{SUMMARY['test_end']:%m/%d}")]),
                                    html.Span([html.Small("실제 피크"), html.B(f"{SUMMARY['peak_count']}시간")]),
                                    html.Span([html.Small("미탐지"), html.B(f"{SUMMARY['fn']}건")]),
                                    html.Span([html.Small("경보 재현율"), html.B(f"{SUMMARY['alert_recall']:.1%}")]),
                                ],
                                className="alarm-metrics",
                            ),
                        ],
                        className="alert-copy",
                    ),
                    html.Button("오류 사례 검토  ›", id="open-action-drawer", className="danger-button"),
                ],
                className="danger-banner danger-flash",
            ),
            hero_kpi_card(),
            html.Div(
                [
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([html.H3("Test 전력 재생", className="section-title"), html.Span("REPLAY · 1시간 간격", className="live-label")], className="panel-title-row"),
                                            html.P("실측 / 앙상블 예측 / 검증 잔차 3σ 관리구간", className="section-subtitle"),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Button("6시간", id="range-6h", className="filter-chip active"),
                                            html.Button("12시간", id="range-12h", className="filter-chip"),
                                            html.Button("24시간", id="range-24h", className="filter-chip"),
                                        ],
                                        className="range-control",
                                    ),
                                ],
                                className="panel-header",
                            ),
                            dcc.Loading(
                                dcc.Graph(id="energy-chart", figure=energy_figure(6), config={"displayModeBar": False}, className="plot home-energy-plot"),
                                custom_spinner=html.Div(className="chart-skeleton"),
                                delay_show=120,
                            ),
                        ],
                        className="panel main-chart",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H3("시점별 트리 모델 기여도", id="home-cause-title", className="section-title"),
                                    html.Span("08/17 00:00 입력", id="home-analysis-time", className="analysis-time"),
                                ],
                                className="cause-panel-header",
                            ),
                            dcc.Link(
                                dcc.Graph(id="home-cause-chart", figure=cause_figure(470), config={"displayModeBar": False}, className="plot"),
                                href="/causes",
                                className="chart-link",
                                title="기여·점검 화면으로 이동",
                            ),
                            html.Div(
                                [
                                    html.Span("모델 근거", className="ai-label"),
                                    html.Span("저장 트리 모델의 168시간 입력 TreeSHAP을 변수별로 합산했습니다.", id="home-cause-insight"),
                                ],
                                className="home-cause-insight",
                            ),
                        ],
                        className="panel side-chart",
                    ),
                ],
                className="home-chart-grid",
            ),
            simulation_note("TEST REPLAY · 저장된 2021년 홀드아웃 703시간을 재생합니다. 실시간 설비 추론이 아닙니다."),
            action_drawer(),
        ],
        className="page-content home-control-page",
    )


def action_drawer():
    return html.Div(
        [
            html.Button("", id="drawer-backdrop", className="drawer-backdrop", **{"aria-label": "조치 패널 닫기"}),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Div([html.Span("TEST REVIEW", className="drawer-eyebrow"), html.H3("오류 사례 검토", className="drawer-title")]),
                            html.Button("×", id="close-action-drawer", className="drawer-close", title="닫기"),
                        ],
                        className="drawer-header",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong(f"FN {SUMMARY['fn']} · FP {SUMMARY['fp']}"), html.Span("Test 경보 오류")], className="drawer-risk-summary"),
                            html.H4("검토 항목"),
                            html.P("저장된 Test 결과의 오류 사례를 검토하는 세션 기록입니다.", className="drawer-help"),
                            html.Div(
                                [
                                    html.Div([html.Span("검토", className="recommend-badge"), html.Span("미탐지 원인 확인")], className="recommend-item"),
                                    html.Div([html.Span("검토", className="recommend-badge"), html.Span("오경보 시간대 확인")], className="recommend-item"),
                                ],
                                className="recommend-list",
                            ),
                            html.Label("실행할 조치", className="field-label"),
                            dcc.Checklist(
                                id="drawer-action-types",
                                options=[
                                    {"label": "미탐지 원인 확인", "value": "미탐지 원인 확인"},
                                    {"label": "오경보 시간대 확인", "value": "오경보 시간대 확인"},
                                    {"label": "입력 데이터 점검", "value": "입력 데이터 점검"},
                                    {"label": "임계값 재검토", "value": "임계값 재검토"},
                                ],
                                value=[],
                                className="checklist action-choice-list",
                            ),
                            html.Label("담당자", className="field-label"),
                            dcc.Dropdown(id="drawer-assignee", options=["김현수", "이지민", "박서준"], placeholder="담당자를 선택하세요", clearable=False),
                            html.Label("메모", className="field-label"),
                            dcc.Textarea(id="drawer-memo", placeholder="확인 내용이나 전달 사항을 입력하세요", className="memo-field drawer-memo"),
                            html.Div("조치를 하나 이상 선택해야 저장할 수 있습니다.", id="drawer-validation", className="drawer-validation"),
                        ],
                        className="drawer-body",
                    ),
                    html.Div(
                        [
                            html.Button("취소", id="cancel-action-drawer", className="secondary-button"),
                            html.Button("조치 저장", id="save-action-drawer", className="primary-button", disabled=True),
                        ],
                        className="drawer-footer",
                    ),
                ],
                className="action-drawer-panel",
            ),
        ],
        id="action-drawer",
        className="action-drawer",
    )


def causes_page():
    rows = feature_rows()
    return html.Main(
        [
            dcc.Store(id={"type": "page-replay-time", "page": "causes"}),
            control_page_header(
                "MODEL EXPLANATION / TEST SET",
                "변수 중요도 및 오류 조건 점검",
                [("LOCAL TREESHAP", "danger"), ("재생 시각 기준", "normal")],
            ),
            html.Div(
                [
                    html.Span("ANALYSIS", className="event-code"),
                    html.Strong("시점별 트리 모델 기여도"),
                    html.Span("XGBoost + LightGBM"),
                    html.Span("9개 공통 입력변수", className="danger-text"),
                    html.Span("재생 데이터에 따라 갱신", className="event-note"),
                ],
                className="event-strip",
            ),
            html.Div(
                [
                    graph_card("모델별 시점 기여도 · 168시간 입력", cause_figure(360), "cause-main", "cause-detail-chart"),
                    html.Div(
                        [
                            graph_card("두 트리 모델 평균 시점 기여도", importance_figure(), graph_id="cause-average-chart"),
                            html.Section(
                                [html.Div([html.H3("분석 해석", className="section-title"), html.Span("MODEL EXPLANATION", className="panel-code")], className="industrial-panel-header"), html.P("저장된 XGBoost·LightGBM의 TreeSHAP을 168시간 lag 전체에서 입력변수별로 합산했습니다. 양수는 해당 시점 예측을 높이고 음수는 낮춥니다.", className="insight-copy")],
                                className="panel insight-panel",
                            ),
                        ],
                        className="cause-side",
                    ),
                ],
                className="cause-grid",
            ),
            html.Section(
                [
                    html.Div([html.Div([html.H3("현재 시점 기여도 상위 변수", className="section-title"), html.Span("TOP 3 · LOCAL TREESHAP", className="panel-code")], className="panel-title-row"), html.Div([html.Button("계산 완료", className="secondary-button", disabled=True), html.Button("모델 근거", className="danger-button small", disabled=True)])], className="panel-header"),
                    html.Div(
                        [
                            html.Div([html.Strong("순위"), html.Strong("입력변수"), html.Strong("평균 기여"), html.Strong("계산 방식"), html.Strong("영향 방향"), html.Strong("상태")], className="data-row header-row"),
                            html.Div(feature_table_rows(rows), id="feature-rows-container", className="feature-rows"),
                        ],
                        className="data-table condition-table",
                    ),
                ],
                className="panel table-panel",
            ),
            simulation_note("시점 기여도는 저장된 두 트리 모델의 로컬 TreeSHAP입니다. 최종 LSTM+TCN 앙상블의 설명값이나 인과효과를 뜻하지 않습니다."),
        ],
        className="page-content industrial-page causes-control-page",
    )


def model_page():
    return html.Main(
        [
            control_page_header(
                "MODEL VALIDATION / TEST SET",
                "예측 모델 성능 및 오류 분석",
                [("회귀 · 실제 Test", "normal"), ("피크 경보 · 검증 기준", "caution")],
            ),
            html.Div([html.Span("EVIDENCE", className="event-code info"), html.Strong("동일 Test 703시간 비교"), html.Span("seed 42 · 공통 입력 9개"), html.Span("저장된 검증·예측 산출물", className="event-note")], className="event-strip info"),
            industrial_metric_rail(
                [
                    ("최종 모델", SUMMARY["model"], "", "Test RMSE 기준 1위", "forecast"),
                    ("RMSE", f"{SUMMARY['rmse']:.2f}", "kW", "낮을수록 우수", "actual"),
                    ("MAE", f"{SUMMARY['mae']:.2f}", "kW", "평균 절대 오차", "actual"),
                    ("결정계수 R²", f"{SUMMARY['r2']:.4f}", "", "설명력", "forecast"),
                ],
                "four-metrics",
            ),
            html.Div(
                [
                    graph_card("후보 모델 Test RMSE 비교 · 실제 테스트", model_comparison_figure(), "model-chart"),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.Div([html.H3("피크 경보 성능", className="section-title"), html.Span("VALIDATED", className="panel-code caution")], className="panel-header"),
                                    html.Div([kpi_mini("F1", f"{SUMMARY['alert_f1']:.3f}"), kpi_mini("Precision", f"{SUMMARY['alert_precision']:.1%}"), kpi_mini("Recall", f"{SUMMARY['alert_recall']:.1%}"), kpi_mini("FN", str(SUMMARY["fn"])), kpi_mini("FP", str(SUMMARY["fp"]))], className="mini-grid"),
                                ],
                                className="panel",
                            ),
                            html.Section(
                                [html.Div([html.H3("최종 모델 선정 근거", className="section-title"), html.Span("DECISION BASIS", className="panel-code")], className="industrial-panel-header"), html.Div([html.Span("동일 Test 구간 비교"), html.Span("회귀 RMSE 1위"), html.Span("피크 재현율 97.96%")], className="reason-grid")],
                                className="panel reason-panel",
                            ),
                        ],
                        className="model-side",
                    ),
                ],
                className="model-grid",
            ),
            html.Section(
                [html.Div([html.H3("FN·FP 집중 구간", className="section-title"), html.Div([html.Button("시간대", className="filter-chip active"), html.Button("전력구간", className="filter-chip"), html.Button("공정조건", className="filter-chip")])], className="panel-header"), dcc.Graph(figure=error_heatmap_figure(), config={"displayModeBar": False}, className="plot")],
                className="panel heatmap-panel",
            ),
            simulation_note(f"피크 기준 {SUMMARY['peak_kw']:.0f} kW와 경보 확률 기준 {SUMMARY['cutoff']:.2f}는 Test 이전 검증 자료에서 고정했습니다."),
        ],
        className="page-content industrial-page model-control-page",
    )


def kpi_mini(label: str, value: str):
    return html.Div([html.Span(label), html.Strong(value)], className="mini-kpi")


def actions_page():
    return html.Main(
        [
            control_page_header(
                "TEST ERROR REVIEW / SESSION",
                "피크 경보 오류 사례 검토",
                [(f"미탐지 {SUMMARY['fn']}건", "danger"), (f"오경보 {SUMMARY['fp']}건", "caution"), ("세션 메모", "normal")],
            ),
            html.Div([html.Span("TEST EVIDENCE", className="event-code info"), html.Strong("저장된 경보 오류 사례"), html.Span("현재 세션 검토 메모"), html.Span("운영 조치 DB 아님", className="event-note")], className="event-strip info"),
            industrial_metric_rail(
                [
                    ("실제 피크", str(SUMMARY["peak_count"]), "시간", "177 kW 이상", "danger"),
                    ("정탐(TP)", str(SUMMARY["peak_count"] - SUMMARY["fn"]), "건", "검증 경보 적중", "forecast"),
                    ("오경보(FP)", str(SUMMARY["fp"]), "건", "비피크 경보", "threshold"),
                    ("미탐지(FN)", str(SUMMARY["fn"]), "건", "피크 경보 누락", "actual"),
                ],
                "four-metrics",
            ),
            html.Div(
                [
                    dropdown("Test 기간", ["전체 703시간"], "전체 703시간"),
                    dropdown("오류 유형", ["전체", "미탐지", "오경보"], "전체"),
                    dropdown("검토 상태", ["전체", "검토 대상", "처리 중"], "전체"),
                    dropdown("담당자", ["전체", "미지정"], "전체"),
                    html.Button("CSV 원본 · 저장소 포함", className="primary-button export-button", disabled=True),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Section(
                        [html.H3("오류 사례·검토 목록", className="section-title"), html.Div(action_table(), id="action-table-container")],
                        className="panel action-list",
                    ),
                    action_detail(),
                ],
                className="action-grid",
            ),
            simulation_note("오류 사례는 저장된 Test 예측에서 읽습니다. 새 검토 기록만 현재 브라우저 세션에 보관되며 운영 DB에는 쓰지 않습니다."),
        ],
        className="page-content industrial-page actions-control-page",
    )


def dropdown(label: str, options: list[str], value: str):
    return html.Div(
        [html.Label(label), dcc.Dropdown(options=[{"label": option, "value": option} for option in options], value=value, clearable=False, searchable=False)],
        className="filter-control",
    )


def action_table(records: list[dict] | None = None):
    header = html.Div([html.Strong(label) for label in ["Test 시각", "오류 유형", "실측 전력", "담당자", "검토 항목", "검토 상태"]], className="action-row header-row")
    rows = []
    for idx, row in enumerate(records or ACTION_ROWS):
        risk_tone = "danger" if row["단계"] == "미탐지" else "caution"
        state_tone = "success" if row["상태"] == "완료" else "info" if row["상태"] == "처리 중" else "danger"
        rows.append(
            html.Div(
                [html.Span(row["시각"]), status_pill(row["단계"], risk_tone), html.Span(row["전력"]), html.Span(row["담당자"]), html.Span(row["조치"]), status_pill(row["상태"], state_tone)],
                className=f"action-row{' selected' if idx == 0 else ''}",
            )
        )
    return html.Div([header, *rows], className="data-table")


def action_detail():
    first = ACTION_ROWS[0]
    return html.Aside(
        [
            html.H3("검토 상세", className="section-title"),
            html.Div([detail_item("Test 시각", first["시각"]), detail_item("오류 유형", first["단계"]), detail_item("실측 전력", first["전력"])], className="detail-summary"),
            html.H4("검토 체크리스트"),
            dcc.Checklist(
                options=[
                    {"label": "해당 시각 입력값 확인", "value": "inputs"},
                    {"label": "경보 확률·기준값 확인", "value": "cutoff"},
                    {"label": "직전 생산량·기온 확인", "value": "context"},
                ],
                value=["inputs", "cutoff"],
                className="checklist",
            ),
            html.Label("검토 유형", className="field-label"),
            dcc.Dropdown(options=["사후 모델 검토", "임계값 검토", "입력 데이터 검토"], value="사후 모델 검토", clearable=False),
            html.Label("메모", className="field-label"),
            dcc.Textarea(value="Test 오류 사례 검토 메모", className="memo-field"),
            html.Div([html.Button("세션 저장", className="secondary-button"), html.Button("검토 완료", className="primary-button")], className="detail-actions"),
        ],
        className="panel action-detail sticky-detail",
    )


def detail_item(label: str, value: str):
    return html.Div([html.Span(label), html.Strong(value)])


def page_for(pathname: str):
    if pathname.startswith("/causes"):
        return causes_page()
    if pathname.startswith("/model"):
        return model_page()
    if pathname.startswith("/actions"):
        return actions_page()
    return home_page()
