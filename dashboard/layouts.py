"""목업을 Dash 컴포넌트로 옮긴 화면 레이아웃."""

from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

from mock_data import (
    ACTION_ROWS,
    cause_figure,
    energy_figure,
    error_heatmap_figure,
    importance_figure,
    model_comparison_figure,
)


NAV_ITEMS = [
    ("/", "activity", "실시간 피크 관제"),
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
                    html.Span("SIMULATION", className="badge badge-teal"),
                    html.Span([html.I(className="status-led normal"), "데이터 재생 정상"], className="top-chip status-chip"),
                    html.Span("15분 센서 주기", className="top-chip"),
                    html.Span("최종 수신 14:30", id="simulation-time", className="top-chip time-chip"),
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
        ("1시간 최대 예상", "176.8", "kW", "기준 초과 +16.8", "forecast"),
        ("피크 위험 확률", "82", "%", "위험", "danger"),
        ("현재 실제 전력", "148.2", "kW", "수신 14:30", "actual"),
        ("1시간 평균 예상", "158.4", "kW", "예측 구간 평균", "forecast"),
        ("위험 기준", "160.0", "kW", "직전 7일 상위 5%", "threshold"),
    ], "kpi-updated")


def status_pill(text: str, tone: str):
    return html.Span(text, className=f"status-pill {tone}")


def simulation_note(text: str):
    return html.Div([html.Span("i", className="info-dot"), html.Span(text)], className="simulation-note")


def home_page():
    return html.Main(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div([html.Span("CRITICAL", className="alarm-code"), html.Strong("전력 피크 발생 예상")], className="alarm-title"),
                            html.Div(
                                [
                                    html.Span([html.Small("예상 시각"), html.B("15:15")]),
                                    html.Span([html.Small("최대 예상"), html.B("176.8 kW")]),
                                    html.Span([html.Small("기준 초과"), html.B("+16.8 kW")]),
                                    html.Span([html.Small("위험 확률"), html.B("82%")]),
                                ],
                                className="alarm-metrics",
                            ),
                        ],
                        className="alert-copy",
                    ),
                    html.Button("조치 시작  ›", id="open-action-drawer", className="danger-button"),
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
                                            html.Div([html.H3("전력 수요 추이", className="section-title"), html.Span("LIVE · 15분 간격", className="live-label")], className="panel-title-row"),
                                            html.P("실제 전력 / 1시간 예측", className="section-subtitle"),
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
                                    html.H3("피크 예측 기여 요인", id="home-cause-title", className="section-title"),
                                    html.Span("15:15 예측 기준", id="home-analysis-time", className="analysis-time"),
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
                                    html.Span("AI 설명", className="ai-label"),
                                    html.Span("최근 3시간 조업 모멘텀이 예측값 상승에 가장 크게 기여했습니다.", id="home-cause-insight"),
                                ],
                                className="home-cause-insight",
                            ),
                        ],
                        className="panel side-chart",
                    ),
                ],
                className="home-chart-grid",
            ),
            simulation_note("SIMULATION · 차트 시점을 선택하면 같은 시점의 피크 예측 기여 요인을 확인할 수 있습니다."),
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
                            html.Div([html.Span("위험 대응", className="drawer-eyebrow"), html.H3("조치 등록", className="drawer-title")]),
                            html.Button("×", id="close-action-drawer", className="drawer-close", title="닫기"),
                        ],
                        className="drawer-header",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong("176.8 kW"), html.Span("1시간 최대 예상")], className="drawer-risk-summary"),
                            html.H4("추천 조치"),
                            html.P("AI 추천은 참고 정보이며, 실제 조치는 사용자가 선택합니다.", className="drawer-help"),
                            html.Div(
                                [
                                    html.Div([html.Span("AI 추천", className="recommend-badge"), html.Span("관리자 호출")], className="recommend-item"),
                                    html.Div([html.Span("AI 추천", className="recommend-badge"), html.Span("동시 가동 방지")], className="recommend-item"),
                                ],
                                className="recommend-list",
                            ),
                            html.Label("실행할 조치", className="field-label"),
                            dcc.Checklist(
                                id="drawer-action-types",
                                options=[
                                    {"label": "관리자 호출", "value": "관리자 호출"},
                                    {"label": "설비 상태 점검", "value": "설비 상태 점검"},
                                    {"label": "가동 연기", "value": "가동 연기"},
                                    {"label": "동시 가동 방지", "value": "동시 가동 방지"},
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
    rows = [
        ("1", "3시간 조업 모멘텀", "151.2 kW", "80~140 kW", "위험 증가", "미확인", "danger"),
        ("2", "점심 반등 예상량", "42.6 kW", "0~35 kW", "위험 증가", "점검 중", "caution"),
        ("3", "기온 축열", "31.4℃", "18~28℃", "위험 증가", "점검 필요", "caution"),
    ]
    return html.Main(
        [
            control_page_header(
                "CAUSE INSPECTION / 15:15",
                "피크 예측 기여 요인 및 공정조건 점검",
                [("예측 위험 82%", "danger"), ("선택 시점 15:15", "normal")],
            ),
            html.Div(
                [
                    html.Span("ANALYSIS", className="event-code"),
                    html.Strong("예측 기여 요인 점검"),
                    html.Span("1시간 최대 176.8 kW"),
                    html.Span("기준 초과 +16.8 kW", className="danger-text"),
                    html.Span("인과관계가 아닌 모델 기여도", className="event-note"),
                ],
                className="event-strip",
            ),
            html.Div(
                [
                    graph_card("선택 시점의 피크 예측 기여 요인 · 15:15", cause_figure(360), "cause-main", "cause-detail-chart"),
                    html.Div(
                        [
                            graph_card("전체 기간 변수 중요도 · 평균 |SHAP|", importance_figure()),
                            html.Section(
                                [html.Div([html.H3("AI 분석 요약", className="section-title"), html.Span("MODEL EXPLANATION", className="panel-code")], className="industrial-panel-header"), html.P("최근 3시간의 높은 조업 부하가 피크 예측을 가장 크게 높였습니다.", className="insight-copy")],
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
                    html.Div([html.Div([html.H3("공정조건 점검 우선순위", className="section-title"), html.Span("TOP 3 · 현재값/참고범위", className="panel-code")], className="panel-title-row"), html.Div([html.Button("점검 완료", className="secondary-button", disabled=True, title="실제 점검 데이터 연결 후 활성화"), html.Button("조치 시작 ›", className="danger-button small", disabled=True, title="조치 워크플로 연결 후 활성화")])], className="panel-header"),
                    html.Div(
                        [
                            html.Div([html.Strong("우선순위"), html.Strong("공정조건"), html.Strong("현재값"), html.Strong("참고범위"), html.Strong("영향 방향"), html.Strong("점검 상태")], className="data-row header-row"),
                            *[
                                html.Div([html.Span(rank), html.Span(condition), html.Strong(current, className="danger-text"), html.Span(reference), html.Span("↑  " + direction, className="danger-text"), status_pill(status, tone)], className="data-row")
                                for rank, condition, current, reference, direction, status, tone in rows
                            ],
                        ],
                        className="data-table condition-table",
                    ),
                ],
                className="panel table-panel",
            ),
            simulation_note("모델 설명은 인과관계가 아닌 예측 기여도를 의미합니다."),
        ],
        className="page-content industrial-page causes-control-page",
    )


def model_page():
    return html.Main(
        [
            control_page_header(
                "MODEL VALIDATION / TEST SET",
                "예측 모델 성능 및 오류 분석",
                [("회귀 · 실제 테스트", "normal"), ("피크 분류 · 시뮬레이션", "caution")],
            ),
            html.Div([html.Span("EVIDENCE", className="event-code info"), html.Strong("회귀 성능은 실제 테스트 결과"), html.Span("피크 분류 지표는 시뮬레이션"), html.Span("최종 파이프라인 연결 전", className="event-note")], className="event-strip info"),
            industrial_metric_rail(
                [
                    ("최종 회귀 모델", "Top-3", "GBDT", "테스트 RMSE 기준 1위", "forecast"),
                    ("RMSE", "9.34", "kW", "낮을수록 우수", "actual"),
                    ("MAE", "6.13", "kW", "평균 절대 오차", "actual"),
                    ("결정계수 R²", "0.9730", "", "설명력", "forecast"),
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
                                    html.Div([html.H3("피크 분류 성능", className="section-title"), html.Span("SIMULATION", className="panel-code caution")], className="panel-header"),
                                    html.Div([kpi_mini("F1", "0.86"), kpi_mini("Precision", "0.83"), kpi_mini("Recall", "0.89"), kpi_mini("FN", "8"), kpi_mini("FP", "13")], className="mini-grid"),
                                ],
                                className="panel",
                            ),
                            html.Section(
                                [html.Div([html.H3("최종 모델 선정 근거", className="section-title"), html.Span("DECISION BASIS", className="panel-code")], className="industrial-panel-header"), html.Div([html.Span("비선형 조업 패턴 대응"), html.Span("일반화 성능 1위"), html.Span("실시간 추론 가능")], className="reason-grid")],
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
            simulation_note("피크 분류 지표는 실제 분류 파이프라인 연결 후 교체합니다."),
        ],
        className="page-content industrial-page model-control-page",
    )


def kpi_mini(label: str, value: str):
    return html.Div([html.Span(label), html.Strong(value)], className="mini-kpi")


def actions_page():
    return html.Main(
        [
            control_page_header(
                "ALARM RESPONSE / SESSION",
                "경보 조치 및 처리 이력",
                [("미확인 2건", "danger"), ("처리 중 3건", "caution"), ("세션 저장", "normal")],
            ),
            html.Div([html.Span("SIMULATION", className="event-code info"), html.Strong("운영 조치 프로토타입"), html.Span("현재 세션 데이터"), html.Span("영구 DB 미연결", className="event-note")], className="event-strip info"),
            industrial_metric_rail(
                [
                    ("미확인", "2", "건", "즉시 확인 필요", "danger"),
                    ("처리 중", "3", "건", "담당자 배정", "threshold"),
                    ("완료", "12", "건", "선택 기간 누계", "forecast"),
                    ("평균 처리시간", "18", "분", "시뮬레이션 기준", "actual"),
                ],
                "four-metrics",
            ),
            html.Div(
                [
                    dropdown("기간", ["최근 7일", "최근 30일"], "최근 7일"),
                    dropdown("위험 단계", ["전체", "위험", "주의"], "전체"),
                    dropdown("처리 상태", ["전체", "미확인", "처리 중", "완료"], "전체"),
                    dropdown("담당자", ["전체", "김현수", "이지민", "박서준"], "전체"),
                    html.Button("CSV 내보내기 · 준비 중", className="primary-button export-button", disabled=True, title="영구 저장소 연결 후 활성화"),
                ],
                className="filter-bar",
            ),
            html.Div(
                [
                    html.Section(
                        [html.H3("경보·조치 목록", className="section-title"), html.Div(action_table(), id="action-table-container")],
                        className="panel action-list",
                    ),
                    action_detail(),
                ],
                className="action-grid",
            ),
            simulation_note("조치 기록은 현재 브라우저 세션에 저장되며, 영구 DB 저장은 최종 파이프라인 연결 후 활성화됩니다."),
        ],
        className="page-content industrial-page actions-control-page",
    )


def dropdown(label: str, options: list[str], value: str):
    return html.Div(
        [html.Label(label), dcc.Dropdown(options=[{"label": option, "value": option} for option in options], value=value, clearable=False, searchable=False)],
        className="filter-control",
    )


def action_table(records: list[dict] | None = None):
    header = html.Div([html.Strong(label) for label in ["경보 시각", "위험 단계", "최대 예상 전력", "담당자", "조치 유형", "처리 상태"]], className="action-row header-row")
    rows = []
    for idx, row in enumerate(records or ACTION_ROWS):
        risk_tone = "danger" if row["단계"] == "위험" else "caution"
        state_tone = "success" if row["상태"] == "완료" else "info" if row["상태"] == "처리 중" else "danger"
        rows.append(
            html.Div(
                [html.Span(row["시각"]), status_pill(row["단계"], risk_tone), html.Span(row["전력"]), html.Span(row["담당자"]), html.Span(row["조치"]), status_pill(row["상태"], state_tone)],
                className=f"action-row{' selected' if idx == 0 else ''}",
            )
        )
    return html.Div([header, *rows], className="data-table")


def action_detail():
    return html.Aside(
        [
            html.H3("조치 상세", className="section-title"),
            html.Div([detail_item("경보 시각", "10/15 14:30"), detail_item("담당자", "김현수"), detail_item("처리 상태", "처리 중")], className="detail-summary"),
            html.H4("공정조건 점검"),
            dcc.Checklist(
                options=[
                    {"label": "3시간 조업 모멘텀 확인", "value": "momentum"},
                    {"label": "점심 반등 예상량 확인", "value": "lunch"},
                    {"label": "기온 축열 확인", "value": "temperature"},
                ],
                value=["momentum", "lunch"],
                className="checklist",
            ),
            html.Label("조치 유형", className="field-label"),
            dcc.Dropdown(options=["관리자 호출", "설비 상태 점검", "가동 연기", "작업순서 조정"], value="관리자 호출", clearable=False),
            html.Label("메모", className="field-label"),
            dcc.Textarea(value="피크 위험 공유, 생산계획 조정 검토 중", className="memo-field"),
            html.Div([html.Button("저장", className="secondary-button"), html.Button("처리 완료", className="primary-button")], className="detail-actions"),
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
