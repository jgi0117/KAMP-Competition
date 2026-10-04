"""
generate_presentation.py
PowerPoint presentation generator for OKM Power Forecasting project.
Creates a professional, modern 16:9 widescreen presentation deck (PPTX + PDF).

High-Legibility & Scaled Typography Edition:
- Main Titles scaled to 24pt (Cover: 42pt, Dividers: 40pt)
- Subtitles scaled to 13.5pt - 18pt
- Card Titles scaled to 13.5pt - 17pt
- Body & Table Text scaled to 10.5pt - 12.5pt (up from 8.5pt - 9pt)
- Natural Korean line breaks & clean spacing for superior readability on projectors/monitors.
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
WORKSPACE = Path(__file__).resolve().parent.parent
REPORTS_DIR = WORKSPACE / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PPTX_PATH = REPORTS_DIR / "EDA_전처리_및_피처엔지니어링_보고.pptx"
PDF_PATH = REPORTS_DIR / "EDA_전처리_및_피처엔지니어링_보고.pdf"

# Design Palette (Modern Tech Blue)
COLOR_BG_WHITE = RGBColor(255, 255, 255)
COLOR_NAVY = RGBColor(15, 23, 42)          # #0F172A (Deep Navy Title)
COLOR_BLUE = RGBColor(37, 99, 235)         # #2563EB (Accent Primary Blue)
COLOR_LIGHT_BLUE = RGBColor(239, 246, 255) # #EFF6FF (Card/Row Tint)
COLOR_BORDER_BLUE = RGBColor(191, 219, 254)# #BFDBFE (Card Border)
COLOR_SLATE = RGBColor(51, 65, 85)         # #334155 (Body text)
COLOR_MUTED = RGBColor(100, 116, 139)      # #64748B (Subtitles/meta)
COLOR_CARD_BG = RGBColor(248, 250, 252)    # #F8FAFC (Card Fill)
COLOR_CARD_BORDER = RGBColor(226, 232, 240)# #E2E8F0 (Card Border)
COLOR_AMBER = RGBColor(217, 119, 6)        # #D97706 (Highlights)
COLOR_GREEN = RGBColor(16, 185, 129)       # #10B981 (Success/Done)
FONT_FAMILY = "맑은 고딕"

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    TOTAL_SLIDES = "18"

    def add_header(slide, section_tag, title_text, subtitle_text, slide_num_str):
        # Section Tag (Scaled: 11.5pt)
        tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.35), Inches(10.0), Inches(0.28))
        tf_tag = tag_box.text_frame
        tf_tag.word_wrap = True
        tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
        p_tag = tf_tag.paragraphs[0]
        p_tag.text = section_tag.upper()
        p_tag.font.name = FONT_FAMILY
        p_tag.font.size = Pt(11.5)
        p_tag.font.bold = True
        p_tag.font.color.rgb = COLOR_BLUE

        # Title (Scaled: 24pt)
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(10.5), Inches(0.58))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_FAMILY
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_NAVY

        # Subtitle (Scaled: 13.5pt)
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.25), Inches(10.5), Inches(0.38))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_FAMILY
        p_sub.font.size = Pt(13.5)
        p_sub.font.color.rgb = COLOR_MUTED

        # Slide Number (Scaled: 12pt)
        num_box = slide.shapes.add_textbox(Inches(11.5), Inches(0.35), Inches(1.0), Inches(0.28))
        tf_num = num_box.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = slide_num_str
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = FONT_FAMILY
        p_num.font.size = Pt(12)
        p_num.font.bold = True
        p_num.font.color.rgb = COLOR_MUTED

        # Header dividing line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.72), Inches(11.733), Inches(0.015))
        line.fill.solid()
        line.fill.fore_color.rgb = COLOR_CARD_BORDER
        line.line.color.rgb = COLOR_CARD_BORDER

    def add_section_divider(slide, chapter_tag, title_text, desc_text, slide_num_str):
        top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
        top_bar.fill.solid()
        top_bar.fill.fore_color.rgb = COLOR_BLUE
        top_bar.line.fill.background()

        num_box = slide.shapes.add_textbox(Inches(11.5), Inches(0.38), Inches(1.0), Inches(0.28))
        tf_num = num_box.text_frame
        p_num = tf_num.paragraphs[0]
        p_num.text = slide_num_str
        p_num.alignment = PP_ALIGN.RIGHT
        p_num.font.name = FONT_FAMILY
        p_num.font.size = Pt(12)
        p_num.font.bold = True
        p_num.font.color.rgb = COLOR_MUTED

        center_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.5), Inches(10.933), Inches(4.8))
        center_card.fill.solid()
        center_card.fill.fore_color.rgb = COLOR_CARD_BG
        center_card.line.color.rgb = COLOR_CARD_BORDER
        center_card.line.width = Pt(1.5)

        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.8), Inches(2.1), Inches(3.0), Inches(0.48))
        pill.fill.solid()
        pill.fill.fore_color.rgb = COLOR_LIGHT_BLUE
        pill.line.color.rgb = COLOR_BORDER_BLUE
        tf_pill = pill.text_frame
        p_pill = tf_pill.paragraphs[0]
        p_pill.text = chapter_tag
        p_pill.alignment = PP_ALIGN.CENTER
        p_pill.font.name = FONT_FAMILY
        p_pill.font.size = Pt(14)
        p_pill.font.bold = True
        p_pill.font.color.rgb = COLOR_BLUE

        tb_t = slide.shapes.add_textbox(Inches(1.8), Inches(2.85), Inches(9.8), Inches(1.3))
        tf_t = tb_t.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.name = FONT_FAMILY
        p_t.font.size = Pt(32)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY

        acc_line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.8), Inches(4.35), Inches(9.6), Inches(0.03))
        acc_line.fill.solid()
        acc_line.fill.fore_color.rgb = COLOR_BLUE
        acc_line.line.fill.background()

        tb_d = slide.shapes.add_textbox(Inches(1.8), Inches(4.6), Inches(9.6), Inches(1.3))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        tf_d.margin_left = tf_d.margin_top = tf_d.margin_right = tf_d.margin_bottom = 0
        p_d = tf_d.paragraphs[0]
        p_d.text = desc_text
        p_d.font.name = FONT_FAMILY
        p_d.font.size = Pt(17)
        p_d.font.color.rgb = COLOR_SLATE

    def add_card_box(slide, left, top, width, height, title, items, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER, title_color=COLOR_NAVY, title_size=Pt(14), body_size=Pt(11)):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1)

        tb = slide.shapes.add_textbox(left + Inches(0.18), top + Inches(0.12), width - Inches(0.36), height - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_FAMILY
        p_t.font.size = title_size
        p_t.font.bold = True
        p_t.font.color.rgb = title_color
        p_t.space_after = Pt(4)

        for item in items:
            p = tf.add_paragraph()
            p.font.name = FONT_FAMILY
            p.font.size = body_size
            p.space_after = Pt(3)

            if isinstance(item, tuple):
                tag, desc = item
                r_tag = p.add_run()
                r_tag.text = tag + " "
                r_tag.font.bold = True
                r_tag.font.color.rgb = COLOR_NAVY
                
                r_desc = p.add_run()
                r_desc.text = desc
                r_desc.font.color.rgb = COLOR_SLATE
            else:
                p.text = item
                p.font.color.rgb = COLOR_SLATE

    def add_table(slide, left, top, width, height, headers, rows, col_widths=None, header_bg=COLOR_NAVY, font_size=Pt(10.5)):
        table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
        tbl = table_shape.table

        if col_widths:
            for i, w in enumerate(col_widths):
                tbl.columns[i].width = w

        for c, h in enumerate(headers):
            cell = tbl.cell(0, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            p.text = h
            p.font.name = FONT_FAMILY
            p.font.size = font_size
            p.font.bold = True
            p.font.color.rgb = RGBColor(255, 255, 255)
            p.alignment = PP_ALIGN.CENTER

        for r, row in enumerate(rows):
            bg = COLOR_LIGHT_BLUE if r % 2 == 1 else COLOR_BG_WHITE
            for c, val in enumerate(row):
                cell = tbl.cell(r + 1, c)
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE
                cell.margin_left = Inches(0.06)
                cell.margin_right = Inches(0.06)
                cell.margin_top = Inches(0.02)
                cell.margin_bottom = Inches(0.02)

                lines = str(val).split("\n")
                for li, l_text in enumerate(lines):
                    p = cell.text_frame.paragraphs[0] if li == 0 else cell.text_frame.add_paragraph()
                    p.text = l_text
                    p.font.name = FONT_FAMILY
                    p.font.size = font_size
                    p.font.color.rgb = COLOR_NAVY if (c == 0 and len(row) > 2) else COLOR_SLATE
                    if c == 0 or (len(row) == 2 and c == 0):
                        p.font.bold = True
                        if len(headers) >= 3 and c == 0:
                            p.font.color.rgb = COLOR_BLUE

    def add_image_panel(slide, img_path, left, top, width, height, caption=""):
        frame = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        frame.fill.solid()
        frame.fill.fore_color.rgb = RGBColor(255, 255, 255)
        frame.line.color.rgb = COLOR_CARD_BORDER
        frame.line.width = Pt(1)

        if os.path.exists(img_path):
            img_margin = Inches(0.06)
            slide.shapes.add_picture(
                str(img_path),
                left + img_margin,
                top + img_margin,
                width - (img_margin * 2),
                height - (img_margin * 2) - (Inches(0.24) if caption else 0)
            )
            if caption:
                cap_box = slide.shapes.add_textbox(left, top + height - Inches(0.24), width, Inches(0.22))
                tf = cap_box.text_frame
                tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
                p = tf.paragraphs[0]
                p.text = caption
                p.alignment = PP_ALIGN.CENTER
                p.font.name = FONT_FAMILY
                p.font.size = Pt(10)
                p.font.color.rgb = COLOR_MUTED

    # -------------------------------------------------------------
    # SLIDE 1: COVER
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    top_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = COLOR_BLUE
    top_bar.line.fill.background()

    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(1.6), Inches(3.6), Inches(0.48))
    pill.fill.solid()
    pill.fill.fore_color.rgb = COLOR_LIGHT_BLUE
    pill.line.color.rgb = COLOR_BORDER_BLUE
    tf_pill = pill.text_frame
    p_pill = tf_pill.paragraphs[0]
    p_pill.text = "KAMP 제조 AI 경진대회 분석 보고서"
    p_pill.alignment = PP_ALIGN.CENTER
    p_pill.font.name = FONT_FAMILY
    p_pill.font.size = Pt(13)
    p_pill.font.bold = True
    p_pill.font.color.rgb = COLOR_BLUE

    tb_title = s1.shapes.add_textbox(Inches(1.2), Inches(2.35), Inches(11.0), Inches(1.9))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    
    p1 = tf_title.paragraphs[0]
    p1.text = "OKM 제조 설비 전력사용량 예측을 위한"
    p1.font.name = FONT_FAMILY
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_NAVY
    p1.space_after = Pt(6)

    p2 = tf_title.add_paragraph()
    p2.text = "데이터 전처리 및 피처 엔지니어링 계획"
    p2.font.name = FONT_FAMILY
    p2.font.size = Pt(42)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_BLUE

    tb_desc = s1.shapes.add_textbox(Inches(1.2), Inches(4.5), Inches(10.5), Inches(0.85))
    tf_desc = tb_desc.text_frame
    tf_desc.word_wrap = True
    tf_desc.margin_left = tf_desc.margin_top = tf_desc.margin_right = tf_desc.margin_bottom = 0
    p_desc = tf_desc.paragraphs[0]
    p_desc.text = "도메인 물리량 기반 데이터 무결성 검증, 심층 EDA 시각화 5대 인사이트,\n그리고 예측 정확도 극대화를 위한 31개 파생변수 후보군 상세 명세"
    p_desc.font.name = FONT_FAMILY
    p_desc.font.size = Pt(16)
    p_desc.font.color.rgb = COLOR_MUTED

    meta_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(5.6), Inches(10.933), Inches(1.05))
    meta_card.fill.solid()
    meta_card.fill.fore_color.rgb = COLOR_CARD_BG
    meta_card.line.color.rgb = COLOR_CARD_BORDER
    
    tb_meta = s1.shapes.add_textbox(Inches(1.5), Inches(5.72), Inches(10.3), Inches(0.8))
    tf_meta = tb_meta.text_frame
    tf_meta.word_wrap = True
    p_meta1 = tf_meta.paragraphs[0]
    p_meta1.text = "작성자 / 발표자 : 김정현 (LSAXMakers Data Team)       |       프로젝트 브랜치 : git checkout kjh"
    p_meta1.font.name = FONT_FAMILY
    p_meta1.font.size = Pt(13)
    p_meta1.font.bold = True
    p_meta1.font.color.rgb = COLOR_NAVY
    p_meta1.space_after = Pt(4)
    
    p_meta2 = tf_meta.add_paragraph()
    p_meta2.text = "핵심 키워드 : #결측치·이상치완전정제  #도메인대기전력검증  #시계열자기상관성(168h)  #31개파생변수  #TimeSeriesSplit"
    p_meta2.font.name = FONT_FAMILY
    p_meta2.font.size = Pt(11.5)
    p_meta2.font.color.rgb = COLOR_BLUE

    # -------------------------------------------------------------
    # SLIDE 2: AGENDA
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "AGENDA", "전체 회의 목차 및 진행 순서", "데이터 정제부터 시각화 인사이트, 파생변수 계획까지 4단계로 논의합니다.", f"02 / {TOTAL_SLIDES}")

    agenda_cards = [
        ("01", "프로젝트 배경 및 데이터 개요", [
            ("• 예측 목표:", "OKM 제조 공장의 시간대별 전력 예측 및 180 kW 이상 피크 부하 사전 탐지"),
            ("• 데이터 규모:", "2021년 1-9월 총 6,168시간 연속 시계열 및 3개 도메인 18개 원본 피처 특성")
        ]),
        ("02", "전처리 파이프라인 및 무결성 검증", [
            ("• 4대 정제 성과:", "공장인원 결측(17건) 셧다운 0.0 대체, 풍속/강수 선형 보간, 7월 시간 이상치 정상 복원"),
            ("• 도메인 검증:", "생산량 ↔ 인원 1:1 완벽 일치 및 무생산 시간 대기전력 Base Load(평균 46.5 kW) 입증")
        ]),
        ("03", "심층 EDA 5대 인사이트 & 파생변수 (핵심)", [
            ("• 5대 시각화 규명:", "12시 점심 급감(-43kW), 요일x시간 조업 히트맵, 7일 주기(168h), 이봉분포, 기상 U자 곡선"),
            ("• 파생변수 도출:", "각 시각화 결과와 1:1로 직결되는 5개 범주 총 31개 파생변수 풀 명세")
        ]),
        ("04", "모델링 검증 전략 및 향후 로드맵", [
            ("• 검증 전략:", "미래 데이터 누수(Data Leakage)를 원천 차단하는 TimeSeriesSplit 4-Fold 교차검증"),
            ("• 마일스톤:", "LightGBM/XGBoost/CatBoost 베이스라인 벤치마킹 및 4단계 실행 로드맵")
        ])
    ]

    for i, (num, title, items) in enumerate(agenda_cards):
        top_pos = Inches(1.95 + i * 1.28)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_pos, Inches(11.733), Inches(1.15))
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER

        badge = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.05), top_pos + Inches(0.2), Inches(0.9), Inches(0.75))
        badge.fill.solid()
        badge.fill.fore_color.rgb = COLOR_BLUE
        badge.line.fill.background()
        tf_b = badge.text_frame
        p_b = tf_b.paragraphs[0]
        p_b.text = num
        p_b.alignment = PP_ALIGN.CENTER
        p_b.font.name = FONT_FAMILY
        p_b.font.size = Pt(22)
        p_b.font.bold = True
        p_b.font.color.rgb = RGBColor(255, 255, 255)

        tb = s2.shapes.add_textbox(Inches(2.2), top_pos + Inches(0.1), Inches(10.0), Inches(0.95))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_FAMILY
        p_t.font.size = Pt(16)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_NAVY
        p_t.space_after = Pt(2)

        for tag, desc in items:
            p_i = tf.add_paragraph()
            p_i.font.name = FONT_FAMILY
            p_i.font.size = Pt(12)
            r1 = p_i.add_run()
            r1.text = tag + " "
            r1.font.bold = True
            r1.font.color.rgb = COLOR_BLUE
            r2 = p_i.add_run()
            r2.text = desc
            r2.font.color.rgb = COLOR_SLATE

    # -------------------------------------------------------------
    # SLIDE 3: SECTION 01 DIVIDER
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_section_divider(
        s3, "SECTION 01 / 04",
        "프로젝트 배경 및 데이터 개요",
        "제조 설비 전력사용량 예측 과제의 목표 정의와 6,168시간 원본 데이터셋의 도메인별 특성을 소개합니다.",
        f"03 / {TOTAL_SLIDES}"
    )

    # -------------------------------------------------------------
    # SLIDE 4: SECTION 01 CONTENT (문제 정의 및 예측 목표 - LARGE FONT & KEYWORDS)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "SECTION 01. 프로젝트 배경 및 데이터 개요", "제조 설비 전력사용량 예측 과제 정의 및 원본 데이터 구조", "단순 평균 전력 예측을 넘어 전력 피크(Peak Load) 위험을 사전 감지하는 것이 핵심 목표입니다.", f"04 / {TOTAL_SLIDES}")

    add_card_box(
        s4, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.1),
        "🎯 문제 정의 및 예측 목표",
        [
            ("📌 핵심 예측 목표:", "OKM 공장의 시간대별 전력사용량(kW) 정밀 예측"),
            ("💰 비즈니스 가치:", "180 kW 이상 최대 피크 부하 사전 탐지\n• 기본요금 할증 방지 및 공장 설비 과부하 예방"),
            ("📊 데이터 수집 규모:", "총 6,168시간 연속 시계열 (2021.01.01 - 09.14)"),
            ("⚡ 해결 과제 1 (조업 급변):", "출퇴근 · 교대근무 · 점심 43kW 급감 · 주말 휴무"),
            ("⚙️ 해결 과제 2 (기동 부하):", "월요일 08시 설비 재가동(Start-up) 초기 피크"),
            ("🌡️ 해결 과제 3 (기상 비선형):", "외기 온·습도에 따른 냉난방 공조 부하의 U자 반응"),
            ("💡 분석 전략 방향:", "공장 조업 규칙을 반영한 파생변수 집중 주입")
        ],
        title_color=COLOR_NAVY, title_size=Pt(17), body_size=Pt(12.5)
    )

    right_cards = [
        ("📅 1. 시간 및 캘린더 영역 (8개 피처)", [
            ("• 주요 피처:", "날짜(datetime), 월(m), 일(d), 요일(day: 1-7), 시간(0-23시)"),
            ("• 도메인 가치:", "교대근무, 점심시간, 주말 휴무 등 규칙적 조업 주기 파악"),
            ("• 품질 이슈:", "7월 시간 컬럼 70-188 비정상값 48건 발생 (기온 곡선 검증 후 정상 복원 완료)")
        ], Inches(1.9), Inches(1.58)),
        ("🏭 2. 공장 조업 및 설비 영역 (3개 피처)", [
            ("• 주요 피처:", "공장인원(명), 생산량(개), 설비용량(kW - 상수값 300)"),
            ("• 도메인 가치:", "제품 생산 활동 및 작업자 투입에 따른 직접적 전력 소비량 추적"),
            ("• 품질 이슈:", "8월 말 인원 17시간 연속 결측 발생 (셧다운 인과성 규명 후 0.0 대체)")
        ], Inches(3.64), Inches(1.58)),
        ("🌤️ 3. 기상 및 환경 관측 영역 (7개 피처)", [
            ("• 주요 피처:", "기온(℃), 습도(%), 풍속(m/s), 강수량(mm), 일조, 일사, 적설"),
            ("• 도메인 가치:", "외부 온습도 변화에 따른 공장 내부 공조/냉난방 설비 부하 추적"),
            ("• 품질 이슈:", "풍속 3건, 강수량 1건 결측 발생 (대기 연속성 기반 선형 보간 적용)")
        ], Inches(5.38), Inches(1.62))
    ]
    for c_title, c_items, c_top, c_h in right_cards:
        add_card_box(s4, Inches(6.8), c_top, Inches(5.733), c_h, c_title, c_items, bg_color=COLOR_LIGHT_BLUE, border_color=COLOR_BORDER_BLUE, title_color=COLOR_BLUE, title_size=Pt(14), body_size=Pt(11))

    # -------------------------------------------------------------
    # SLIDE 5: SECTION 02 DIVIDER
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_section_divider(
        s5, "SECTION 02 / 04",
        "전처리 파이프라인 및 데이터 품질 검증",
        "결측치 0개 정제, 시간 이상치 복원 및 생산-인원 정비례와 대기전력(Base Load)의 물리 법칙을 규명합니다.",
        f"05 / {TOTAL_SLIDES}"
    )

    # -------------------------------------------------------------
    # SLIDE 6: SECTION 02 CONTENT 1
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "SECTION 02. 전처리 파이프라인 및 무결성 검증", "4대 핵심 전처리 기법 적용 및 시계열 무결성 확보", "도메인 인과관계를 철저히 검증하여 결측치 0개 및 시간 이상치 완전 복원을 달성했습니다.", f"06 / {TOTAL_SLIDES}")

    prep_cards = [
        ("👥 1. 공장인원 결측 (17건)", [
            ("• 발생 현황:", "8/28 18시 - 8/29 10시 17시간 연속 결측"),
            ("• 정제 조치:", "해당 기간 전력 0 kW(셧다운), 생산량 0 확인 후 '0.0' 대체 확정"),
            ("• 도메인 근거:", "전체 데이터에서 인원=0 ↔ 생산량=0 100% 성립 (휴무 기간 결측 입증)")
        ], Inches(1.9), Inches(1.15)),
        ("🌬️ 2. 풍속·강수량 결측 (4건)", [
            ("• 발생 현황:", "전체 6,168시간 중 단 1-2건 국소 누락"),
            ("• 정제 조치:", "직전·직후 시간 관측치 기준 '선형 보간(Linear Interpolation)' 적용"),
            ("• 도메인 근거:", "대기 물리량의 시간적 연속성을 반영하여 왜곡 최소화")
        ], Inches(3.18), Inches(1.15)),
        ("⏰ 3. 시간 컬럼 이상치 (48건)", [
            ("• 발생 현황:", "7/13, 7/15에 70-188 비정상값 기록"),
            ("• 정제 조치:", "기온 일변화 곡선 100% 일치 확인 후 행 순서대로 '0-23시' 정상 재부여"),
            ("• 도메인 근거:", "전후 정상일과 24시간 기온 일변화 궤적이 완벽 일치 (원본 보존)")
        ], Inches(4.46), Inches(1.15)),
        ("⚡ 4. 전력 계산 정밀도 (float64) & 셧다운 플래그", [
            ("• 정제 조치 1:", "원본 평균 반올림 오차(+-0.5 kW) 제거 위해 '전력_평균_실수' 신설"),
            ("• 정제 조치 2:", "정전/셧다운 특수 구간 분리를 위한 '공장_셧다운_여부' 불리언 피처 생성")
        ], Inches(5.74), Inches(1.26))
    ]

    for p_title, p_items, p_top, p_h in prep_cards:
        add_card_box(s6, Inches(0.8), p_top, Inches(5.6), p_h, p_title, p_items, title_color=COLOR_NAVY, title_size=Pt(13.5), body_size=Pt(10.5))

    add_image_panel(
        s6, FIGURES_DIR / "00_full_timeseries_overview.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[그림 1] 9개월 전체 시계열(6,168시간) 전력 추이 및 이상치 발생 위치 대조"
    )

    # -------------------------------------------------------------
    # SLIDE 7: SECTION 02 CONTENT 2
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "SECTION 02. 전처리 품질 검증 (국소 확대 & 대기전력)", "생산량-인원 정비례 검증 및 비정상 셧다운 통계 분석", "비생산 시간에도 대기전력(평균 46.5 kW)이 항상 흐르고 있어 0 kW 셧다운과의 물리적 차이가 입증됩니다.", f"07 / {TOTAL_SLIDES}")

    val_cards = [
        ("🔗 1. 생산량 ↔ 인원 1:1 완벽 일치", [
            ("• 통계 확인:", "전체 6,168행에서 인원=0 ↔ 생산량=0 완벽 성립"),
            ("• 규명 의미:", "작업자가 없으면 생산도 없다는 도메인 인과성을 산점도로 통계적 입증")
        ], Inches(1.9), Inches(1.15)),
        ("🔌 2. 대기전력(Base Load)의 상시 존재", [
            ("• 통계 확인:", "무생산(생산량=0) 시간에도 평균 46.5 kW (최소 17 kW) 상시 소모"),
            ("• 규명 의미:", "공장이 쉬어도 조명, 통신, 보안 유지를 위한 기저 부하가 24시간 작동")
        ], Inches(3.18), Inches(1.15)),
        ("🛑 3. 8/28-29 비정상 셧다운 통계적 규명", [
            ("• 통계 확인:", "해당 17시간 동안은 전력이 완전 0.0 kW로 차단됨"),
            ("• 규명 의미:", "단순 무생산 휴무(46.5 kW)와 달리 전원이 완전 차단된 정전/셧다운 규명")
        ], Inches(4.46), Inches(1.15)),
        ("🔍 4. 국소 확대(Local Zoom-in) 검증 의의", [
            ("• 통계 확인:", "6,168행 전체 시계열 대비 36시간 및 13시간 국소 확대 대조"),
            ("• 규명 의미:", "전체 시계열에서 안 보이는 결측치 전후 연속성을 확대 플롯으로 완벽 입증")
        ], Inches(5.74), Inches(1.26))
    ]

    for v_title, v_items, v_top, v_h in val_cards:
        add_card_box(s7, Inches(0.8), v_top, Inches(5.6), v_h, v_title, v_items, title_color=COLOR_NAVY, title_size=Pt(13.5), body_size=Pt(10.5))

    add_image_panel(
        s7, FIGURES_DIR / "04_production_power_relationship.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[그림 2] 생산량-인원 정비례 관계(좌) 및 생산량=0 구간 대기전력 분포(우)"
    )

    # -------------------------------------------------------------
    # SLIDE 8: SECTION 03 DIVIDER
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_section_divider(
        s8, "SECTION 03 / 04",
        "심층 EDA 5대 인사이트 & 파생변수 도출",
        "도메인 시각화 분석을 통해 조업 주기, 설비 상태, 기상 U자 곡선을 규명하고 31개 핵심 파생변수를 1:1로 도출합니다.",
        f"08 / {TOTAL_SLIDES}"
    )

    # -------------------------------------------------------------
    # SLIDE 9: INSIGHT 01 (TABLES - SCALED TO 10.5pt)
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "SECTION 03. 심층 EDA 5대 인사이트 & 파생변수 도출", "인사이트 01: 시간대별 조업 주기와 점심시간 급감(Dip)", "평일 12시 점심시간에 전력이 평균 43.3 kW 급감하며, 평일 주간과 주말의 전력 프로파일이 완전히 분리됩니다.", f"09 / {TOTAL_SLIDES}")

    s9_stat_h = ["통계 분석 항목", "정량적 수치", "쉬운 설명 (도메인 해석)"]
    s9_stat_r = [
        ("12시 점심 급감\n(Dip 현상)", "11시 127.3 kW ->\n12시 84.0 kW 급감\n(-43.3 kW, -34.0%)", "식사 시간 설비 일시정지로 급감 후\n13시(123.5 kW)에 즉시 복귀"),
        ("주간 집중 조업", "오전(08-11시) 110-127 kW\n오후(13-16시) 118-124 kW", "공장 정규 생산 라인이 풀가동되는\n하루 중 최고 부하 집중 시간대"),
        ("평일 vs 주말", "평일 112.4 kW vs 주말 48.2 kW\n(일요일 중앙값 22.5 kW)", "주말에는 생산 라인이 멈추고\n기본 인프라 대기전력만 소모")
    ]
    add_table(s9, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4), s9_stat_h, s9_stat_r, [Inches(1.4), Inches(1.6), Inches(2.6)], font_size=Pt(10.5))

    s9_feat_h = ["생성 파생변수", "산출 로직", "시각화 도출 근거 및 모델링 기대 효과"]
    s9_feat_r = [
        ("점심시간여부", "(시간 == 12).astype(int)", "12시 43kW V자 급감 반영, 점심 과대 예측 방지"),
        ("주말여부", "요일.isin([6, 7]).astype(int)", "평일-주말 박스플롯 분리 확인, 주말 조업 분기"),
        ("조업시간대", "0:심야, 1:오전, 2:점심, 3:오후, 4:야간", "시간대별 계단식 전력 레벨 범주형 학습"),
        ("시간_sin, cos", "sin/cos(2 * pi * 시간 / 24)", "23시와 00시의 원형 연속성 보존")
    ]
    add_table(s9, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5), s9_feat_h, s9_feat_r, [Inches(1.3), Inches(1.6), Inches(2.7)], header_bg=COLOR_BLUE, font_size=Pt(10.5))

    add_image_panel(
        s9, FIGURES_DIR / "fe_01_time_day_patterns.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[시각화 01] 시간대별 평균 전력 및 점심 급감(좌) / 요일별 조업-주말 프로파일(우)"
    )

    # -------------------------------------------------------------
    # SLIDE 10: INSIGHT 02 (TABLES - SCALED TO 10.5pt)
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "SECTION 03. 심층 EDA 5대 인사이트 & 파생변수 도출", "인사이트 02: 2차원 조업 히트맵과 월요일 기동(Start-up) 부하", "월-금 08-17시에 고부하가 집중되며, 특히 월요일 아침 설비 재가동 시 순간 피크 부하가 발생합니다.", f"10 / {TOTAL_SLIDES}")

    s10_stat_h = ["통계 분석 항목", "정량적 수치", "쉬운 설명 (도메인 해석)"]
    s10_stat_r = [
        ("월-금 주간 블록", "평일 08-17시 평균 140-170 kW\n(히트맵상 청색 사각 영역)", "월-금 정규 조업시간대에만 140 kW 이상\n고부하가 직사각형 블록으로 집중"),
        ("월요일 08시 기동", "월요일 08시 평균 169 kW\n(화-금 대비 약 35 kW 높음)", "주말 동안 냉각된 설비 재가동 시\n순간 시동(Start-up) 피크 발생"),
        ("토요일 교대/휴무", "00-06시 85-101 kW -> 07시후 30-35 kW\n(일요일 25-55 kW 유지)", "금요일 야간 조업 잔여 후\n토요일 오전부터 비가동 전환")
    ]
    add_table(s10, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4), s10_stat_h, s10_stat_r, [Inches(1.4), Inches(1.6), Inches(2.6)], font_size=Pt(10.5))

    s10_feat_h = ["생성 파생변수", "산출 로직", "시각화 도출 근거 및 모델링 기대 효과"]
    s10_feat_r = [
        ("조업집중시간여부", "(평일==1) & (시간 8-11, 13-17)", "140-170 kW 고부하 집중 영역 마스킹"),
        ("월요일기동시간여부", "(요일==1) & (시간==8)", "월요일 08시 예열 피크 부하 오차 보정"),
        ("공휴일여부", "2021년 법정공휴일 매핑", "평일 요일 공휴일의 비가동 저부하 특성 반영"),
        ("영업일여부", "(주말==0) & (공휴일==0)", "실제 공장 정상 조업 수행일 마스터 피처")
    ]
    add_table(s10, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5), s10_feat_h, s10_feat_r, [Inches(1.4), Inches(1.5), Inches(2.7)], header_bg=COLOR_BLUE, font_size=Pt(10.5))

    add_image_panel(
        s10, FIGURES_DIR / "fe_02_heatmap_hour_day.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[시각화 02] 요일 x 시간대 2차원 전력사용량 히트맵 (평일 조업 집중 및 월요일 기동 부하)"
    )

    # -------------------------------------------------------------
    # SLIDE 11: INSIGHT 03 (TABLES - SCALED TO 10.5pt)
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "SECTION 03. 심층 EDA 5대 인사이트 & 파생변수 도출", "인사이트 03: 시계열 지연(Lag) 상관성과 7일 조업 주기", "1시간 전 전력(r=0.90)과 1주일 전 전력(r=0.74)이 전일 동시간(r=0.37)보다 2배 강력한 상관성을 보입니다.", f"11 / {TOTAL_SLIDES}")

    s11_stat_h = ["통계 분석 항목", "정량적 수치", "쉬운 설명 (도메인 해석)"]
    s11_stat_r = [
        ("초단기 관성 (Lag 1h)", "피어슨 상관계수 r = 0.9009\n(Lag 2h 0.85, Lag 3h 0.79)", "직전 1시간 전력은 현재 전력을\n결정하는 가장 강력한 기준점"),
        ("7일 주기 (Lag 168h)", "1주일 전 상관계수 r = 0.7397\n(전일 동시간 r=0.37 대비 2배)", "'월-금 가동, 토-일 휴무' 주기로 인해\n1주일 전 동일 요일과 완벽 동기화"),
        ("24시간 ACF 파동", "24, 48, 72시간마다 규칙적 피크", "하루 24시간 일변화 조업 사이클의 규칙적 반복")
    ]
    add_table(s11, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4), s11_stat_h, s11_stat_r, [Inches(1.4), Inches(1.6), Inches(2.6)], font_size=Pt(10.5))

    s11_feat_h = ["생성 파생변수", "산출 로직", "시각화 도출 근거 및 모델링 기대 효과"]
    s11_feat_r = [
        ("전력_lag1/2/3", "전력(t-1), 전력(t-2), 전력(t-3)", "산점도상 초강력 선형 관성(r=0.90) 모델 주입"),
        ("전력_lag168", "전력(t-168) (7일 전 동일 시간)", "7일 조업 주기(평일-주말 사이클) 완벽 동기화"),
        ("전력_lag24", "전력(t-24) (전일 동일 시간)", "어제 대비 오늘의 조업 레벨 증감 추세 반영"),
        ("rolling_mean/std", "최근 3h, 24h 이동평균 및 표준편차", "단기 노이즈 평활화 및 급격한 변동성 감지")
    ]
    add_table(s11, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5), s11_feat_h, s11_feat_r, [Inches(1.3), Inches(1.6), Inches(2.7)], header_bg=COLOR_BLUE, font_size=Pt(10.5))

    add_image_panel(
        s11, FIGURES_DIR / "fe_03_lag_correlations.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[시각화 03] 시계열 지연 산점도(Lag 1h, 2h, 24h, 168h) 및 168시간(1주일) ACF 파동"
    )

    # -------------------------------------------------------------
    # SLIDE 12: INSIGHT 04 (TABLES - SCALED TO 10.5pt)
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "SECTION 03. 심층 EDA 5대 인사이트 & 파생변수 도출", "인사이트 04: 생산 활동과 대기전력의 이봉분포(Bimodal)", "공장 전력은 대기전력(20-25 kW)과 정상 가동(100-160 kW)의 완전한 2단계 계층으로 분리됩니다.", f"12 / {TOTAL_SLIDES}")

    s12_stat_h = ["통계 분석 항목", "정량적 수치", "쉬운 설명 (도메인 해석)"]
    s12_stat_r = [
        ("이봉분포 (Bimodal)", "대기 23 kW 피크 vs 가동 135 kW 피크\n(중간값 50-80 kW 거의 없음)", "단일 정규분포가 아닌\n완전한 2대 기본 모드로 분리"),
        ("설비 재가동 부하", "정지->가동 전환 시간 평균 118.8 kW\n(생산량=0 구간 46.5 kW 대비 급등)", "꺼진 설비를 켤 때 모터 기동 전류와\n예열 부하로 순간 급증"),
        ("생산량 비선형성", "생산량 0개: 46kW -> 1-500개: 118kW\n(500개 초과: 138-160kW 완만 증가)", "가동 시작 시 기본 부하(100kW) 후\n생산량에 따라 계단식 증가")
    ]
    add_table(s12, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4), s12_stat_h, s12_stat_r, [Inches(1.4), Inches(1.6), Inches(2.6)], font_size=Pt(10.5))

    s12_feat_h = ["생성 파생변수", "산출 로직", "시각화 도출 근거 및 모델링 기대 효과"]
    s12_feat_r = [
        ("가동상태여부", "(생산량 > 0).astype(int)", "Bimodal 2대 모드(대기 vs 가동) 1차 판별"),
        ("설비기동여부", "(생산량_lag1==0) & (생산량>0)", "라인 재가동 시 초기 기동 부하(118.8kW) 포착"),
        ("인당생산량", "생산량 / (공장인원 + 0.1)", "작업자 투입 대비 조업 밀도 및 설비 효율 정량화"),
        ("부하율_lag1", "전력_평균_실수(t-1) / 설비용량(300)", "설비 연속 가동률 및 정속 운전 상태 구분")
    ]
    add_table(s12, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5), s12_feat_h, s12_feat_r, [Inches(1.3), Inches(1.6), Inches(2.7)], header_bg=COLOR_BLUE, font_size=Pt(10.5))

    add_image_panel(
        s12, FIGURES_DIR / "fe_04_production_dynamics.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[시각화 04] 전력 이봉분포 KDE 곡선(좌) 및 생산량 구간별 전력사용량 박스플롯(우)"
    )

    # -------------------------------------------------------------
    # SLIDE 13: INSIGHT 05 (TABLES - SCALED TO 10.5pt)
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "SECTION 03. 심층 EDA 5대 인사이트 & 파생변수 도출", "인사이트 05: 기상 요인 비선형 U자 곡선과 냉난방 부하", "15-20℃ 쾌적 구간에서 전력이 최저이며, 불쾌지수 80 이상 혹서기에 피크 전력 위험이 4.2배 급증합니다.", f"13 / {TOTAL_SLIDES}")

    s13_stat_h = ["통계 분석 항목", "정량적 수치", "쉬운 설명 (도메인 해석)"]
    s13_stat_r = [
        ("기온-전력 U자 곡선", "15-20℃ 최저 (공조 부하 최소)\n(0℃ 이하 상승, 25℃ 이상 급상승)", "저온 난방기 및 고온 대형 냉방기 가동에\n따른 전형적 비선형 U자 곡선"),
        ("혹서기 피크 (DI>=80)", "불쾌지수 80 이상 구간에서\n175-208 kW 최대 피크 집중 (4.2배)", "기온과 습도가 동반 상승하는 혹서기에\n공장 냉방 공조 풀가동"),
        ("습도 복합 부하", "습도 80% 이상 시 냉각 부하 가중", "동일 고온이라도 습도에 따라\n냉각탑 및 제습 공조 추가 가동")
    ]
    add_table(s13, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4), s13_stat_h, s13_stat_r, [Inches(1.4), Inches(1.6), Inches(2.6)], font_size=Pt(10.5))

    s13_feat_h = ["생성 파생변수", "산출 로직", "시각화 도출 근거 및 모델링 기대 효과"]
    s13_feat_r = [
        ("냉방도일_CDD", "max(기온 - 22, 0)", "22℃ 초과 고온 냉방 부하 선형 분리"),
        ("난방도일_HDD", "max(15 - 기온, 0)", "15℃ 미만 저온 난방 부하 선형 분리"),
        ("불쾌지수_DI", "0.81T + 0.01RH(0.99T-14.3) + 46.3", "온·습도 결합 복합 체감 공조 부하 수치화"),
        ("혹서기_냉방경보", "(불쾌지수_DI >= 80).astype(int)", "180 kW 이상 최대 피크 위험 집중 학습 지원")
    ]
    add_table(s13, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5), s13_feat_h, s13_feat_r, [Inches(1.3), Inches(1.6), Inches(2.7)], header_bg=COLOR_BLUE, font_size=Pt(10.5))

    add_image_panel(
        s13, FIGURES_DIR / "fe_05_weather_cooling_heating.png",
        Inches(6.6), Inches(1.9), Inches(5.933), Inches(5.1),
        caption="[시각화 05] 기온-전력 비선형 2차 회귀 U자 곡선(좌) 및 불쾌지수 구간별 전력 피크 밀도(우)"
    )

    # -------------------------------------------------------------
    # SLIDE 14: 31 FEATURES CATALOG (MASTER TABLE - SCALED TO 11pt)
    # -------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "SECTION 04. 파생변수 종합 명세 (Feature Catalog)", "5대 도메인 범주별 31개 파생변수 풀 종합 정리", "시각화 인사이트를 통해 물리적·도메인적 타당성이 검증된 변수들로만 체계화했습니다.", f"14 / {TOTAL_SLIDES}")

    cat_headers = ["도메인 범주", "변수수", "생성 파생변수 목록", "핵심 도메인 가치 및 모델링 예측 역할"]
    cat_rows = [
        ("1. 시간 및 캘린더", "9종", "시간_sin/cos, 월_sin/cos, 주말여부, 점심시간여부,\n조업시간대(5분할), 공휴일여부, 영업일여부", "23-00시 순환성 보존, 12시 43kW 급감 분리,\n주말 조업 중단(48kW) 및 평일 정상 조업 분기"),
        ("2. 조업 스케줄·패턴", "4종", "조업집중시간여부, 월요일기동시간여부,\n주차(week), 분기(quarter)", "월-금 08-17시 고부하 마스킹, 월요일 아침 설비 예열\n피크(169kW) 포착, 연간·분기별 거시 조업 흐름 반영"),
        ("3. 시계열 지연·롤링", "9종", "전력_lag_1/2/3/24/168h, rolling_mean_3/24h,\nrolling_std_3/24h, 전력_diff_1h", "단기 관성(Lag 1h r=0.90) 및 7일 조업주기(Lag 168h r=0.74),\n가동 전환 시 변동성 감지 및 급상승 모멘텀 포착"),
        ("4. 생산 및 설비 운영", "5개", "가동상태여부, 설비기동여부, 인당생산량,\n생산량_lag1, 생산량_변화량", "이봉분포(대기 23kW vs 가동 135kW) 모드 분리,\n라인 재가동 시 기동 부하(118.8kW) 및 조업 밀도 반영"),
        ("5. 기상 및 냉난방", "6개", "냉방도일(CDD), 난방도일(HDD), 불쾌지수(DI),\n혹서기_냉방경보, 체감온도, 기온_변화량", "15-20℃ 비선형 U자 공조 부하 선형 분리,\n혹서기(DI 80 이상) 180 kW 이상 최대 피크 위험 경보")
    ]
    add_table(s14, Inches(0.8), Inches(1.9), Inches(11.733), Inches(5.1), cat_headers, cat_rows, [Inches(1.8), Inches(0.8), Inches(4.3), Inches(4.833)], font_size=Pt(11))

    # -------------------------------------------------------------
    # SLIDE 15: SECTION 04 DIVIDER
    # -------------------------------------------------------------
    s15 = prs.slides.add_slide(blank_layout)
    add_section_divider(
        s15, "SECTION 04 / 04",
        "머신러닝 모델링 검증 전략 및 향후 로드맵",
        "미래 데이터 누수를 원천 차단하는 TimeSeriesSplit 4-Fold 교차검증 체계와 단계별 실행 계획을 공유합니다.",
        f"15 / {TOTAL_SLIDES}"
    )

    # -------------------------------------------------------------
    # SLIDE 16: SECTION 04 CONTENT 1 (SCALED TO 11.5pt / 15pt)
    # -------------------------------------------------------------
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "SECTION 04. 머신러닝 모델링 및 검증 전략", "TimeSeriesSplit 4-Fold 교차검증 및 다차원 평가 체계", "시계열 순서를 보존하여 미래 데이터 누수(Data Leakage)를 원천 차단하고 피크 적중률을 집중 평가합니다.", f"16 / {TOTAL_SLIDES}")

    add_card_box(
        s16, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.1),
        "🛡️ TimeSeriesSplit 4-Fold 전진 교차검증 체계",
        [
            ("• 데이터 누수 차단:", "무작위 셔플 시 미래 정보가 유출되므로 시간 순서를 엄격히 준수"),
            ("• [Fold 1] 간절기 전이:", "학습 1-3월 (2,160h) -> 검증 4월 (720h) | 동절기 후 봄철 전이 예측"),
            ("• [Fold 2] 냉방 진입기:", "학습 1-4월 (2,880h) -> 검증 5월 (744h) | 초여름 냉방 부하 초기 검증"),
            ("• [Fold 3] 하절기/이상치:", "학습 1-5월 (3,624h) -> 검증 6-7월 (1,464h) | 7월 복원 시간 검증"),
            ("• [Fold 4] 혹서기 피크:", "학습 1-7월 (5,088h) -> 검증 8-9월 (1,080h) | 연중 최대 혹서기 피크 종합 평가"),
            ("• 검증 신뢰도 효과:", "과거 데이터를 점진적으로 누적하며 항상 미확인 미래 구간을 정밀 평가")
        ],
        title_color=COLOR_NAVY, title_size=Pt(15), body_size=Pt(11.5)
    )

    add_card_box(
        s16, Inches(6.8), Inches(1.9), Inches(5.733), Inches(5.1),
        "🎯 평가 지표 체계 & 3대 베이스라인 알고리즘",
        [
            ("• RMSE (최우선 최적화):", "피크 부하 구간의 큰 오차에 높은 페널티를 부여하는 핵심 지표"),
            ("• MAE (직관적 오차):", "실제 전력과 예측 전력 간의 평균 절대 오차 (단위: kW)"),
            ("• MAPE (백분율 오차):", "전체 전력 대비 백분율 오차율 (공장 운영진 보고용, 10% 이내 목표)"),
            ("• 피크 적중률 (Recall):", "180 kW 이상 최대 피크 위험 시간대 사전 경보 성공률 (y >= 180kW)"),
            ("• LightGBM:", "대용량 테이블 시계열 최고 속도, 비선형 변수 상호작용 학습 우수"),
            ("• XGBoost:", "강력한 정규화 페널티(L1/L2)로 이상치 및 결측 구간 안정적 수렴"),
            ("• CatBoost:", "조업시간대, 요일 등 범주형 변수를 타깃 인코딩 손실 없이 최적 처리"),
            ("• 앙상블 블렌딩:", "3대 모델 가중치 결합으로 단일 모델의 극단적 오차 상호 보완")
        ],
        bg_color=COLOR_LIGHT_BLUE, border_color=COLOR_BORDER_BLUE, title_color=COLOR_BLUE, title_size=Pt(15), body_size=Pt(11.0)
    )

    # -------------------------------------------------------------
    # SLIDE 17: SECTION 04 CONTENT 2 (SCALED TO 11pt / 14pt)
    # -------------------------------------------------------------
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "SECTION 04. 향후 단계별 진행 로드맵", "기획 완료 후 실제 구현 및 모델링 4단계 타임라인", "팀원 회의 피드백을 수렴하여 파생변수 생성부터 모델 벤치마킹까지 단계별로 수행합니다.", f"17 / {TOTAL_SLIDES}")

    phase_cards = [
        ("Phase 1", "전처리 및 파생변수 기획 [완료]", [
            ("• 수행 내용:", "6,168시간 결측치 0개 및 시간 이상치 완전 정제, 5대 심층 EDA 플롯 완료"),
            ("• 주요 산출물:", "reports/feature_engineering_plan.md, figures/fe_0*.png, 발표용 보고서(PDF/PPTX)"),
            ("• 현재 상태:", "GitHub kjh 브랜치 공유 및 팀 피드백 수렴 단계")
        ], Inches(1.9), Inches(1.15), COLOR_LIGHT_BLUE, COLOR_BLUE),
        ("Phase 2", "파생변수 파이프라인 구현 [차기 진행]", [
            ("• 수행 내용:", "팀 회의 피드백 기반 31개 파생변수 우선순위 확정 및 클래스 기반 파이프라인 구축"),
            ("• 주요 산출물:", "src/features.py 모듈 작성, notebooks/kjh/03_feature_engineering_eda.ipynb"),
            ("• 목표 일정:", "내일 회의 직후 2단계 모듈 구현 착수")
        ], Inches(3.18), Inches(1.15), COLOR_CARD_BG, COLOR_CARD_BORDER),
        ("Phase 3", "머신러닝 베이스라인 구축 [대기]", [
            ("• 수행 내용:", "TimeSeriesSplit 4-Fold 교차검증 연결, LightGBM/XGBoost/CatBoost 성능 벤치마킹"),
            ("• 주요 산출물:", "src/models/baseline.py, reports/baseline_performance.csv (RMSE/MAE 비교표)"),
            ("• 검증 목표:", "원본 피처 대비 31개 파생변수 투입 후 예측 오차 감소율 정량 검증")
        ], Inches(4.46), Inches(1.15), COLOR_CARD_BG, COLOR_CARD_BORDER),
        ("Phase 4", "피크 분석 및 모델 고도화 [대기]", [
            ("• 수행 내용:", "SHAP 및 Feature Importance 분석, 180 kW 이상 최대 피크 발생 조건 역추적 규명"),
            ("• 주요 산출물:", "reports/peak_risk_analysis.md, final_submission_pipeline.py 최종 제출 파이프라인"),
            ("• 최종 목표:", "KAMP 경진대회 제출용 앙상블 모델 완성 및 피크 부하 사전 경보 시스템 수립")
        ], Inches(5.74), Inches(1.26), COLOR_CARD_BG, COLOR_CARD_BORDER)
    ]

    for p_id, p_title, p_items, p_top, p_h, p_bg, p_border in phase_cards:
        add_card_box(s17, Inches(0.8), p_top, Inches(11.733), p_h, f"[{p_id}] {p_title}", p_items, bg_color=p_bg, border_color=p_border, title_color=COLOR_NAVY if p_bg==COLOR_CARD_BG else COLOR_BLUE, title_size=Pt(14), body_size=Pt(11))

    # -------------------------------------------------------------
    # SLIDE 18: CLOSING & Q&A (SCALED TO 11pt / 14pt)
    # -------------------------------------------------------------
    s18 = prs.slides.add_slide(blank_layout)
    top_bar18 = s18.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    top_bar18.fill.solid()
    top_bar18.fill.fore_color.rgb = COLOR_BLUE
    top_bar18.line.fill.background()

    tb_end = s18.shapes.add_textbox(Inches(1.2), Inches(1.65), Inches(11.0), Inches(1.4))
    tf_end = tb_end.text_frame
    p_end1 = tf_end.paragraphs[0]
    p_end1.text = "경청해 주셔서 감사합니다"
    p_end1.font.name = FONT_FAMILY
    p_end1.font.size = Pt(38)
    p_end1.font.bold = True
    p_end1.font.color.rgb = COLOR_NAVY
    p_end1.space_after = Pt(6)

    p_end2 = tf_end.add_paragraph()
    p_end2.text = "Q & A 및 팀원 피드백 논의"
    p_end2.font.name = FONT_FAMILY
    p_end2.font.size = Pt(26)
    p_end2.font.bold = True
    p_end2.font.color.rgb = COLOR_BLUE

    qa_cards = [
        ("안건 1. 파생변수 우선순위 확정", [
            ("• 논의 내용:", "31개 후보 변수 중 1차 베이스라인 모델에 우선 투입할 핵심 피처 선정 및 추가 아이디어"),
            ("• 목표 산출물:", "우선 투입 파생변수 목록 확정 및 제외 변수 합의")
        ], Inches(3.35), Inches(0.88)),
        ("안건 2. 지연 피처(Lag) 윈도우 합의", [
            ("• 논의 내용:", "Lag 168h(1주일) 및 Lag 24h(전일) 생성 시 발생하는 초기 결측 구간 처리 방식 확정"),
            ("• 목표 산출물:", "초기 결측 행 마스킹 또는 제거 기준 합의")
        ], Inches(4.33), Inches(0.88)),
        ("안건 3. 평가 지표 가중치 결정", [
            ("• 논의 내용:", "전체 평균 오차(MAE/RMSE)와 최대 피크 적중률(Peak Recall) 중 최우선 최적화 지표 확정"),
            ("• 목표 산출물:", "모델 튜닝 목적함수 및 손실 가중치 합의")
        ], Inches(5.31), Inches(0.88)),
        ("안건 4. 모델링 역할 분담", [
            ("• 논의 내용:", "LightGBM, XGBoost, CatBoost 모델별 파라미터 튜닝 및 Feature Importance 분석 파트너링"),
            ("• 목표 산출물:", "팀원별 머신러닝 알고리즘 R&R 분담 확정")
        ], Inches(6.29), Inches(0.88))
    ]

    for q_title, q_items, q_top, q_h in qa_cards:
        add_card_box(s18, Inches(1.2), q_top, Inches(10.933), q_h, q_title, q_items, bg_color=COLOR_LIGHT_BLUE, border_color=COLOR_BORDER_BLUE, title_color=COLOR_BLUE, title_size=Pt(13.5), body_size=Pt(11))

    # Save PPTX
    prs.save(str(PPTX_PATH))
    print(f"PPTX successfully created at: {PPTX_PATH}")

def convert_pptx_to_pdf():
    import comtypes.client
    
    abs_pptx = str(PPTX_PATH.resolve())
    abs_pdf = str(PDF_PATH.resolve())
    
    print("Converting PPTX to PDF via PowerPoint COM...")
    powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
    try:
        deck = powerpoint.Presentations.Open(abs_pptx, WithWindow=False)
        deck.SaveAs(abs_pdf, 32)
        deck.Close()
        print(f"PDF successfully created at: {abs_pdf}")
    finally:
        powerpoint.Quit()

if __name__ == "__main__":
    create_presentation()
    convert_pptx_to_pdf()
