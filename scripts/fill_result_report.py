"""Fill the supplied HWPX competition form without changing its package format."""

from copy import deepcopy
import json
from pathlib import Path
import zipfile

from lxml import etree
import pandas as pd
from PIL import Image
from report_chapters import build_chapters


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = next((ROOT / "docs" / "templates").glob("*결과보고서*.hwpx"))
OUTPUT = ROOT / "docs" / "report" / "OKM_경진대회_결과보고서_작성본.hwpx"
HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
NS = {"hp": HP}
HH = "http://www.hancom.co.kr/hwpml/2011/head"
HC = "http://www.hancom.co.kr/hwpml/2011/core"
OPF = "http://www.idpf.org/2007/opf/"


def text_of(p):
    return "".join(p.xpath(".//hp:t/text()", namespaces=NS)).strip()


def set_text(p, content):
    ts = p.xpath("./hp:run/hp:t", namespaces=NS)
    if not ts:
        run = p.find(f"{{{HP}}}run")
        if run is None:
            run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef="0")
        ts = [etree.SubElement(run, f"{{{HP}}}t")]
    ts[0].text = content
    for other in ts[1:]:
        other.text = ""


def paragraph(template, content, *, prefix="", prefix_char=None, text_char="8"):
    p = deepcopy(template)
    for child in list(p):
        if etree.QName(child).localname in {"run", "linesegarray"}:
            p.remove(child)
    if prefix:
        run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef=prefix_char)
        etree.SubElement(run, f"{{{HP}}}t").text = prefix
    run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef=text_char)
    etree.SubElement(run, f"{{{HP}}}t").text = content
    return p


def native_table_paragraph(template_table, headers, rows, widths, table_id):
    """Build an editable HWPX table using the supplied form's native cell styles."""
    total_width = 46202
    minimum_height = 1900
    table = deepcopy(template_table)
    for child in list(table):
        if etree.QName(child).localname == "tr":
            table.remove(child)
    table.set("id", str(2100000000 + table_id))
    table.set("zOrder", str(60 + table_id))
    table.set("rowCnt", str(len(rows) + 1))
    table.set("colCnt", str(len(headers)))
    table.set("repeatHeader", "1")
    table.set("pageBreak", "CELL")
    size = table.find(f"{{{HP}}}sz")
    size.set("width", str(total_width))
    size.set("height", str(minimum_height * (len(rows) + 1)))

    source_row = next(node for node in template_table if etree.QName(node).localname == "tr")
    source_cell = next(node for node in source_row if etree.QName(node).localname == "tc")
    column_widths = [round(total_width * value) for value in widths]
    column_widths[-1] += total_width - sum(column_widths)

    for row_index, values in enumerate([headers, *rows]):
        tr = etree.Element(f"{{{HP}}}tr")
        for column_index, (value, column_width) in enumerate(zip(values, column_widths)):
            cell = deepcopy(source_cell)
            cell.set("header", "1" if row_index == 0 else "0")
            cell.set("borderFillIDRef", "6" if row_index == 0 or column_index == 0 else "8")
            address = next(node for node in cell if etree.QName(node).localname == "cellAddr")
            address.set("colAddr", str(column_index))
            address.set("rowAddr", str(row_index))
            span = next(node for node in cell if etree.QName(node).localname == "cellSpan")
            span.set("colSpan", "1")
            span.set("rowSpan", "1")
            cell_size = next(node for node in cell if etree.QName(node).localname == "cellSz")
            cell_size.set("width", str(column_width))
            cell_size.set("height", str(minimum_height))
            cell_margin = next(node for node in cell
                               if etree.QName(node).localname == "cellMargin")
            cell_margin.set("left", "260")
            cell_margin.set("right", "260")
            cell_margin.set("top", "180")
            cell_margin.set("bottom", "180")
            cell_paragraph = next(node for node in cell.iter()
                                  if etree.QName(node).localname == "p")
            centered = row_index == 0 or column_index == 0 or len(headers) == 7
            cell_paragraph.set("paraPrIDRef", "47" if centered else "46")
            cell_paragraph.set("pageBreak", "0")
            for child in list(cell_paragraph):
                if etree.QName(child).localname in {"run", "linesegarray"}:
                    cell_paragraph.remove(child)
            run = etree.SubElement(
                cell_paragraph, f"{{{HP}}}run",
                charPrIDRef="21" if row_index == 0 or column_index == 0 else "10",
            )
            etree.SubElement(run, f"{{{HP}}}t").text = str(value)
            tr.append(cell)
        table.append(tr)

    p = etree.Element(
        f"{{{HP}}}p", id=str(2100010000 + table_id), paraPrIDRef="0",
        styleIDRef="0", pageBreak="0", columnBreak="0", merged="0",
    )
    run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef="10")
    run.append(table)
    return p


def report_tables(scores):
    grid = pd.read_csv(ROOT / "docs/report/evidence/grid_search_summary.csv", encoding="utf-8-sig")
    specs = []
    table_number = 1
    for model in ("lstm", "tcn"):
        stages = grid.loc[grid.model.eq(model)].sort_values("stage")
        selected = stages.iloc[-1]
        params = json.loads(selected.selected_params)
        folds = pd.read_csv(ROOT / f"neural/results/grid_search/{model}.csv")
        if model == "lstm":
            rows = [
                ["탐색 방식", "4단계 순차 탐색 · 각 단계 3개 시간순 fold"],
                ["입력 길이", "24 · 48 · 72 · 168시간"],
                ["모델 구조", "hidden_units: 32 · 64 · 128 / num_layers: 1 · 2"],
                ["최적화", "learning_rate: 0.0001 · 0.0003 · 0.001 / dropout: 0 · 0.1 · 0.2 · 0.3"],
                ["배치 크기", "16 · 32 · 64"],
                ["탐색 규모", f"{folds.config_id.nunique()}개 고유 후보 · {len(folds)}개 fold 결과"],
                ["단계별 RMSE", " → ".join(f"{value:.2f}" for value in stages.validation_rmse) + " kW"],
                ["최종 선택", f"{params['lookback']}시간 / {params['hidden_units']}유닛 / {params['num_layers']}층 / lr {params['learning_rate']} / dropout {params['dropout']} / batch {params['batch_size']}"],
                ["검증 성능", f"RMSE {selected.validation_rmse:.2f} kW / 피크 MAE {selected.validation_peak_mae:.2f} kW"],
            ]
            position, caption = (1, 5), "표 1. LSTM Grid Search 파라미터 후보값과 최종 선택"
        else:
            rows = [
                ["탐색 방식", "4단계 순차 탐색 · 각 단계 3개 시간순 fold"],
                ["입력 길이", "24 · 48 · 72 · 168시간"],
                ["모델 구조", "filters: 32 · 64 · 128 / kernel: 2 · 3 · 5 / dilation: auto · auto+1"],
                ["최적화", "learning_rate: 0.0001 · 0.0003 · 0.001 / dropout: 0 · 0.1 · 0.2 · 0.3"],
                ["배치 크기", "16 · 32 · 64"],
                ["탐색 규모", f"{folds.config_id.nunique()}개 고유 후보 · {len(folds)}개 fold 결과"],
                ["단계별 RMSE", " → ".join(f"{value:.2f}" for value in stages.validation_rmse) + " kW"],
                ["최종 선택", f"{params['lookback']}시간 / {params['filters']}필터 / kernel {params['kernel_size']} / {params['dilations']} / lr {params['learning_rate']} / dropout {params['dropout']} / batch {params['batch_size']}"],
                ["검증 성능", f"RMSE {selected.validation_rmse:.2f} kW / 피크 MAE {selected.validation_peak_mae:.2f} kW"],
            ]
            position, caption = (1, 7), "표 2. TCN Grid Search 파라미터 후보값과 최종 선택"
        specs.append((*position, caption, ["항목", "내용"], rows, [.24, .76], table_number))
        table_number += 1

    for model, caption, position in (
        ("xgboost", "표 3. XGBoost Grid Search 파라미터 후보값과 최종 선택", (1, 10)),
        ("lightgbm", "표 4. LightGBM Grid Search 파라미터 후보값과 최종 선택", (1, 12)),
    ):
        selected = grid.loc[grid.model.eq(model)].iloc[-1]
        params = json.loads(selected.selected_params)
        if model == "xgboost":
            rows = [["max_depth", "3 · 5 · 7"], ["learning_rate", "0.01 · 0.05 · 0.1"],
                    ["n_estimators", "200 · 500 · 1000"], ["min_child_weight", "1 · 5"],
                    ["colsample_bytree", "0.8 · 1.0"],
                    ["최종 선택", f"depth {params['max_depth']} / lr {params['learning_rate']} / trees {params['n_estimators']} / min_child {params['min_child_weight']} / colsample {params['colsample_bytree']}"]]
        else:
            rows = [["num_leaves", "7 · 15 · 31"], ["learning_rate", "0.01 · 0.05 · 0.1"],
                    ["n_estimators", "200 · 500 · 1000"], ["min_child_samples", "20 · 50"],
                    ["colsample_bytree", "0.8 · 1.0"],
                    ["최종 선택", f"leaves {params['num_leaves']} / lr {params['learning_rate']} / trees {params['n_estimators']} / min_child {params['min_child_samples']} / colsample {params['colsample_bytree']}"]]
        rows = [["탐색 방식", "전체 조합 Grid Search · 3개 시간순 fold"], *rows[:5],
                ["탐색 규모", f"{int(selected.completed_candidates)}개 후보 · {int(selected.fold_results)}개 fold 결과"],
                ["선택 규칙", f"최저 RMSE {selected.lowest_rmse:.2f} kW의 2% 이내 {int(selected.rmse_2pct_candidates)}개 후보 중 피크 MAE 최저"],
                rows[5], ["검증 성능", f"RMSE {selected.validation_rmse:.2f} kW / 피크 MAE {selected.validation_peak_mae:.2f} kW"]]
        specs.append((*position, caption, ["항목", "내용"], rows, [.24, .76], table_number))
        table_number += 1

    metric_rows = []
    labels = {"lstm": "LSTM", "tcn": "TCN", "ensemble": "가중 앙상블",
              "xgboost": "XGBoost", "lightgbm": "LightGBM"}
    for model in ("lstm", "tcn", "ensemble", "xgboost", "lightgbm"):
        row = scores.loc[model]
        metric_rows.append([labels[model], f"{row.rmse:.2f}", f"{row.mae:.2f}",
                            f"{row.alert_f1:.3f}", f"{row.alert_recall:.1%}",
                            str(int(row.fn)), str(int(row.fp))])
    specs.append((1, 15, "표 5. 동일 Test 703시간의 다섯 모델 성능",
                  ["모델", "RMSE", "MAE", "피크 F1", "재현율", "미탐지", "오경보"], metric_rows,
                  [.20, .13, .13, .15, .15, .12, .12], table_number))
    table_number += 1
    reproduce_rows = [
        ["환경 설치", "python -m pip install -r requirements.txt", "requirements.txt", "통합 실행 환경"],
        ["전처리 검증", "python scripts/verify_cleaned_data.py", "data/*.csv", "6,168행·22열 검증"],
        ["트리 학습", "python scripts/train_tree_models.py --models lightgbm xgboost --n-jobs 8", "data/·model_search/", "results/tree_models/"],
        ["5개 모델 비교", "python scripts/build_model_comparison.py", "neural/results/·results/tree_models/", "results/model_comparison.csv"],
        ["대시보드", "python dashboard/app.py", "저장된 Test 예측", "127.0.0.1:8050"],
        ["화면 캡처", "python scripts/capture_dashboard.py", "Dash 화면", "03_dashboard.png"],
        ["제출물 재생성", "python scripts/reproduce_submission.py --capture-dashboard", "전처리·모델 결과", "근거표·도표·HWPX 검증"],
    ]
    specs.append((5, 6, "표 6. 전처리부터 제출물까지의 실행 명령·입력·산출물",
                  ["단계", "실행 명령", "입력", "주요 산출물"], reproduce_rows,
                  [.17, .39, .20, .24], table_number))
    return specs


def picture_paragraph(template, original_pic, image_id, path, number):
    """Reuse the form's picture element with a new embedded PNG and flow layout."""
    p = deepcopy(template)
    p.set("paraPrIDRef", "42")
    p.set("pageBreak", "0")
    for child in list(p):
        p.remove(child)
    run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef="0")
    pic = deepcopy(original_pic)
    pic.set("id", str(2000000000 + number))
    pic.set("instid", str(2000001000 + number))
    pic.set("zOrder", str(20 + number))
    pic.find(f"{{{HC}}}img").set("binaryItemIDRef", image_id)
    with Image.open(path) as im:
        pixels_w, pixels_h = im.size
    original_w, original_h = pixels_w * 75, pixels_h * 75
    width = 46500
    height = round(width * pixels_h / pixels_w)
    if height > 40000:
        height = 40000
        width = round(height * pixels_w / pixels_h)
    for tag in ("orgSz", "imgDim"):
        node = pic.find(f"{{{HP}}}{tag}")
        if tag == "orgSz":
            node.set("width", str(original_w)); node.set("height", str(original_h))
        else:
            node.set("dimwidth", str(original_w)); node.set("dimheight", str(original_h))
    pic.find(f"{{{HP}}}curSz").set("width", str(width))
    pic.find(f"{{{HP}}}curSz").set("height", str(height))
    pic.find(f"{{{HP}}}sz").set("width", str(width))
    pic.find(f"{{{HP}}}sz").set("height", str(height))
    rect = pic.find(f"{{{HP}}}imgRect")
    for corner, x, y in [("pt0", 0, 0), ("pt1", original_w, 0),
                         ("pt2", original_w, original_h), ("pt3", 0, original_h)]:
        node = rect.find(f"{{{HC}}}{corner}")
        node.set("x", str(x)); node.set("y", str(y))
    clip = pic.find(f"{{{HP}}}imgClip")
    clip.set("right", str(original_w)); clip.set("bottom", str(original_h))
    rotation = pic.find(f"{{{HP}}}rotationInfo")
    rotation.set("centerX", str(width // 2)); rotation.set("centerY", str(height // 2))
    scale = pic.find(f"{{{HP}}}renderingInfo").find(f"{{{HC}}}scaMatrix")
    scale.set("e1", f"{width / original_w:.6f}")
    scale.set("e5", f"{height / original_h:.6f}")
    position = pic.find(f"{{{HP}}}pos")
    position.set("treatAsChar", "1")
    position.set("flowWithText", "1")
    position.set("allowOverlap", "0")
    position.set("vertRelTo", "PARA")
    position.set("horzRelTo", "COLUMN")
    pic.find(f"{{{HP}}}shapeComment").text = f"그림 {number}: {path.name}"
    run.append(pic)
    return p


def layout_header(original):
    header = etree.fromstring(original)
    properties = next(x for x in header.iter()
                      if etree.QName(x).localname == "paraProperties")
    caption = next(x for x in properties if x.get("id") == "27")
    caption.find(f"{{{HH}}}align").set("horizontal", "CENTER")
    break_setting = next(x for x in caption
                         if etree.QName(x).localname == "breakSetting")
    break_setting.set("keepWithNext", "0")
    break_setting.set("keepLines", "1")
    for margin in (x for x in caption.iter() if etree.QName(x).localname == "margin"):
        for value in margin:
            name = etree.QName(value).localname
            value.set("value", "150" if name == "prev" else "1000" if name == "next" else "0")
    picture = deepcopy(caption)
    picture.set("id", "42")
    picture_break = next(x for x in picture if etree.QName(x).localname == "breakSetting")
    picture_break.set("keepWithNext", "1")
    for margin in (x for x in picture.iter() if etree.QName(x).localname == "margin"):
        for value in margin:
            name = etree.QName(value).localname
            value.set("value", "900" if name == "prev" else "200" if name == "next" else "0")
    properties.append(picture)
    def clone_paragraph_style(source_id, target_id, prev, next_, keep_with_next):
        source = next(x for x in properties if x.get("id") == source_id)
        style = deepcopy(source)
        style.set("id", target_id)
        setting = next(x for x in style if etree.QName(x).localname == "breakSetting")
        setting.set("keepWithNext", "1" if keep_with_next else "0")
        for margin in (x for x in style.iter() if etree.QName(x).localname == "margin"):
            for value in margin:
                name = etree.QName(value).localname
                if name == "prev":
                    value.set("value", str(prev))
                elif name == "next":
                    value.set("value", str(next_))
        properties.append(style)
    clone_paragraph_style("20", "43", 1400, 700, True)
    clone_paragraph_style("25", "44", 900, 350, True)
    clone_paragraph_style("26", "45", 0, 250, False)
    clone_paragraph_style("21", "46", 0, 0, False)
    clone_paragraph_style("21", "47", 0, 0, False)
    for style_id, alignment in (("46", "LEFT"), ("47", "CENTER")):
        style = next(x for x in properties if x.get("id") == style_id)
        style.find(f"{{{HH}}}align").set("horizontal", alignment)
        for margin in (x for x in style.iter() if etree.QName(x).localname == "margin"):
            for value in margin:
                value.set("value", "0")
    properties.set("itemCnt", str(len(properties)))
    return etree.tostring(header, encoding="UTF-8", xml_declaration=True)


def fill():
    scores = pd.read_csv(ROOT / "results/model_comparison.csv", encoding="utf-8-sig").set_index("model")
    if set(scores.index) != {"lstm", "tcn", "ensemble", "xgboost", "lightgbm"}:
        raise ValueError("Five comparable results are required")
    if not scores.seed.eq(42).all() or not scores.n_features.eq(9).all():
        raise ValueError("Result conditions changed")
    e = scores.loc["ensemble"]
    with zipfile.ZipFile(TEMPLATE) as source:
        root = etree.fromstring(source.read("Contents/section0.xml"))
        package = etree.fromstring(source.read("Contents/content.hpf"))
        children = list(root)
        original_pic = next(x for x in root.iter() if etree.QName(x).localname == "pic")
        template_table = next(
            x for x in root.iter()
            if etree.QName(x).localname == "tbl"
            and x.get("rowCnt") == "2" and x.get("colCnt") == "2"
        )
        cover = children[0]
        cover_ps = cover.xpath(".//hp:p", namespaces=NS)
        # The competition form reserves blank cells for project, team and summary.
        set_text(cover_ps[3], "OKM 제조공정 전력 피크 예측")
        set_text(cover_ps[5], "OKM [팀명 확인]")
        set_text(cover_ps[7],
                 "2021년 제조공정의 시간별 전력·생산·기상 기록과 9개 공통 입력 변수를 바탕으로 "
                 "LSTM, TCN, 가중 앙상블, XGBoost, LightGBM을 시간 순서 검증과 동일 Test 구간에서 비교했다. "
                 f"최종 앙상블은 Test 703시간에서 RMSE {e.rmse:.2f} kW, 피크 경보 F1 {e.alert_f1:.3f}, "
                 f"재현율 {e.alert_recall:.1%}를 기록했다. 본문에는 탐색·오류분석·현장 적용 절차를 제시했다.")

        chapters = build_chapters(ROOT, scores)
        tables = report_tables(scores)
        headings = [1, 8, 16, 24, 32, 40]
        new_children = [cover]
        figures = [
            (0, 2, "08_eda_overview.png", "그림 1. 일별 평균 전력과 생산량의 동시 변화"),
            (0, 6, "09_eda_quality.png", "그림 2. 원본 결측과 셧다운 전후 전력 진단"),
            (0, 10, "10_eda_heatmap.png", "그림 3. 요일·시간별 평균 전력 열지도"),
            (0, 10, "11_eda_operating_modes.png", "그림 4. 생산량과 전력 부하·대기 부하 분포"),
            (0, 10, "04_data_profile.png", "그림 5. 평일과 주말의 시간별 평균 전력 패턴"),
            (1, 7, "05_grid_search.png", "그림 6. 신경망 단계별 탐색과 트리 전체 조합의 검증 결과"),
            (1, 15, "01_model_comparison.png", "그림 7. 다섯 후보의 RMSE와 피크 경보 F1"),
            (2, 4, "06_feature_interaction.png", "그림 8. 변수 묶음 중요도와 생산량·조업 시간의 교차 분석"),
            (2, 8, "07_error_conditions.png", "그림 9. 시간대별 오경보 및 미탐지 집중 구간"),
            (2, 7, "02_peak_timeline.png", "그림 10. 피크 집중 기간의 실측·예측·경보"),
            (3, 7, "03_dashboard.png", "그림 11. 3σ 관리상·하한과 이탈을 표시한 Dash 대시보드"),
        ]
        def add_figures(chapter, line_index, caption_template, picture_template):
            for number, (target, target_index, filename, caption) in enumerate(figures, start=1):
                if target != chapter or target_index != line_index:
                    continue
                path = ROOT / "docs/report/figures" / filename
                if not path.exists():
                    raise FileNotFoundError(f"Generate report figures first: {path}")
                caption_p = paragraph(caption_template, caption, text_char="15")
                caption_p.set("pageBreak", "0")
                new_children.append(picture_paragraph(picture_template, original_pic,
                                                       f"report_image{number}", path, number))
                new_children.append(caption_p)
        def add_tables(chapter, line_index, caption_template):
            for target, target_index, caption, headers, rows, widths, table_id in tables:
                if target != chapter or target_index != line_index:
                    continue
                new_children.append(native_table_paragraph(
                    template_table, headers, rows, widths, table_id
                ))
                caption_p = paragraph(caption_template, caption, text_char="15")
                caption_p.set("pageBreak", "0")
                new_children.append(caption_p)
        for n, position in enumerate(headings):
            heading = deepcopy(children[position])
            heading.set("paraPrIDRef", "43")
            new_children.append(heading)
            block_end = headings[n + 1] if n + 1 < len(headings) else 58
            form_block = children[position + 1:block_end]
            subheading_template = next(
                item for item in form_block
                if item.get("paraPrIDRef") == "25" and text_of(item)
            )
            body_template = next(
                item for item in form_block
                if item.get("paraPrIDRef") == "26" and text_of(item).lstrip().startswith("-")
            )
            caption_template = next(
                item for item in form_block
                if item.get("paraPrIDRef") == "27" and text_of(item).lstrip().startswith("*")
            )
            for line_index, line in enumerate(chapters[n]):
                if line.startswith("◦ "):
                    item = paragraph(
                        subheading_template, line[2:], prefix=" ◦ ",
                        prefix_char="14", text_char="8")
                    item.set("paraPrIDRef", "44")
                    new_children.append(item)
                elif line.startswith("- "):
                    item = paragraph(
                        body_template, line[2:], prefix="   - ",
                        prefix_char="13", text_char="8")
                    item.set("paraPrIDRef", "45")
                    new_children.append(item)
                else:
                    item = paragraph(body_template, line, text_char="8")
                    item.set("paraPrIDRef", "45")
                    new_children.append(item)
                add_figures(n, line_index, caption_template, body_template)
                add_tables(n, line_index, caption_template)
        # Keep the form's mandatory survey section. The submission owner must
        # replace the example image with their own completion screenshot.
        new_children.extend(children[58:])
        for child in list(root):
            root.remove(child)
        for child in new_children:
            root.append(child)
        # The supplied form stores layout coordinates calculated for its short
        # placeholders. Keeping them makes the new lines paint on top of one
        # another. Hancom recalculates these optional segments on opening.
        for paragraph_node in root.xpath(".//hp:p", namespaces=NS):
            for segments in paragraph_node.findall(f"{{{HP}}}linesegarray"):
                paragraph_node.remove(segments)
        manifest = package.find(f"{{{OPF}}}manifest")
        for n, (_, _, filename, _) in enumerate(figures, start=1):
            etree.SubElement(manifest, f"{{{OPF}}}item", id=f"report_image{n}",
                             href=f"BinData/{filename}", **{"media-type": "image/png", "isEmbeded": "1"})
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUTPUT, "w") as target:
            for item in source.infolist():
                if item.filename == "Contents/section0.xml":
                    payload = etree.tostring(root, encoding="UTF-8", xml_declaration=True)
                elif item.filename == "Contents/header.xml":
                    payload = layout_header(source.read(item.filename))
                elif item.filename == "Contents/content.hpf":
                    payload = etree.tostring(package, encoding="UTF-8", xml_declaration=True)
                elif item.filename == "Preview/PrvText.txt":
                    payload = "\n".join(text_of(x) for x in new_children if text_of(x)).encode("utf-8")
                else:
                    payload = source.read(item.filename)
                target.writestr(item, payload)
            for _, _, filename, _ in figures:
                target.write(ROOT / "docs/report/figures" / filename, f"BinData/{filename}")
        review = ["# OKM 제조공정 전력 피크 예측: 심사기준별 결과보고서", ""]
        for number, (position, chapter) in enumerate(zip(headings, chapters), start=1):
            review.extend([f"## {text_of(children[position])}", ""])
            for line_index, line in enumerate(chapter):
                review.extend([line, ""])
                for target, target_index, filename, caption in figures:
                    if target == number - 1 and target_index == line_index:
                        review.extend([caption, "", f"![{caption}](figures/{filename})", ""])
                for target, target_index, caption, headers, rows, _, _ in tables:
                    if target != number - 1 or target_index != line_index:
                        continue
                    review.extend([caption, "", "| " + " | ".join(headers) + " |",
                                   "|" + "|".join(["---"] * len(headers)) + "|"])
                    review.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
                    review.append("")
        (OUTPUT.parent / "심사기준별_상세내용.md").write_text("\n".join(review), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        finished = etree.fromstring(check.read("Contents/section0.xml"))
        finished_header = etree.fromstring(check.read("Contents/header.xml"))
        plain = " ".join(finished.xpath(".//hp:t/text()", namespaces=NS))
        assert "8.67 kW" in plain and "108개 조합" in plain and "작성 요령" not in plain
        assert len([x for x in finished.iter() if etree.QName(x).localname == "pic"]) == len(figures) + 1
        assert len([x for x in finished.iter() if etree.QName(x).localname == "tbl"]) == len(tables) + 2
        assert not any(etree.QName(x).localname == "linesegarray" for x in finished.iter())
        assert sum(x.get("pageBreak") == "1" for x in list(finished)) == 1
        char_properties = next(
            x for x in finished_header.iter()
            if etree.QName(x).localname == "charProperties")
        assert len(char_properties) == 37
        for p in finished.xpath("./hp:p", namespaces=NS):
            text = text_of(p)
            char_refs = {run.get("charPrIDRef") for run in p.findall(f"{{{HP}}}run")}
            if text.startswith("◦ "):
                assert p.get("paraPrIDRef") == "44" and {"14", "8"} <= char_refs
            elif text.lstrip().startswith("-") and p.get("paraPrIDRef") not in {"20"}:
                assert p.get("paraPrIDRef") == "45" and {"13", "8"} <= char_refs
            elif text.startswith(("그림 ", "표 ")):
                assert p.get("paraPrIDRef") == "27" and char_refs == {"15"}
                assert p.get("pageBreak") == "0"
        for pic in (x for x in finished.iter() if etree.QName(x).localname == "pic"):
            image = pic.find(f"{{{HC}}}img")
            if image.get("binaryItemIDRef", "").startswith("report_image"):
                size = pic.find(f"{{{HP}}}curSz")
                assert int(size.get("height")) <= 40000
    print(OUTPUT)


if __name__ == "__main__":
    fill()
