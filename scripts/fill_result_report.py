"""Fill the supplied HWPX competition form without changing its package format."""

from copy import deepcopy
from pathlib import Path
import zipfile

from lxml import etree
import pandas as pd
from PIL import Image
from report_chapters import build_chapters


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = next((ROOT / "docs").glob("*결과보고서*.hwpx"))
OUTPUT = ROOT / "report" / "OKM_경진대회_결과보고서_작성본.hwpx"
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


def paragraph(template, content):
    p = deepcopy(template)
    set_text(p, content)
    for run in list(p.findall(f"{{{HP}}}run"))[1:]:
        if not any((node.text or "").strip() for node in run.findall(f"{{{HP}}}t")):
            p.remove(run)
    if content.startswith("- "):
        p.find(f"{{{HP}}}run").set("charPrIDRef", "37")
    return p


def picture_paragraph(template, original_pic, image_id, path, number):
    """Reuse the form's picture element with a new embedded PNG and flow layout."""
    p = deepcopy(template)
    for child in list(p):
        p.remove(child)
    run = etree.SubElement(p, f"{{{HP}}}run", charPrIDRef="37")
    pic = deepcopy(original_pic)
    pic.set("id", str(2000000000 + number))
    pic.set("instid", str(2000001000 + number))
    pic.set("zOrder", str(20 + number))
    pic.find(f"{{{HC}}}img").set("binaryItemIDRef", image_id)
    with Image.open(path) as im:
        pixels_w, pixels_h = im.size
    original_w, original_h = pixels_w * 75, pixels_h * 75
    width = 42500 if path.name == "03_dashboard.png" else 45000
    height = round(width * pixels_h / pixels_w)
    if height > 43500:
        height = 43500
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


def font_header(original):
    header = etree.fromstring(original)
    properties = next(x for x in header.iter() if etree.QName(x).localname == "charProperties")
    base = next(x for x in properties if x.get("id") == "14")
    for ident, height in [("37", "1200"), ("38", "1000")]:
        style = deepcopy(base)
        style.set("id", ident)
        style.set("height", height)
        properties.append(style)
    properties.set("itemCnt", str(len(properties)))
    return etree.tostring(header, encoding="UTF-8", xml_declaration=True)


def fill():
    scores = pd.read_csv(ROOT / "results/base9_comparison.csv", encoding="utf-8-sig").set_index("model")
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
        headings = [1, 8, 16, 24, 32, 40]
        new_children = [cover]
        figures = [
            (0, 2, "08_eda_overview.png", "그림 1. 일별 평균 전력과 생산량의 동시 변화"),
            (0, 6, "09_eda_quality.png", "그림 2. 원본 결측과 셧다운 전후 전력 진단"),
            (0, 10, "10_eda_heatmap.png", "그림 3. 요일·시간별 평균 전력 열지도"),
            (0, 10, "11_eda_operating_modes.png", "그림 4. 생산량과 전력 부하·대기 부하 분포"),
            (0, 10, "04_data_profile.png", "그림 5. 평일과 주말의 시간별 평균 전력 패턴"),
            (1, 7, "05_grid_search.png", "그림 6. 신경망 단계별 탐색과 트리 전체 조합의 검증 결과"),
            (1, 12, "12_grid_selection_table.png", "표 1. 모델별 탐색 후보·fold 결과와 최종 선택"),
            (1, 15, "13_test_metrics_table.png", "표 2. 동일 Test 703시간의 다섯 모델 성능"),
            (1, 15, "01_model_comparison.png", "그림 7. 다섯 후보의 RMSE와 피크 경보 F1"),
            (2, 4, "06_feature_interaction.png", "그림 8. 변수 묶음 중요도와 생산량·조업 시간의 교차 분석"),
            (2, 8, "07_error_conditions.png", "그림 9. 시간대별 오경보 및 미탐지 집중 구간"),
            (2, 7, "02_peak_timeline.png", "그림 10. 피크 집중 기간의 실측·예측·경보"),
            (3, 5, "03_dashboard.png", "그림 11. Test 예측을 표시한 Dash 대시보드 화면"),
        ]
        def add_figures(chapter, line_index):
            for number, (target, target_index, filename, caption) in enumerate(figures, start=1):
                if target != chapter or target_index != line_index:
                    continue
                path = ROOT / "report/figures" / filename
                if not path.exists():
                    raise FileNotFoundError(f"Generate report figures first: {path}")
                caption_p = paragraph(bullet, caption)
                caption_p.find(f"{{{HP}}}run").set("charPrIDRef", "38")
                caption_p.set("pageBreak", "1")
                new_children.append(caption_p)
                new_children.append(picture_paragraph(children[position + 2], original_pic,
                                                       f"report_image{number}", path, number))
        for n, position in enumerate(headings):
            heading = children[position]
            heading.set("pageBreak", "1")
            new_children.append(heading)
            bullet = children[position + 1]
            for line_index, line in enumerate(chapters[n]):
                new_children.append(paragraph(bullet, line))
                add_figures(n, line_index)
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
                    payload = font_header(source.read(item.filename))
                elif item.filename == "Contents/content.hpf":
                    payload = etree.tostring(package, encoding="UTF-8", xml_declaration=True)
                elif item.filename == "Preview/PrvText.txt":
                    payload = "\n".join(text_of(x) for x in new_children if text_of(x)).encode("utf-8")
                else:
                    payload = source.read(item.filename)
                target.writestr(item, payload)
            for _, _, filename, _ in figures:
                target.write(ROOT / "report/figures" / filename, f"BinData/{filename}")
        review = ["# OKM 제조공정 전력 피크 예측: 심사기준별 결과보고서", ""]
        for number, (position, chapter) in enumerate(zip(headings, chapters), start=1):
            review.extend([f"## {text_of(children[position])}", ""])
            for line_index, line in enumerate(chapter):
                review.extend([line, ""])
                for target, target_index, filename, caption in figures:
                    if target == number - 1 and target_index == line_index:
                        review.extend([caption, "", f"![{caption}](figures/{filename})", ""])
        (OUTPUT.parent / "심사기준별_상세내용.md").write_text("\n".join(review), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        finished = etree.fromstring(check.read("Contents/section0.xml"))
        plain = " ".join(finished.xpath(".//hp:t/text()", namespaces=NS))
        assert "8.67 kW" in plain and "108개 조합" in plain and "작성 요령" not in plain
        assert len([x for x in finished.iter() if etree.QName(x).localname == "pic"]) == 14
        assert not any(etree.QName(x).localname == "linesegarray" for x in finished.iter())
    print(OUTPUT)


if __name__ == "__main__":
    fill()
