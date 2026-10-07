"""Fill the supplied HWPX competition form without changing its package format."""

from copy import deepcopy
from pathlib import Path
import zipfile

from lxml import etree
import pandas as pd
from PIL import Image


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
    width = 45000 if number != 3 else 42500
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
                 "2021년 제조공정의 시간별 전력·생산·기상 기록에서 9개 기본 변수만 사용하였다. "
                 "LSTM, TCN, 가중 앙상블, XGBoost, LightGBM을 동일한 검증·Test 시점에서 비교했다. "
                 f"최종 앙상블은 Test 703시간에서 RMSE {e.rmse:.2f} kW, 피크 경보 F1 {e.alert_f1:.3f}, "
                 f"재현율 {e.alert_recall:.1%}를 기록했다. 신규 피처 엔지니어링은 수행하지 않았다.")

        chapters = [
            [
                "◦ 분석 대상과 문제 정의",
                "- 2021년 1월 1일~9월 14일 시간별 제조공정 자료 6,168개 관측치를 대상으로 다음 시점의 전력 평균값(kW)을 예측한다. 실측 전력이 177 kW 이상인 시간을 피크로 정의한다.",
                "- 기본 입력 9개: 과거 전력_평균_실수, 생산량, 기온, 풍속, 습도, 강수량, 시간_sin, 시간_cos, 주말여부. 마지막 3개는 기존 LHS 기본 입력 정의에 따라 날짜·시간에서 생성하며, 추가 피처 엔지니어링 실험은 하지 않았다.",
                "- 전력과 생산량은 설비 가동 강도, 기상 변수는 외부 운전 조건, 시간·주말 변수는 운영 주기를 나타낸다. 이 정보로 피크 시점과 전력 규모를 함께 파악한다.",
                "◦ 정제 및 누수 방지",
                "- KJH 전처리 코드를 사용한 cleaned CSV만 학습에 사용한다. 원본의 시간 이상 48건, 작업자 결측 17건, 풍속 결측 3건, 강수량 결측 1건, 셧다운 17시간을 처리한다. 원본 증강 데이터는 학습하지 않는다.",
                "- 시간 순서를 유지해 3개 확장 검증 fold를 만들고 마지막 구간을 Test로 보관했다. fold 평가 시작 전 14일은 학습에서 제외했다. 검증과 Test 구간의 정답으로 입력을 보정하지 않았다.",
            ],
            [
                "◦ 공통 평가 설계",
                "- 모든 후보는 seed 42, cleaned 데이터의 동일한 9개 변수, 동일한 시간 검증 경계와 Test 시점을 사용한다. Test는 2021년 8월 16일~9월 14일의 유효한 703시간이다. 평가 지표는 RMSE·MAE·R²와 177 kW 피크 경보 F1·재현율이다.",
                "- LSTM은 과거 168시간, TCN은 24시간을 입력한다. 두 예측을 검증 MSE의 역수로 가중해 앙상블을 구성하며 가중치는 LSTM 0.444, TCN 0.556이다. XGBoost와 LightGBM은 과거 168시간 창을 사용한다. 입력 길이가 달라 완전히 동일한 구조의 실험은 아니지만 평가 시각과 원천 변수는 같다.",
                "- LSTM·TCN의 기존 9변수 그리드 결과를 보존하고, XGBoost·LightGBM은 각각 108개 조합×3 fold의 전체 그리드를 새로 탐색했다. 검증 RMSE 최저치의 2% 이내 후보에서 피크 구간 MAE가 가장 낮은 설정을 선택했다. 경보 기준값도 검증 세트에서 정했다.",
                f"◦ Test 결과: LSTM {scores.loc['lstm'].rmse:.2f}, TCN {scores.loc['tcn'].rmse:.2f}, 앙상블 {e.rmse:.2f}, XGBoost {scores.loc['xgboost'].rmse:.2f}, LightGBM {scores.loc['lightgbm'].rmse:.2f} kW (RMSE).",
                f"- 앙상블의 MAE는 {e.mae:.2f} kW, R²는 {e.r2:.3f}, 피크 경보 F1은 {e.alert_f1:.3f}, 재현율은 {e.alert_recall:.1%}다. 다섯 후보 중 RMSE와 경보 F1이 모두 가장 좋아 최종 모델로 선정했다.",
                "- 과거 동일 시각 전력 사용의 단순 기준선 RMSE는 발표자료 기준 12.40 kW다. 기준선 수치는 발표자료의 기존 결과이며 이번 트리 재학습에서 별도 재계산하지 않았다.",
            ],
            [
                "◦ 피크 구간의 실패 사례",
                f"- Test의 177 kW 이상 피크는 49시간이다. 최종 앙상블은 이 중 {int(49-e.fn)}시간을 경보하고 {int(e.fn)}시간을 놓쳤으며, 피크가 아닌 시간에서 {int(e.fp)}건의 오경보를 냈다. 재현율을 높인 운영 설정에는 오경보 부담이 따른다.",
                f"- 후보별 미탐지/오경보: LSTM {int(scores.loc['lstm'].fn)}/{int(scores.loc['lstm'].fp)}, TCN {int(scores.loc['tcn'].fn)}/{int(scores.loc['tcn'].fp)}, XGBoost {int(scores.loc['xgboost'].fn)}/{int(scores.loc['xgboost'].fp)}, LightGBM {int(scores.loc['lightgbm'].fn)}/{int(scores.loc['lightgbm'].fp)}. 동일한 Test 703시간의 판정이다.",
                "- 오경보는 실제 피크가 아닌데 운영 담당자에게 확인을 요구한다. 현장 조치 전에는 예측 전력량과 최근 실측, 생산계획을 함께 확인하도록 설계한다.",
                "◦ 영향요인 해석 범위",
                "- 전력 이력·생산·기상·시간 정보의 조합으로 예측한다. 현재 보존된 다섯 후보 비교만으로 개별 변수의 인과적 효과나 원인 순위를 확정할 수 없다. 별도 피처 엔지니어링 또는 변수 제거 실험은 결과에 포함하지 않았다.",
            ],
            [
                "◦ 현장 사용 흐름",
                "- 시간별 계측값과 생산·기상 정보를 정제해 모델에 공급하고, 다음 시점의 예상 전력(kW)과 177 kW 초과 위험 경보를 담당자에게 제시한다. 운영자는 생산 일정·설비 상태를 확인해 조정 가능 부하를 검토한다.",
                "- 높은 위험이 이어질 때는 비필수 설비 가동 시점 조정, 공정 간 부하 분산, 에너지 담당자의 사전 확인을 제안한다. 실제 제어는 안전·품질·생산 제약을 검토한 후 사람이 결정한다.",
                "- 대시보드는 보존된 2021년 Test 예측을 재생한다. 신규 시점의 자동 추론이나 설비 실시간 연결은 구현하지 않았다. 현장 투입 전에는 신경망 가중치 보존, 신규 데이터 추론, 재학습·드리프트 감시가 필요하다.",
            ],
            [
                "◦ 비교 가능성을 중심으로 한 구성",
                "- 9개 기본 변수와 동일한 시간 경계를 유지한 채 시퀀스 모델 2종, 그 가중 앙상블, 트리 모델 2종을 비교했다. 두 트리 모델은 이번 브랜치에서만 재학습했고 그리드 후보별 fold 결과를 모두 저장했다.",
                f"- 검증 오차로 정한 앙상블 가중치가 Test RMSE를 LSTM 대비 {scores.loc['lstm'].rmse-e.rmse:.2f} kW, TCN 대비 {scores.loc['tcn'].rmse-e.rmse:.2f} kW 낮췄다. Test 결과로 가중치를 다시 선택하지 않았다.",
                "- 피크 경보를 단일 RMSE만으로 판단하지 않고 미탐지·오경보를 함께 보고한다. 사용자는 대시보드에서 5개 모델 오차와 최종 모델의 시간별 예측·경보를 확인할 수 있다.",
            ],
            [
                "◦ 재현 순서와 제출 구성",
                "- data/okm_cleaned_2021.csv는 KJH 전처리 결과다. src/preprocessing.py에 정제 코드가 있고, lhs_cleaned/grid_search.py 및 final_evaluate.py에 LSTM·TCN 탐색과 평가가 있다. 두 신경망의 seed 42 결과 파일도 lhs_cleaned/results/에 보관한다.",
                "- python scripts/train_base9_trees.py --models lightgbm xgboost --n-jobs 8 로 트리 그리드를 실행한다. 중간 fold 결과는 results/base9_tree/grid_search/에 저장되어 중단 후 재개할 수 있다. 학습 모델과 Test 예측은 results/base9_tree/models/ 및 predictions/에 있다.",
                "- python scripts/build_base9_comparison.py 를 실행하면 두 트리의 108개×3 fold 완료 여부와 다섯 후보의 Test 시각·실측값 일치를 확인하고 results/base9_comparison.csv를 만든다. python dashboard/app.py 는 저장된 Test 예측을 읽어 화면에 보여준다.",
                "- 재현 환경과 경로는 저장소 README.md 및 BASE9_COMPARISON.md에 정리했다. 무작위 분할이나 Test 기반 하이퍼파라미터 선택은 사용하지 않았다.",
            ],
        ]
        headings = [1, 8, 16, 24, 32, 40]
        new_children = [cover]
        figures = {
            1: ("01_model_comparison.png", "그림 1. 다섯 후보의 동일 Test 구간 RMSE 및 피크 경보 F1"),
            2: ("02_peak_timeline.png", "그림 2. 피크가 집중된 기간의 실측·앙상블 예측·경보"),
            3: ("03_dashboard.png", "그림 3. 저장된 Test 예측을 재생하는 Dash 대시보드 실제 화면"),
        }
        for n, position in enumerate(headings):
            heading = children[position]
            heading.set("pageBreak", "1")
            new_children.append(heading)
            bullet = children[position + 1]
            for line in chapters[n]:
                new_children.append(paragraph(bullet, line))
            if n in figures:
                filename, caption = figures[n]
                path = ROOT / "report/figures" / filename
                if not path.exists():
                    raise FileNotFoundError(f"Generate report figures first: {path}")
                number = n
                new_children.append(paragraph(bullet, caption))
                new_children[-1].find(f"{{{HP}}}run").set("charPrIDRef", "38")
                new_children[-1].set("pageBreak", "1")
                new_children.append(picture_paragraph(children[position + 2], original_pic,
                                                       f"report_image{number}", path, number))
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
        for n, (filename, _) in figures.items():
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
            for filename, _ in figures.values():
                target.write(ROOT / "report/figures" / filename, f"BinData/{filename}")
    with zipfile.ZipFile(OUTPUT) as check:
        assert check.testzip() is None
        finished = etree.fromstring(check.read("Contents/section0.xml"))
        plain = " ".join(finished.xpath(".//hp:t/text()", namespaces=NS))
        assert "8.67 kW" in plain and "108개 조합" in plain and "작성 요령" not in plain
        assert len([x for x in finished.iter() if etree.QName(x).localname == "pic"]) == 4
        assert not any(etree.QName(x).localname == "linesegarray" for x in finished.iter())
    print(OUTPUT)


if __name__ == "__main__":
    fill()
