"""Check that the filled HWPX keeps the form chapters and embedded images."""

from io import BytesIO
from pathlib import Path
import re
import zipfile

from lxml import etree
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/report/OKM_경진대회_결과보고서_작성본.hwpx"
TEMPLATE = next((ROOT / "docs/templates").glob("*결과보고서*.hwpx"))


def main():
    with zipfile.ZipFile(REPORT) as package:
        if package.testzip() is not None:
            raise ValueError("HWPX ZIP has a corrupt member")
        section = etree.fromstring(package.read("Contents/section0.xml"))
        header = etree.fromstring(package.read("Contents/header.xml"))
        manifest = etree.fromstring(package.read("Contents/content.hpf"))
        text = " ".join(node.text or "" for node in section.iter()
                        if etree.QName(node).localname == "t")
        for chapter in range(1, 7):
            if f"□ 제{chapter}장" not in text:
                raise ValueError(f"Missing chapter {chapter}")
        for required in ("F1은 0.366", "F1 0.412", "0.161", "0.115", "관리한계 내부"):
            if required not in text:
                raise ValueError(f"Missing corrected report text: {required}")
        if "만족도 조사 완료" not in text or "작성 요령" in text:
            raise ValueError("Form control sections were altered")
        if any(etree.QName(node).localname == "linesegarray" for node in section.iter()):
            raise ValueError("Stale line layout remains")
        if sum(node.get("pageBreak") == "1" for node in list(section)) != 1:
            raise ValueError("Unexpected forced page breaks remain")
        char_properties = next(node for node in header.iter()
                               if etree.QName(node).localname == "charProperties")
        if len(char_properties) != 37:
            raise ValueError("Report must reuse the form's original font sizes")
        char_ids = {node.get("id") for node in char_properties}
        char_refs = {node.get("charPrIDRef") for node in section.iter()
                     if node.get("charPrIDRef") is not None}
        if not char_refs <= char_ids:
            raise ValueError(f"Unknown character properties: {sorted(char_refs - char_ids)}")
        para_properties = next(node for node in header.iter()
                               if etree.QName(node).localname == "paraProperties")
        para_ids = {node.get("id") for node in para_properties}
        para_refs = {node.get("paraPrIDRef") for node in section.iter()
                     if node.get("paraPrIDRef") is not None}
        if not para_refs <= para_ids:
            raise ValueError(f"Unknown paragraph properties: {sorted(para_refs - para_ids)}")
        if not {"42", "43", "44", "45"} <= para_ids:
            raise ValueError("Missing centered figure or title/body spacing styles")
        for para_id in ("27", "42", "43", "44", "45", "46", "47"):
            style = next(node for node in para_properties if node.get("id") == para_id)
            line_spacings = [node for node in style.iter()
                             if etree.QName(node).localname == "lineSpacing"]
            if not line_spacings or any(node.get("type") != "PERCENT"
                                        or node.get("value") != "160"
                                        for node in line_spacings):
                raise ValueError(f"Paragraph style {para_id} must keep 160% line spacing")
        expected_margins = {
            "27": ("150", "1000"),
            "42": ("900", "200"),
            "43": ("1400", "700"),
            "44": ("900", "350"),
            "45": ("0", "250"),
            "46": ("0", "0"),
            "47": ("0", "0"),
        }
        for para_id, expected in expected_margins.items():
            style = next(node for node in para_properties if node.get("id") == para_id)
            margins = {
                (
                    next(child for child in margin if etree.QName(child).localname == "prev").get("value"),
                    next(child for child in margin if etree.QName(child).localname == "next").get("value"),
                )
                for margin in style.iter()
                if etree.QName(margin).localname == "margin"
            }
            if margins != {expected}:
                raise ValueError(f"Paragraph style {para_id} has unexpected before/after spacing")
        for para_id in ("27", "42", "47"):
            style = next(node for node in para_properties if node.get("id") == para_id)
            align = next(node for node in style.iter() if etree.QName(node).localname == "align")
            if align.get("horizontal") != "CENTER":
                raise ValueError(f"Paragraph style {para_id} must be centered")
        left_table_style = next(node for node in para_properties if node.get("id") == "46")
        left_align = next(node for node in left_table_style.iter()
                          if etree.QName(node).localname == "align")
        if left_align.get("horizontal") != "LEFT":
            raise ValueError("Paragraph style 46 must be left-aligned")
        with zipfile.ZipFile(TEMPLATE) as template_package:
            template_header = etree.fromstring(template_package.read("Contents/header.xml"))
        template_chars = next(node for node in template_header.iter()
                              if etree.QName(node).localname == "charProperties")
        for char_id in ("8", "13", "14", "15"):
            expected = next(node for node in template_chars if node.get("id") == char_id)
            actual = next(node for node in char_properties if node.get("id") == char_id)
            if etree.tostring(actual) != etree.tostring(expected):
                raise ValueError(f"Character style {char_id} differs from the supplied template")
        image_ids = {node.get("binaryItemIDRef") for node in section.iter()
                     if etree.QName(node).localname == "img"}
        items = {node.get("id"): node.get("href") for node in manifest.iter()
                 if etree.QName(node).localname == "item"}
        if len(image_ids) != 15 or not image_ids <= items.keys():
            raise ValueError("Expected fourteen report figures plus the form image")
        figure_numbers = []
        for paragraph in list(section):
            caption_text = "".join(node.text or "" for node in paragraph.iter()
                                   if etree.QName(node).localname == "t")
            match = re.match(r"그림 (\d+)\.", caption_text)
            if match and paragraph.get("paraPrIDRef") == "27":
                figure_numbers.append(int(match.group(1)))
        if figure_numbers != list(range(1, 15)):
            raise ValueError(f"Report figure numbers are out of order: {figure_numbers}")
        report_tables = [node for node in section.iter()
                         if etree.QName(node).localname == "tbl"
                         and (node.get("id") or "").startswith("210000")]
        if len(report_tables) != 6:
            raise ValueError("Expected six editable report tables")
        if any(table.get("pageBreak") != "CELL" for table in report_tables):
            raise ValueError("Editable report tables must allow row-wise page breaks")
        table_para_refs = {node.get("paraPrIDRef") for table in report_tables
                           for node in table.iter()
                           if etree.QName(node).localname == "p"}
        if not table_para_refs <= {"46", "47"} or not {"46", "47"} <= table_para_refs:
            raise ValueError("Editable report tables must use dedicated left/center styles")
        if "관리상한은 예측값(t)+3σ" not in text or "관리하한은 max(0" not in text:
            raise ValueError("Missing detailed 3-sigma operating rule")
        report_pictures = [node for node in section.iter()
                           if etree.QName(node).localname == "pic"
                           and node.find("{http://www.hancom.co.kr/hwpml/2011/core}img").get(
                               "binaryItemIDRef", "").startswith("report_image")]
        if any(pic.getparent().getparent().get("paraPrIDRef") != "42" for pic in report_pictures):
            raise ValueError("Report figures must use the centered picture paragraph style")
        top_level = list(section)
        for index, paragraph in enumerate(top_level[:-1]):
            report_images = [node for node in paragraph.iter()
                             if etree.QName(node).localname == "img"
                             and node.get("binaryItemIDRef", "").startswith("report_image")]
            if not report_images:
                continue
            caption = top_level[index + 1]
            caption_text = "".join(node.text or "" for node in caption.iter()
                                   if etree.QName(node).localname == "t")
            if caption.get("paraPrIDRef") != "27" or not caption_text.startswith(("그림 ", "표 ")):
                raise ValueError("Each report figure must be followed by a centered caption")
        for table in report_tables:
            parent_paragraph = table.getparent().getparent()
            index = top_level.index(parent_paragraph)
            caption = top_level[index + 1]
            caption_text = "".join(node.text or "" for node in caption.iter()
                                   if etree.QName(node).localname == "t")
            if caption.get("paraPrIDRef") != "27" or not caption_text.startswith("표 "):
                raise ValueError("Each editable table must be followed by a centered caption")
        for ident in image_ids:
            with Image.open(BytesIO(package.read(items[ident]))) as image:
                image.verify()
        print(f"HWPX 정상: 6개 장, {len(image_ids)}개 내장 이미지, {len(package.namelist())}개 패키지 항목")


if __name__ == "__main__":
    main()
