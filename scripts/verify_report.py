"""Check that the filled HWPX keeps the form chapters and embedded images."""

from io import BytesIO
from pathlib import Path
import zipfile

from lxml import etree
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/OKM_경진대회_결과보고서_작성본.hwpx"


def main():
    with zipfile.ZipFile(REPORT) as package:
        if package.testzip() is not None:
            raise ValueError("HWPX ZIP has a corrupt member")
        section = etree.fromstring(package.read("Contents/section0.xml"))
        manifest = etree.fromstring(package.read("Contents/content.hpf"))
        text = " ".join(node.text or "" for node in section.iter()
                        if etree.QName(node).localname == "t")
        for chapter in range(1, 7):
            if f"제{chapter}장" not in text:
                raise ValueError(f"Missing chapter {chapter}")
        if "만족도 조사 완료" not in text or "작성 요령" in text:
            raise ValueError("Form control sections were altered")
        if any(etree.QName(node).localname == "linesegarray" for node in section.iter()):
            raise ValueError("Stale line layout remains")
        image_ids = {node.get("binaryItemIDRef") for node in section.iter()
                     if etree.QName(node).localname == "img"}
        items = {node.get("id"): node.get("href") for node in manifest.iter()
                 if etree.QName(node).localname == "item"}
        if len(image_ids) != 4 or not image_ids <= items.keys():
            raise ValueError("Expected three report figures plus the form image")
        for ident in image_ids:
            with Image.open(BytesIO(package.read(items[ident]))) as image:
                image.verify()
        print(f"HWPX 정상: 6개 장, {len(image_ids)}개 내장 이미지, {len(package.namelist())}개 패키지 항목")


if __name__ == "__main__":
    main()
