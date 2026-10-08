"""Capture all four live Dash views for the competition report."""

from pathlib import Path
import sys
from threading import Thread

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from app import app  # noqa: E402


def main():
    output_dir = ROOT / "docs/report/figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = (
        ("/", "피크 경보 검증 결과", "03_dashboard.png", 2, 1000),
        ("/causes", "변수 중요도 및 오류 조건 점검", "12_dashboard_causes.png", 2, 1000),
        ("/model", "예측 모델 성능 및 오류 분석", "13_dashboard_model.png", 2, 1250),
        ("/actions", "피크 경보 오류 사례 검토", "14_dashboard_actions.png", 0, 1000),
    )
    server = make_server("127.0.0.1", 0, app.server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
            except Exception:
                browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            for route, title, filename, minimum_plots, height in routes:
                page.set_viewport_size({"width": 1440, "height": height})
                page.goto(f"http://127.0.0.1:{server.server_port}{route}", wait_until="networkidle")
                page.get_by_text(title, exact=True).wait_for()
                if minimum_plots:
                    page.wait_for_function(
                        "count => document.querySelectorAll('.js-plotly-plot').length >= count",
                        arg=minimum_plots,
                    )
                page.screenshot(path=str(output_dir / filename), full_page=False)
            browser.close()
    finally:
        server.shutdown()
        thread.join()
    for _, _, filename, _, _ in routes:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
