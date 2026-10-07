"""Capture the actual Dash page for the HWPX report (requires Playwright and Chrome)."""

from pathlib import Path
import sys
from threading import Thread

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from app import app  # noqa: E402


def main():
    output = ROOT / "report/figures/03_dashboard.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    server = make_server("127.0.0.1", 0, app.server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
            except Exception:
                browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
            page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
            page.locator("#timeline .js-plotly-plot").wait_for()
            page.locator("#ranking .js-plotly-plot").wait_for()
            page.screenshot(path=str(output), full_page=True)
            browser.close()
    finally:
        server.shutdown()
        thread.join()
    print(output)


if __name__ == "__main__":
    main()
