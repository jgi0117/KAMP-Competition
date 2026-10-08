"""Browser smoke tests for every dashboard route and core interaction."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import tempfile
from threading import Thread

from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))
from app import app  # noqa: E402
from data import feature_rows, local_tree_importance  # noqa: E402


ROUTES = {
    "/": ("피크 경보 검증 결과", 2),
    "/causes": ("변수 중요도 및 오류 조건 점검", 2),
    "/model": ("예측 모델 성능 및 오류 분석", 2),
    "/actions": ("피크 경보 오류 사례 검토", 0),
}


def assert_no_horizontal_overflow(page) -> None:
    result = page.evaluate(
        """() => ({
            viewport: window.innerWidth,
            scrollWidth: document.documentElement.scrollWidth,
            offenders: [...document.querySelectorAll('body *')]
                .map((element) => {
                    const box = element.getBoundingClientRect();
                    return {tag: element.tagName, id: element.id, cls: element.className,
                            left: Math.round(box.left), right: Math.round(box.right), width: Math.round(box.width)};
                })
                .filter((box) => box.right > window.innerWidth + 2 || box.left < -2)
                .sort((a, b) => b.right - a.right)
                .slice(0, 12)
        })"""
    )
    if result["scrollWidth"] > result["viewport"] + 2:
        raise AssertionError(f"Horizontal overflow: {result}")


def main() -> None:
    first_local = local_tree_importance("2021-08-17 00:00")
    later_local = local_tree_importance("2021-08-17 12:00")
    assert len(first_local) == len(later_local) == 18
    assert not first_local.contribution_kw.equals(later_local.contribution_kw)
    assert feature_rows("2021-08-17 00:00") != feature_rows("2021-08-17 12:00")

    output_dir = Path(tempfile.mkdtemp(prefix="dashboard-validation-"))
    server = make_server("127.0.0.1", 0, app.server)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    errors: list[str] = []
    try:
        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(channel="chrome", headless=True)
            except Exception:
                browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1920, "height": 1000})
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            base = f"http://127.0.0.1:{server.server_port}"

            for route, (expected_text, minimum_plots) in ROUTES.items():
                page.goto(base + route, wait_until="networkidle")
                page.get_by_text(expected_text, exact=True).wait_for()
                assert page.locator(".js-plotly-plot").count() >= minimum_plots
                assert_no_horizontal_overflow(page)
                name = "home" if route == "/" else route.strip("/")
                page.screenshot(path=str(output_dir / f"{name}-1920.png"), full_page=True)

            page.goto(base + "/", wait_until="networkidle")
            page.locator("#range-12h").click()
            page.wait_for_timeout(300)
            assert "active" in (page.locator("#range-12h").get_attribute("class") or "")
            page.locator("#open-action-drawer").click()
            expect(page.locator("#action-drawer")).to_have_class(re.compile(r"\bopen\b"))
            page.locator("#close-action-drawer").click()
            expect(page.locator("#action-drawer")).not_to_have_class(re.compile(r"\bopen\b"))

            # Reproduce the original failure path: navigate from the home page,
            # remain on /causes for another global replay tick, and verify all
            # three local-explanation views change without console errors.
            page.locator('a[href="/causes"]').first.click()
            page.wait_for_url(base + "/causes")
            page.locator("#cause-detail-chart .js-plotly-plot").wait_for()
            before_chart = page.locator("#cause-detail-chart .js-plotly-plot").evaluate(
                "el => el.data.map(trace => trace.x)"
            )
            before_average = page.locator("#cause-average-chart .js-plotly-plot").evaluate(
                "el => el.data.map(trace => trace.x)"
            )
            before_rows = page.locator("#feature-rows-container").inner_text()
            page.wait_for_timeout(5_500)
            after_chart = page.locator("#cause-detail-chart .js-plotly-plot").evaluate(
                "el => el.data.map(trace => trace.x)"
            )
            after_average = page.locator("#cause-average-chart .js-plotly-plot").evaluate(
                "el => el.data.map(trace => trace.x)"
            )
            after_rows = page.locator("#feature-rows-container").inner_text()
            assert before_chart != after_chart, "Model-specific local contributions did not update"
            assert before_average != after_average, "Averaged local contributions did not update"
            assert before_rows != after_rows, "Top-three local contribution rows did not update"

            page.set_viewport_size({"width": 1280, "height": 900})
            page.goto(base + "/", wait_until="networkidle")
            assert_no_horizontal_overflow(page)
            page.screenshot(path=str(output_dir / "home-1280.png"), full_page=True)
            browser.close()
    finally:
        server.shutdown()
        thread.join()

    if errors:
        raise RuntimeError(f"Browser console errors: {errors}")
    print(f"Dashboard routes, interactions, and dynamic TreeSHAP updates passed. Screenshots: {output_dir}")


if __name__ == "__main__":
    main()
