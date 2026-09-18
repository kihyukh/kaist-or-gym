"""Optional real-browser checks for touch targets and small-screen overflow.

Install Playwright and Chromium to run these checks. A locally installed Chrome
or COFFEE_BROWSER_EXECUTABLE can also supply the browser, without a download.
"""

import os
import shutil
from pathlib import Path

import pytest

from kaist_rl_lab.envs.coffee_pouring_canvas import (
    CANVAS_CSS,
    CANVAS_HTML,
    CANVAS_JAVASCRIPT,
)


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("COFFEE_BROWSER_EXECUTABLE")
    if not executable:
        executable = shutil.which("chromium") or shutil.which("google-chrome")
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not executable and mac_chrome.exists():
        executable = str(mac_chrome)
    with playwright.sync_playwright() as runtime:
        try:
            instance = runtime.chromium.launch(executable_path=executable, headless=True)
        except playwright.Error as exc:
            pytest.skip(f"A Chromium browser is required for mobile layout checks: {exc}")
        yield instance
        instance.close()


@pytest.mark.parametrize("width,height", [(360, 780), (390, 844), (430, 932), (740, 360)])
def test_scene_and_all_joint_controls_fit_phone(browser, width, height):
    page = browser.new_page(viewport={"width": width, "height": height}, has_touch=True)
    try:
        page.set_content(
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<style>body {{margin: 12px;}} {CANVAS_CSS}</style>"
            f'<main id="demo">{CANVAS_HTML}</main>'
        )
        page.add_script_tag(content="const element = document.querySelector('#demo');\n" +
                            CANVAS_JAVASCRIPT)
        page.evaluate("""updateJointControls({motors:[0,0,0,0,0,0],running:true,
            fill:0.123,targetFill:0.25,spill:0.009,sourceRemaining:0.368})""")
        dimensions = page.evaluate("""() => {
            const stage = document.querySelector('.coffee-stage').getBoundingClientRect();
            return {
                pageWidth: document.documentElement.scrollWidth,
                stageHeight: stage.height,
                buttons: Array.from(document.querySelectorAll('.coffee-joint-button'), button => {
                    const bounds = button.getBoundingClientRect();
                    return {width:bounds.width, height:bounds.height,
                        inside:bounds.left >= stage.left && bounds.right <= stage.right,
                        enabled:!button.disabled};
                })
            };
        }""")
        assert dimensions["pageWidth"] <= width
        assert dimensions["stageHeight"] < (330 if width > height else 570)
        assert len(dimensions["buttons"]) == 18
        for button in dimensions["buttons"]:
            assert button["width"] >= 44
            assert button["height"] >= 44
            assert button["inside"]
            assert button["enabled"]
        assert page.locator('[data-coffee-stat="fill"]').inner_text() == "123 / 250 mL"
        assert page.locator('[data-coffee-stat="spill"]').inner_text() == "9 mL"
        assert page.locator('[data-coffee-stat="remaining"]').inner_text() == "368 mL"
        assert page.locator('.coffee-joint-button[aria-pressed="true"]').count() == 6
    finally:
        page.close()
