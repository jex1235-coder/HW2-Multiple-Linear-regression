"""Optional real-browser checks. Start Streamlit locally before running these tests."""

import os
import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402


URL = os.environ.get("STREAMLIT_TEST_URL", "http://127.0.0.1:8501/")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


@pytest.fixture(scope="module")
def browser():
    if not CHROME.is_file():
        pytest.skip("本機沒有可用的 Chrome；可改以 AppTest 執行核心功能測試。")
    try:
        with urlopen(URL + "_stcore/health", timeout=3) as response:
            if response.status != 200:
                pytest.skip("Streamlit 本機伺服器尚未就緒。")
    except URLError:
        pytest.skip("先執行 streamlit run app.py 再跑瀏覽器測試。")
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(executable_path=str(CHROME), headless=True)
        yield instance
        instance.close()


def test_animated_guide_and_mobile_layout(browser):
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.goto(URL, wait_until="domcontentloaded")
    guide_heading = page.get_by_text("用六個決策讀懂完整報告")
    guide_heading.wait_for(timeout=30000)
    assert not page.evaluate("document.documentElement.scrollWidth > innerWidth")
    assert page.get_by_text("讓每一筆耗電資料，").is_visible()
    guide_heading.scroll_into_view_if_needed()
    guide = page.frame_locator("iframe")
    guide.get_by_role("button", name="下一步 →").click()
    assert guide.locator("#count").inner_text() == "02 / 06"
    assert guide.locator("#copy h3").inner_text() == "先看分布，再談模型分數。"
    assert page.locator("iframe").first.evaluate("(e) => e.clientHeight") >= 900
    page.close()


def test_real_browser_training_and_tab_persistence(browser):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")
    page.get_by_role("button", name="⚡ 執行完整分析").click()
    page.get_by_text("本次分析完成").wait_for(timeout=90000)
    page.get_by_role("tab", name="◎ 評估 · 95% 區間").click()
    assert page.locator(".js-plotly-plot").count() >= 1
    page.get_by_role("tab", name="↗ 單筆預測實驗室").click()
    page.get_by_role("button", name="⚡ 估計這筆樣本").click()
    page.get_by_text("LinearRegression 點預測").wait_for(timeout=30000)
    assert page.get_by_role("tab", name="↗ 單筆預測實驗室").get_attribute("aria-selected") == "true"
    page.close()


def test_custom_csv_upload_trains_without_default_dataset(browser):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(120, 14)), columns=[f"x{i:02d}" for i in range(14)])
    frame["Profit"] = 100 + 7 * frame["x00"] - 3 * frame["x01"] + rng.normal(size=120)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(URL, wait_until="domcontentloaded")
    page.locator('input[type="file"]').set_input_files({
        "name": "custom_regression.csv", "mimeType": "text/csv",
        "buffer": frame.to_csv(index=False).encode("utf-8"),
    })
    page.get_by_text("custom_regression.csv").first.wait_for(timeout=30000)
    page.get_by_text("120 筆觀測").first.wait_for(timeout=30000)
    page.get_by_label("◎ 預測目標 Target y").click()
    page.get_by_label("◎ 預測目標 Target y").fill("Profit")
    page.get_by_role("option", name=re.compile("Profit")).click()
    page.get_by_role("button", name="⚡ 執行完整分析").click()
    page.get_by_text("本次分析完成").wait_for(timeout=90000)
    assert page.get_by_text("120 筆觀測").first.is_visible()
    page.get_by_role("tab", name="↗ 單筆預測實驗室").click()
    page.get_by_role("button", name="⚡ 估計這筆樣本").click()
    page.get_by_text("LinearRegression 點預測").wait_for(timeout=30000)
    assert page.locator('[data-testid="stException"]').count() == 0
    page.close()
