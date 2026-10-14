"""Optional real-browser checks. Start Streamlit locally before running these tests."""

import os
import re
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright, expect  # noqa: E402


URL = os.environ.get("STREAMLIT_TEST_URL", "http://127.0.0.1:8501/")
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")


@pytest.fixture(scope="module")
def browser():
    if not CHROME.is_file():
        pytest.skip("本機沒有可用的 Chrome；可改以 AppTest 執行核心功能測試。")
    if ".streamlit.app" not in URL:
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


def open_app(browser, viewport):
    host = browser.new_page(viewport=viewport)
    host.goto(URL, wait_until="domcontentloaded", timeout=60000)
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        for frame in host.frames:
            if frame.is_detached():
                continue
            wake = frame.get_by_text(re.compile(r"Yes, get this app back up!?", re.I)).first
            if wake.count() and wake.is_visible():
                wake.click()
            if frame.locator('[data-testid="stApp"]').is_visible():
                frame.get_by_role("button", name="⚡ 執行完整分析").wait_for(timeout=60000)
                return host, frame
        host.wait_for_timeout(1000)
    host.close()
    pytest.fail("The deployed app did not become ready")


def test_animated_guide_and_mobile_layout(browser):
    host, page = open_app(browser, {"width": 390, "height": 844})
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
    host.close()


def test_real_browser_training_and_tab_persistence(browser):
    host, page = open_app(browser, {"width": 1440, "height": 900})
    page.get_by_role("button", name="⚡ 執行完整分析").click()
    page.get_by_text("本次分析完成").wait_for(timeout=90000)
    page.get_by_role("tab", name="◎ 評估 · 95% 區間").click()
    page.locator(".js-plotly-plot:visible").first.wait_for(state="visible", timeout=60000)
    window_slider = page.get_by_role("slider", name="顯示連續測試點數（預設 432；原資料為 72 小時）")
    window_slider.focus()
    window_slider.press("ArrowRight")
    page.get_by_role("tab", name="◎ 評估 · 95% 區間").wait_for()
    assert page.get_by_role("tab", name="◎ 評估 · 95% 區間").get_attribute("aria-selected") == "true"
    with host.expect_download(timeout=30000) as download:
        page.get_by_role("button", name="⬇ 下載本次 Test 預測與區間 CSV").click()
    downloaded = pd.read_csv(download.value.path())
    assert len(downloaded) == 2961
    expect(page.get_by_role("tab", name="◇ 特徵選擇 · VIF")).to_have_count(1, timeout=30000)
    page.get_by_role("tab", name="◇ 特徵選擇 · VIF").click()
    expect(page.get_by_text("VIF 使用 Training", exact=False)).to_be_visible(timeout=30000)
    page.get_by_role("tab", name="⌁ 五項假設").click()
    expect(page.get_by_text("Train Durbin–Watson", exact=True)).to_be_visible(timeout=30000)
    page.get_by_role("tab", name="↗ 單筆預測實驗室").click()
    page.get_by_role("button", name="⚡ 估計這筆樣本").click()
    page.get_by_text("LinearRegression 點預測").wait_for(timeout=30000)
    assert page.get_by_role("tab", name="↗ 單筆預測實驗室").get_attribute("aria-selected") == "true"
    assert page.locator('[data-testid="stException"]').count() == 0
    host.close()


def test_custom_csv_upload_trains_without_default_dataset(browser):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(120, 14)), columns=[f"x{i:02d}" for i in range(14)])
    frame["Profit"] = 100 + 7 * frame["x00"] - 3 * frame["x01"] + rng.normal(size=120)
    host, page = open_app(browser, {"width": 1440, "height": 900})
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
    host.close()
