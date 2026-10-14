"""Regression, assignment-rule, and Streamlit smoke tests for the companion site."""

import logging

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

logging.disable(logging.CRITICAL)
import app  # noqa: E402 - exercises the same module used by Streamlit


@pytest.fixture(scope="module")
def default_frame():
    return app.load_csv(app.DEFAULT_CSV.read_bytes())


@pytest.fixture(scope="module")
def synthetic_frame():
    rng = np.random.default_rng(5115056030)
    n = 150
    frame = pd.DataFrame(rng.normal(size=(n, 15)), columns=[f"x{i:02d}" for i in range(15)])
    frame["x01"] = frame["x00"] * .72 + rng.normal(scale=.28, size=n)
    frame.loc[[7, 31], "x04"] = np.nan
    frame["Profit"] = 70 + 9 * frame["x00"] - 4 * frame["x02"] + rng.normal(scale=3, size=n)
    frame["date"] = pd.date_range("2025-01-01", periods=n, freq="h")
    return frame


def test_report_main_model_reproduces_archived_metrics(default_frame):
    features = [c for c in default_frame.columns if c not in {"date", "Appliances", "rv1", "rv2"}]
    result = app.run_analysis(default_frame, "Appliances", features, "date", app.CHRONO,
                              "Permutation Importance", 14, "LinearRegression", False)
    assert (result["train_n"], result["validation_n"], result["test_n"]) == (13814, 2960, 2961)
    assert result["metrics"]["RMSE"] == pytest.approx(89.20, abs=.02)
    assert result["metrics"]["R²"] == pytest.approx(.0366, abs=.0002)
    assert len(result["selected_names"]) == 14
    assert result["pi_coverage"] == pytest.approx(.9649, abs=.0001)
    point, interval = app.predict_one(result, result["demo_row"])
    assert point == pytest.approx(result["prediction"][0], abs=1e-8)
    assert interval["obs_ci_lower"] == pytest.approx(result["intervals"].iloc[0]["obs_ci_lower"])


def test_report_arx_comparison_reproduces_archived_metrics(default_frame):
    features = [c for c in default_frame.columns if c not in {"date", "Appliances", "rv1", "rv2"}]
    result = app.run_analysis(default_frame, "Appliances", features, "date", app.CHRONO,
                              "Permutation Importance", 28, "LinearRegression", True)
    assert (result["train_n"], result["validation_n"], result["test_n"]) == (13811, 2960, 2961)
    assert result["metrics"]["RMSE"] == pytest.approx(61.40, abs=.02)
    assert result["metrics"]["R²"] == pytest.approx(.5435, abs=.0002)
    assert result["pi_coverage"] == pytest.approx(.9547, abs=.0001)
    peak = result["coverage_table"].set_index("群組").loc[">400 Wh"]
    assert peak["樣本數"] == 69
    assert peak["PI涵蓋率"] == pytest.approx(17 / 69)


@pytest.mark.parametrize("selector", app.SELECTORS)
def test_each_feature_selector_on_custom_csv(synthetic_frame, selector):
    features = [c for c in synthetic_frame if c.startswith("x")]
    result = app.run_analysis(synthetic_frame, "Profit", features, "date", app.CHRONO,
                              selector, 5, "LinearRegression", False)
    assert len(result["selected_names"]) == 5
    assert np.isfinite(result["metrics"]["RMSE"])
    assert result["test_n"] > 0
    assert len(result["intervals"]) == result["test_n"]
    assert not result["coverage_table"].empty


def test_random_split_and_regularized_model(synthetic_frame):
    frame = synthetic_frame.copy()
    frame["segment"] = np.where(np.arange(len(frame)) % 2, "business", "home")
    features = [c for c in frame if c.startswith("x")] + ["segment"]
    result = app.run_analysis(frame, "Profit", features, "date", app.RANDOM,
                              "SelectKBest", 8, "Ridge", False)
    assert result["validation_n"] == 0
    assert result["test_n"] == 30
    assert np.isfinite(result["metrics"]["RMSE"])
    point, interval = app.predict_one(result, result["demo_row"])
    assert np.isfinite(point)
    assert interval["obs_ci_lower"] < interval["obs_ci_upper"]


def test_invalid_lag_and_reserved_columns_are_rejected(synthetic_frame):
    features = [c for c in synthetic_frame if c.startswith("x")]
    with pytest.raises(ValueError, match="只能配合時間順序"):
        app.run_analysis(synthetic_frame, "Profit", features, "date", app.RANDOM,
                         "SelectKBest", 5, "LinearRegression", True)
    frame = synthetic_frame.assign(__target_lag_1=1.0)
    with pytest.raises(ValueError, match="系統保留名稱"):
        app.run_analysis(frame, "Profit", features, "date", app.CHRONO,
                         "SelectKBest", 5, "LinearRegression", False)


def test_csv_header_and_empty_file_are_rejected():
    with pytest.raises(ValueError, match="重複欄名"):
        app.load_csv(b"target,x,x\n1,2,3\n")
    with pytest.raises(ValueError, match="檔案為空"):
        app.load_csv(b"")


def test_default_streamlit_flow_and_single_prediction():
    site = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=180).run()
    assert not site.exception
    assert len(site.tabs) == 5
    next(button for button in site.button if button.label == "⚡ 執行完整分析").click()
    site.run()
    assert not site.exception
    assert len(site.get("plotly_chart")) >= 8
    next(button for button in site.button if button.label == "⚡ 估計這筆樣本").click()
    site.run()
    assert not site.exception
    assert any(metric.label == "LinearRegression 點預測" for metric in site.metric)
