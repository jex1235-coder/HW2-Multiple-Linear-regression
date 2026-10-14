"""Static deployment-file checks; a live cloud wake-up needs a deployed URL."""

import ast
import textwrap
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cloud_requirements_are_pinned_and_include_visualization_stack():
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = dict(line.split("==", 1) for line in lines if line and not line.startswith("#"))
    assert all("==" in line for line in lines if line and not line.startswith("#"))
    assert {"streamlit", "pandas", "numpy", "scikit-learn", "statsmodels", "plotly"} <= packages.keys()


def test_keep_alive_schedule_and_embedded_python_are_valid():
    text = (ROOT / ".github" / "workflows" / "keep_alive.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    triggers = workflow.get("on", workflow.get(True))  # PyYAML 1.1 treats `on` as a boolean.
    assert triggers["schedule"][0]["cron"] == "17 */4 * * *"
    assert "workflow_dispatch" in triggers
    assert triggers["push"]["branches"] == ["main"]
    assert "app-0qudfk.streamlit.app" in workflow["jobs"]["visit"]["env"]["APP_URL"]
    script = text.split("python - <<'PY'", 1)[1].split("\n          PY", 1)[0]
    ast.parse(textwrap.dedent(script))
    assert "STREAMLIT_APP_URL" in text
    assert 'page.on("websocket"' in script
    assert "wake.click(" in script
    assert "for frame in page.frames" in script
