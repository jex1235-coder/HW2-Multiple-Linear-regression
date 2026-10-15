"""Interactive companion to the 5115056030 CRISP-DM regression report.

The historical figures are read from the submitted outputs. A visitor's analysis
is fitted afresh, with every learned transformation restricted to its train set.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "KAG_energydata_complete.csv"
REPORT_PDF = ROOT / "5115056030_hw2_完整報告.pdf"
ASSETS = ROOT / "assets"
SELECTORS = ["SelectKBest", "RFE", "SFS", "Lasso", "Permutation Importance"]
MODELS = ["LinearRegression", "Ridge", "Lasso"]
CHRONO = "依時間順序切分（Chronological 70/15/15）"
RANDOM = "隨機切分（Random 80/20）"

st.set_page_config(page_title="能源回歸研究室 | CRISP-DM Studio", page_icon="⚡", layout="wide", initial_sidebar_state="auto")
sns.set_theme(style="whitegrid", palette="deep")
st.markdown(f"<style>{(ASSETS / 'theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def guide_html(version: int) -> str:
    """Cache the embedded archive images until the guide source changes."""
    html = (ASSETS / "guide.html").read_text(encoding="utf-8")
    images = [
        "5115056030_step2_outputs/03_target_over_time.png",
        "5115056030_step2_outputs/01_target_histogram.png",
        "5115056030_step3_outputs/01_chronological_split.png",
        "5115056030_step4_outputs/01_validation_top16_rmse.png",
        "5115056030_step5_outputs/05_PI_coverage_by_load.png",
        "5115056030_step6_outputs/02_demo_point_predictions.png",
    ]
    for index, relative in enumerate(images):
        file = ROOT / relative
        data_url = "data:image/png;base64," + base64.b64encode(file.read_bytes()).decode("ascii") if file.is_file() else ""
        html = html.replace("{{IMG_" + str(index) + "}}", data_url)
    return html


def chart_style(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 17, "color": "#eef8f7"}},
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,30,42,.62)",
        font={"family": "Noto Sans TC, DM Sans, sans-serif", "color": "#c4dadd"},
        margin={"l": 28, "r": 20, "t": 55, "b": 35},
        hoverlabel={"bgcolor": "#102d3a", "font_color": "#f1fbfa"},
        legend={"orientation": "h", "y": -0.16},
    )
    fig.update_xaxes(gridcolor="rgba(150,203,208,.10)", zerolinecolor="rgba(150,203,208,.25)")
    fig.update_yaxes(gridcolor="rgba(150,203,208,.10)", zerolinecolor="rgba(150,203,208,.25)")
    return fig


@st.cache_data(show_spinner=False)
def load_csv(data: bytes) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "cp950", "latin-1"):
        try:
            header_reader = csv.reader(io.TextIOWrapper(io.BytesIO(data), encoding=encoding, newline=""))
            try:
                header = [column.strip() for column in next(header_reader)]
            except StopIteration as exc:
                raise ValueError("CSV 檔案為空。") from exc
            if len(header) != len(set(header)):
                raise ValueError("CSV 含重複欄名；請先重新命名。")
            frame = pd.read_csv(io.BytesIO(data), encoding=encoding, low_memory=False)
            frame.columns = [str(c).strip() for c in frame.columns]
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"無法解碼 CSV：{last_error}")


def score(y: np.ndarray, prediction: np.ndarray, p: int) -> dict[str, float]:
    n = len(y)
    r2 = float(r2_score(y, prediction))
    return {
        "MAE": float(mean_absolute_error(y, prediction)),
        "MSE": float(mean_squared_error(y, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "R²": r2,
        "Adjusted R²": float(1 - (1 - r2) * (n - 1) / (n - p - 1)) if n > p + 1 else np.nan,
    }


def select_columns(x: np.ndarray, y: np.ndarray, method: str, k: int, chronological: bool):
    """Return indices and method scores. Inputs have already been train fitted/scaled."""
    p = x.shape[1]
    if k == p:
        return np.arange(p), np.ones(p), "全部欄位保留；此設定沒有縮減特徵。"
    if method == "SelectKBest":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            selector = SelectKBest(f_regression, k=k).fit(x, y)
        ranks = np.nan_to_num(selector.scores_, nan=0.0, posinf=0.0)
        return np.flatnonzero(selector.get_support()), ranks, "長條圖顯示模型標準化係數；選取依單變量 F 分數。"
    if method == "RFE":
        selector = RFE(LinearRegression(), n_features_to_select=k).fit(x, y)
        return np.flatnonzero(selector.support_), -selector.ranking_.astype(float), "RFE 依逐輪回歸係數排名。"
    if method == "SFS":
        # Bound the interactive search; the full assignment run remains in 5115056030_hw2.py.
        if len(y) > 1200:
            rows = np.linspace(0, len(y) - 1, 1200, dtype=int)
            x_fit, y_fit = x[rows], y[rows]
            note = "SFS 為使網站可互動，僅用 Training 中均勻抽取的 1,200 筆做選取；最後模型仍用全部 Training 擬合。"
        else:
            x_fit, y_fit = x, y
            note = "SFS 僅在 Training 內以交叉驗證選取特徵。"
        cv = TimeSeriesSplit(n_splits=3) if chronological else KFold(n_splits=3, shuffle=True, random_state=42)
        selector = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=k,
            direction="forward" if k <= p / 2 else "backward",
            scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1,
        ).fit(x_fit, y_fit)
        return np.flatnonzero(selector.get_support()), selector.get_support().astype(float), note
    if method == "Lasso":
        selector = Lasso(alpha=0.1, max_iter=20000).fit(x, y)
        ranks = np.abs(selector.coef_)
        indices = np.argsort(-ranks, kind="stable")[:k]
        note = "按 Lasso 係數絕對值取前 k 名；若非零係數少於 k，零係數欄位也可能入選。"
        return np.sort(indices), ranks, note
    inner = int(len(y) * 0.8)
    if chronological:
        x_fit, x_hold, y_fit, y_hold = x[:inner], x[inner:], y[:inner], y[inner:]
    else:
        x_fit, x_hold, y_fit, y_hold = train_test_split(x, y, test_size=0.2, random_state=42)
    base = LinearRegression().fit(x_fit, y_fit)
    perm = permutation_importance(
        base, x_hold, y_hold, n_repeats=5, random_state=42,
        scoring="neg_mean_squared_error", n_jobs=1,
    )
    ranks = perm.importances_mean
    return np.sort(np.argsort(-ranks, kind="stable")[:k]), ranks, "Permutation 重要度來自 Training 內部保留段；相關欄位的重要度可能被稀釋。"


def make_model(name: str):
    if name == "Ridge":
        return Ridge(alpha=10.0)
    if name == "Lasso":
        return Lasso(alpha=0.1, max_iter=20000)
    return LinearRegression()


def vif_table(x: np.ndarray, names: list[str]) -> pd.DataFrame:
    with_const = sm.add_constant(x, has_constant="add")
    values = []
    for i in range(len(names)):
        try:
            value = float(variance_inflation_factor(with_const, i + 1))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            value = np.inf
        values.append(value)
    result = pd.DataFrame({"特徵": names, "VIF": values})
    result["判讀"] = np.select([result.VIF >= 10, result.VIF >= 5], ["高（≥10）", "留意（≥5）"], default="較低")
    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


def run_analysis(frame: pd.DataFrame, target: str, features: list[str], time_col: str | None,
                 split: str, selector: str, k: int, model_name: str, add_lags: bool) -> dict:
    reserved = {"_time", "__target_lag_1", "__target_lag_2", "__target_lag_3"}
    if reserved.intersection(frame.columns):
        raise ValueError("CSV 欄名包含系統保留名稱 _time 或 __target_lag_1–3；請先重新命名。")
    work = frame.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    if not np.isfinite(work[target].to_numpy(dtype=float)).all():
        raise ValueError("Target 含無限大數值。")
    if work[target].nunique() < 2:
        raise ValueError("Target 至少需要兩個不同數值。")
    if time_col:
        parsed = pd.to_datetime(work[time_col], errors="coerce")
        if parsed.isna().any():
            raise ValueError("時間欄有無法解析的值；請換一欄或使用原始列順序。")
        work = work.assign(_time=parsed).sort_values("_time", kind="stable")
    else:
        work = work.assign(_time=np.arange(len(work)))
    work = work.reset_index(drop=True)
    original_rows = len(work)
    lag_rows = 0
    if add_lags:
        if split != CHRONO:
            raise ValueError("Lag 1–3 只能配合時間順序切分，避免隨機切分造成相鄰 Target 洩漏。")
        for lag in (1, 2, 3):
            work[f"__target_lag_{lag}"] = work[target].shift(lag)
        work = work.iloc[3:].reset_index(drop=True)
        lag_rows = 3
    used_features = features + ([f"__target_lag_{lag}" for lag in (1, 2, 3)] if add_lags else [])
    numeric = [
        col for col in used_features
        if pd.api.types.is_numeric_dtype(work[col])
        or pd.to_numeric(work[col], errors="coerce").notna().mean() >= .95
    ]
    for col in used_features:
        if col in numeric:
            work[col] = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            work[col] = work[col].astype("string").fillna("(missing)").astype(str)
    if len(work) < 80:
        raise ValueError("清理後至少需要 80 筆資料，才能切分並進行統計診斷。")
    if split == CHRONO:
        # Preserve the pre-lag split dates. The first three lagless rows are
        # removed from Training only, as in the submitted ARX comparison.
        n1, n2 = int(original_rows * .70) - lag_rows, int(original_rows * .85) - lag_rows
        train, validation, test = work.iloc[:n1], work.iloc[n1:n2], work.iloc[n2:]
    else:
        train, test = train_test_split(work, test_size=.20, random_state=42, shuffle=True)
        validation = None
        train, test = train.sort_index(), test.sort_index()
    categorical = [c for c in used_features if c not in numeric]
    if any(train[c].notna().sum() == 0 for c in numeric):
        raise ValueError("有數值特徵在 Training 中全為空值；請取消選取該欄。")
    if any(train[c].nunique() > 20 for c in categorical):
        raise ValueError("類別特徵超過 20 個類別；請移除高基數欄位或先整理資料。")
    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical))
    prep = ColumnTransformer(transformers, verbose_feature_names_out=False)
    x_train = np.asarray(prep.fit_transform(train[used_features]), dtype=float)
    names = list(prep.get_feature_names_out())
    if x_train.shape[1] > 80:
        raise ValueError("One-hot 後超過 80 個特徵；請減少候選欄位以便互動分析。")
    varying = np.nanstd(x_train, axis=0) > 1e-12
    names = [name for name, keep in zip(names, varying) if keep]
    x_train = x_train[:, varying]
    if x_train.shape[1] == 0:
        raise ValueError("Training 中沒有可用的變動特徵。")
    if k > x_train.shape[1]:
        raise ValueError(f"有效編碼特徵僅 {x_train.shape[1]} 個，k={k} 過大；請降低 k。")
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(np.asarray(prep.transform(test[used_features]), dtype=float)[:, varying])
    x_val = None if validation is None else scaler.transform(np.asarray(prep.transform(validation[used_features]), dtype=float)[:, varying])
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    selected, selection_scores, selection_note = select_columns(x_train, y_train, selector, k, split == CHRONO)
    x_train_sel, x_test_sel = x_train[:, selected], x_test[:, selected]
    selected_names = [names[i] for i in selected]
    model = make_model(model_name).fit(x_train_sel, y_train)
    prediction = np.asarray(model.predict(x_test_sel))
    validation_metrics = None
    if validation is not None:
        validation_metrics = score(validation[target].to_numpy(dtype=float), model.predict(x_val[:, selected]), k)
    # The selected estimator can be Ridge/Lasso. Classical CI/PI belong to an
    # auxiliary OLS fit on the same train-only selected matrix, not to Ridge/Lasso.
    ols = sm.OLS(y_train, sm.add_constant(x_train_sel, has_constant="add")).fit()
    train_residuals = y_train - model.predict(x_train_sel)
    test_residuals = y_test - prediction
    interval_frame = ols.get_prediction(sm.add_constant(x_test_sel, has_constant="add")).summary_frame(alpha=.05)
    pi_covered = (y_test >= interval_frame["obs_ci_lower"].to_numpy()) & (y_test <= interval_frame["obs_ci_upper"].to_numpy())
    if target == "Appliances" and not frame.attrs.get("uploaded_csv", False):
        groups = pd.cut(y_test, bins=[-np.inf, 100, 200, 400, np.inf], labels=["≤100 Wh", "100–200 Wh", "200–400 Wh", ">400 Wh"])
    else:
        groups = pd.qcut(y_test, q=4, duplicates="drop") if np.unique(y_test).size > 4 else pd.Categorical(y_test.astype(str))
        if pd.isna(groups).all():
            groups = pd.Categorical(y_test.astype(str))
    coverage_frame = pd.DataFrame({"群組": groups.astype(str), "涵蓋": pi_covered})
    coverage_table = coverage_frame.groupby("群組", sort=False, observed=True).agg(樣本數=("涵蓋", "size"), PI涵蓋率=("涵蓋", "mean")).reset_index()
    try:
        bp = het_breuschpagan(train_residuals, sm.add_constant(x_train_sel, has_constant="add"))
        bp_p = float(bp[1])
    except (ValueError, np.linalg.LinAlgError):
        bp_p = np.nan
    jb_stat, jb_p, skew, kurtosis = jarque_bera(train_residuals)
    return {
        "data": work, "original_rows": original_rows, "train_n": len(train),
        "validation_n": 0 if validation is None else len(validation), "test_n": len(test),
        "metrics": score(y_test, prediction, k), "validation_metrics": validation_metrics,
        "selected_names": selected_names, "coefficients": np.asarray(model.coef_),
        "selection_scores": selection_scores[selected], "selection_note": selection_note,
        "all_names": names, "all_scores": selection_scores,
        "vif": vif_table(x_train_sel, selected_names), "actual": y_test,
        "prediction": prediction, "intervals": interval_frame,
        "coverage_table": coverage_table,
        "time": test["_time"].reset_index(drop=True), "test_residuals": test_residuals,
        "dw_train": float(durbin_watson(train_residuals)), "dw_test": float(durbin_watson(test_residuals)),
        "bp_train_p": bp_p, "jb_train_p": float(jb_p), "skew_train": float(skew),
        "kurtosis_train": float(kurtosis), "pi_coverage": float(np.mean(pi_covered)),
        "pi_negative_pct": float(np.mean(interval_frame["obs_ci_lower"] < 0)),
        "model_name": model_name, "selector": selector,
        "rank_deficient": np.linalg.matrix_rank(sm.add_constant(x_train_sel, has_constant="add")) < k + 1,
        "split": split, "add_lags": add_lags,
        "preprocessor": prep, "scaler": scaler, "varying": varying,
        "selected_indices": selected, "model": model, "ols": ols,
        "used_features": used_features, "demo_row": test[used_features].iloc[[0]].copy(),
        "target": target,
    }


def show_archived_image(relative_path: str, caption: str):
    path = ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=f"原研究輸出｜{caption}", width="stretch")


def predict_one(result: dict, edited: pd.DataFrame) -> tuple[float, pd.Series]:
    """Predict one hypothetical row with the already fitted train-only pipeline."""
    if len(edited) != 1:
        raise ValueError("單筆實驗室一次只能預測一筆資料。")
    row = edited[result["used_features"]].copy()
    numeric_columns = result["preprocessor"].transformers_[0][2] if result["preprocessor"].transformers_[0][0] == "num" else []
    for column in numeric_columns:
        row[column] = pd.to_numeric(row[column], errors="coerce")
        if not np.isfinite(row[column].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{column} 含無限大數值。")
    encoded = np.asarray(result["preprocessor"].transform(row), dtype=float)[:, result["varying"]]
    selected = result["scaler"].transform(encoded)[:, result["selected_indices"]]
    estimate = float(result["model"].predict(selected)[0])
    ols_interval = result["ols"].get_prediction(sm.add_constant(selected, has_constant="add")).summary_frame(alpha=.05).iloc[0]
    return estimate, ols_interval


st.markdown(
    """<section class="site-hero"><div class="eyebrow">5115056030 / CRISP-DM INTERACTIVE LAB</div>
    <h1>讓每一筆耗電資料，<br>說出<span class="accent">模型背後的故事。</span></h1>
    <p>從研究問題、資料切分到預測區間，沿著報告的證據一步步探索。你也可以上傳 CSV，親手重跑多元線性回歸分析。</p>
    <div class="hero-tags"><span>⚡ 家電能源預測</span><span>◈ 五種特徵選擇</span><span>⌁ 95% CI / PI</span><span>↗ 自訂 CSV 實驗</span></div>
    <a class="hero-cta" href="#guided-reading">開始閱讀六階段導讀 <span aria-hidden="true">↓</span></a></section>""",
    unsafe_allow_html=True,
)

st.markdown("<div class='section-title'><div class='section-eyebrow'>The research at a glance</div><h2>先看三個關鍵數字</h2><p>這些數字來自已保存的原研究 Test，不會隨下方互動實驗改變。</p></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("主模型 A · Test RMSE", "89.20 Wh", "14 個環境特徵")
c2.metric("ARX 對照 B · Test RMSE", "61.40 Wh", "28 個輸入特徵")
c3.metric("ARX 尖峰 >400 Wh · PI 涵蓋率", "24.64%", "69 筆尖峰")
st.markdown("<div class='insight-strip'>關鍵發現｜在同一批對齊 Test 資料上，環境 25 特徵模型 RMSE 88.25 Wh，加入前三筆真實耗電的 ARX 28 特徵為 61.40 Wh；但傳統 95% 預測區間對高負荷尖峰仍大幅失準。主模型 A 是另由 Validation 選定，請勿混成同一選型實驗。</div>", unsafe_allow_html=True)

st.markdown("<div id='guided-reading' class='section-title'><div class='section-eyebrow'>Guided reading</div><h2>用六個決策讀懂完整報告</h2><p>點選階段或使用「上一步／下一步」；每一步都對應原報告的一張實測圖。</p></div>", unsafe_allow_html=True)
components.html(guide_html((ASSETS / "guide.html").stat().st_mtime_ns), height=625, scrolling=True)

with st.expander("📚 開啟報告與研究資料", expanded=False):
    st.markdown("合併 PDF 第 1–17 頁是 CRISP-DM 報告，第 18–50 頁是現有 GPT 對話匯出；**不能據此保證已涵蓋本專案全部對話**，繳交前仍須由本人核對並補上完整原始對話。研究筆記保留階段歷史，早期『尚未完成』敘述不代表最終狀態。")
    st.markdown("<div class='read-path'><span>① 先讀摘要與研究問題</span><span>② 對照六階段圖表</span><span>③ 檢查五項假設與 PI 限制</span><span>④ 核對對話匯出與研究來源</span></div>", unsafe_allow_html=True)
    if REPORT_PDF.is_file():
        st.download_button("⬇ 下載目前合併報告 PDF", REPORT_PDF.read_bytes(), file_name=REPORT_PDF.name, mime="application/pdf", type="primary")
    documents = [
        ("下載正式報告 Markdown", "5115056030_hw2_report.md"),
        ("下載 NotebookLM 摘要", "5115056030_NotebookLM_研究摘要.md"),
        ("下載完整研究筆記", "5115056030_hw2_研究筆記_v6.md"),
    ]
    for column, (label, filename) in zip(st.columns(3), documents):
        path = ROOT / filename
        if path.is_file():
            column.download_button(label, path.read_bytes(), file_name=filename, mime="text/markdown")

st.markdown("<div class='section-title'><div class='section-eyebrow'>Your experiment</div><h2>換你操控資料與模型</h2><p>左側建立實驗；下方五個互動頁面會呈現本次資料的圖表、診斷與單筆估計。</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚡ 回歸實驗控制台")
st.sidebar.caption("設定資料 → 選特徵 → 訓練 → 閱讀診斷")
uploaded = st.sidebar.file_uploader("上傳 CSV（未上傳則載入預設資料）", type=["csv"])
if uploaded is not None:
    raw = uploaded.getvalue()
    source_name = uploaded.name
elif DEFAULT_CSV.is_file():
    raw = DEFAULT_CSV.read_bytes()
    source_name = DEFAULT_CSV.name
else:
    st.info("找不到預設 CSV。請從左側上傳含數值 Target 的資料集。")
    st.stop()
try:
    df = load_csv(raw)
except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"CSV 載入失敗：{exc}")
    st.stop()
df.attrs["uploaded_csv"] = uploaded is not None
if len(df) == 0 or len(df.columns) < 2:
    st.error("CSV 至少需要一筆資料與兩個欄位。")
    st.stop()
st.sidebar.markdown(f"**◉ 目前資料**　`{source_name}`  ")
st.sidebar.caption(f"{len(df):,} 筆觀測 × {len(df.columns)} 欄；未上傳時自動載入預設資料。")
numeric_targets = [c for c in df if pd.to_numeric(df[c], errors="coerce").notna().mean() >= .95]
if not numeric_targets:
    st.error("沒有可用的數值目標欄位；請上傳含數值 Target 的 CSV。")
    st.stop()
target = st.sidebar.selectbox("◎ 預測目標 Target y", numeric_targets, index=numeric_targets.index("Appliances") if "Appliances" in numeric_targets else 0, format_func=lambda value: "🎯  " + value)
time_candidates = [c for c in df.columns if c != target and ("date" in c.lower() or "time" in c.lower())]
time_options = ["依原始列順序"] + time_candidates
time_choice = st.sidebar.selectbox("◷ 時間欄（排序用）", time_options, index=1 if len(time_options) > 1 else 0, format_func=lambda value: "🕒  " + value)
time_col = None if time_choice == time_options[0] else time_choice
candidate = [c for c in df.columns if c not in (target, time_col)]
default_features = [
    c for c in candidate
    if (pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() >= .95)
    and (uploaded is not None or c not in ("rv1", "rv2"))
][:39]
if not default_features:
    default_features = candidate[:min(25, len(candidate))]
with st.sidebar.expander("◈ 候選輸入特徵 X · 點開編輯", expanded=uploaded is not None):
    st.caption("預設資料已排除隨機控制欄 rv1、rv2。特徵數以原始欄位加上選用的 Lag 計算。")
    features = st.multiselect("選擇候選欄位", candidate, default=default_features)
feature_count = len(features)
st.sidebar.caption(f"目前選取 {feature_count} 個原始輸入欄位")
split = st.sidebar.radio("⌁ 資料切分", [CHRONO, RANDOM], format_func=lambda value: ("📈  " if value == CHRONO else "🎲  ") + value)
selector_icons = {"SelectKBest": "✦", "RFE": "◇", "SFS": "↗", "Lasso": "⌁", "Permutation Importance": "◎"}
selector = st.sidebar.radio("◈ 特徵選擇方法", SELECTORS, index=SELECTORS.index("Permutation Importance"), format_func=lambda value: selector_icons[value] + "  " + value)
add_lags = st.sidebar.checkbox("↺ 加入真實歷史耗電 Lag 1–3", value=False)
model_feature_count = feature_count + (3 if add_lags else 0)
if 13 < model_feature_count < 40:
    st.sidebar.success(f"候選模型輸入 {model_feature_count} 個（原始 {feature_count} + Lag {model_feature_count-feature_count}）：符合 13 < Features < 40。")
else:
    st.sidebar.warning(f"候選模型輸入 {model_feature_count} 個：不符合作業 14–39 個的限制；仍可探索。")
k_default = max(1, min(model_feature_count, 28 if add_lags else 14))
if model_feature_count > 1:
    k = st.sidebar.slider("◫ 選取特徵數 k", min_value=1, max_value=model_feature_count, value=k_default)
else:
    k = 1
    st.sidebar.caption("選取特徵數 k = 1（目前候選欄位不足兩個）")
if k < 14 and 13 < model_feature_count < 40:
    st.sidebar.caption("此 k 可用於探索，但作業原實驗使用 14／20／25 個選中特徵；請勿把小於 14 的設定當作原報告重現。")
model_icons = {"LinearRegression": "📐", "Ridge": "🛡️", "Lasso": "✂️"}
model_name = st.sidebar.selectbox("⌘ 回歸模型", MODELS, format_func=lambda value: model_icons[value] + "  " + value)
if add_lags:
    st.sidebar.caption("Lag 使用前 1–3 筆已觀測 Target。須有可信的時間順序；不適用純未來預報。")
    if uploaded is None and feature_count == 25:
        st.sidebar.caption("重現報告 ARX B：保留 25 個環境欄位，k=28，選 LinearRegression；其結果是另列對照，非原 45 組選型勝出者。")
    if split == RANDOM:
        st.sidebar.error("Lag 請搭配時間順序切分。")
elif split == CHRONO and time_col is None:
    st.sidebar.caption("沒有選時間欄時，請確認 CSV 原始列序就是可信的時間順序。")
run = st.sidebar.button("⚡ 執行完整分析", type="primary", width="stretch")

signature = hashlib.sha256(raw + repr((target, features, time_col, split, selector, k, model_name, add_lags)).encode()).hexdigest()
if run:
    if not features:
        st.sidebar.error("請至少選擇一個輸入特徵。")
    else:
        try:
            with st.spinner("正在執行 Train-only 前處理、特徵選擇、模型與診斷…"):
                result = run_analysis(df, target, features, time_col, split, selector, k, model_name, add_lags)
            st.session_state["result"] = result
            st.session_state["signature"] = signature
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.sidebar.error(str(exc))
result = st.session_state.get("result") if st.session_state.get("signature") == signature else None
if result is None:
    st.info("從左側選擇參數並按「執行完整分析」。資料探索可立即查看；其餘頁面會在訓練完成後顯示。")
else:
    st.success(f"本次分析完成｜{result['train_n']:,} 筆 Training → {result['test_n']:,} 筆 Test｜{result['selector']} → {result['model_name']}｜{len(result['selected_names'])} 特徵")

TAB_DATA = "◫ 資料理解"
TAB_SELECTION = "◇ 特徵選擇 · VIF"
TAB_EVAL = "◎ 評估 · 95% 區間"
TAB_ASSUMPTIONS = "⌁ 五項假設"
TAB_PREDICT = "↗ 單筆預測實驗室"
tab_data, tab_selection, tab_eval, tab_assumptions, tab_predict = st.tabs(
    [TAB_DATA, TAB_SELECTION, TAB_EVAL, TAB_ASSUMPTIONS, TAB_PREDICT],
    default=st.session_state.get("active_tab", TAB_DATA),
)

with tab_data:
    st.subheader("01 / Data Understanding｜先看資料的樣貌")
    st.caption(f"目前資料：{source_name}。這裡的圖與統計依當前 CSV 即時產生；和原研究固定成果分開。")
    a, b, c, d = st.columns(4)
    a.metric("觀測筆數", f"{len(df):,}")
    b.metric("原始欄位", len(df.columns))
    c.metric("空值儲存格", f"{int(df.isna().sum().sum()):,}")
    d.metric("候選 X 特徵", len(features))
    target_values = pd.to_numeric(df[target], errors="coerce").dropna()
    q1, median, q3 = target_values.quantile([.25, .5, .75])
    st.markdown(f"<div class='insight-strip'>Target 分布｜中位數 <b>{median:,.2f}</b>、四分位範圍 <b>{q1:,.2f}–{q3:,.2f}</b>。右尾與尖峰會影響 RMSE；請搭配直方圖判讀。</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Histogram(x=target_values, nbinsx=44, marker_color="#42d9c3", opacity=.84, hovertemplate="數值 %{x}<br>筆數 %{y}<extra></extra>"))
    fig.add_vline(x=float(median), line_color="#efb66c", line_dash="dash", annotation_text="中位數")
    fig.update_layout(xaxis_title=target, yaxis_title="筆數", bargap=.05)
    st.plotly_chart(chart_style(fig, "Target 分布 · 可放大與滑鼠探索", 375), config={"displaylogo": False}, key="target_hist")
    corr_cols = [c for c in features if pd.to_numeric(df[c], errors="coerce").notna().mean() >= .95][:30]
    if corr_cols:
        corr = df[corr_cols + [target]].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
        fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=corr.columns, y=corr.index, colorscale="RdBu", zmin=-1, zmax=1, reversescale=True, hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
        fig.update_layout(xaxis={"tickangle": -45}, yaxis={"autorange": "reversed"})
        st.plotly_chart(chart_style(fig, "Pearson 相關矩陣 · 僅作探索", min(740, max(400, len(corr) * 22))), config={"displaylogo": False}, key="pearson_heatmap")
        top_corr = corr[target].drop(target).dropna().reindex(corr[target].drop(target).abs().sort_values(ascending=False).index).head(8)
        st.dataframe(pd.DataFrame({"特徵": top_corr.index, "與 Target 的 Pearson r": top_corr.values}), width="stretch", hide_index=True)
    else:
        st.info("目前沒有可畫 Pearson 熱力圖的數值候選特徵。")
    with st.expander("查看資料前 12 筆與描述統計"):
        st.dataframe(df.head(12), width="stretch")
        st.dataframe(df.describe(include="all").transpose().head(40), width="stretch")
    if uploaded is None:
        with st.expander("🗂️ 對照原研究的 Data Understanding 圖"):
            show_archived_image("5115056030_step2_outputs/01_target_histogram.png", "Appliances 分布")
            show_archived_image("5115056030_step2_outputs/04_correlation_heatmap.png", "原研究 Pearson 熱力圖")

with tab_selection:
    st.subheader("02 / Feature Selection｜哪些欄位留下來？")
    if result is not None:
        st.caption(result["selection_note"])
        importance = pd.DataFrame({"特徵": result["selected_names"], "標準化係數": result["coefficients"]})
        importance["係數絕對值"] = importance["標準化係數"].abs()
        importance = importance.sort_values("係數絕對值", ascending=True)
        fig = go.Figure(go.Bar(x=importance["標準化係數"], y=importance["特徵"], orientation="h", marker_color=np.where(importance["標準化係數"] >= 0, "#43dcc6", "#f0b671"), hovertemplate="%{y}<br>標準化係數 %{x:.3f}<extra></extra>"))
        fig.update_layout(xaxis_title="標準化模型係數", yaxis_title="")
        st.plotly_chart(chart_style(fig, f"{selector} · 選中 {len(importance)} 個欄位", max(360, len(importance) * 28 + 100)), config={"displaylogo": False}, key="coefficient_chart")
        st.caption("權重是模型在標準化欄位上的係數，不是因果影響；相關特徵會使係數不穩。")
        scores = pd.DataFrame({"候選特徵": result["all_names"], "選擇器分數": result["all_scores"]})
        scores["入選"] = scores["候選特徵"].isin(result["selected_names"])
        with st.expander("查看全部候選欄位的選擇分數"):
            st.dataframe(scores.sort_values("選擇器分數", ascending=False), width="stretch", hide_index=True)
        vif_frame = result["vif"].copy()
        vif_frame["VIF"] = vif_frame["VIF"].replace(np.inf, 9999)
        fig = go.Figure(go.Bar(x=vif_frame["VIF"], y=vif_frame["特徵"], orientation="h", marker_color=np.where(vif_frame["VIF"] >= 10, "#ef987e", "#59cfc0"), hovertemplate="%{y}<br>VIF %{x:.2f}<extra></extra>"))
        fig.add_vline(x=10, line_dash="dash", line_color="#f0b671", annotation_text="高共線性門檻 10")
        fig.update_layout(xaxis_title="VIF（無限大以 9999 顯示）", yaxis_title="", yaxis={"autorange": "reversed"})
        st.plotly_chart(chart_style(fig, "Training 多重共線性檢查", max(360, len(vif_frame) * 27 + 100)), config={"displaylogo": False}, key="vif_chart")
        st.dataframe(result["vif"], width="stretch", hide_index=True)
        st.caption("VIF 使用 Training 的選定特徵計算；≥5 留意，≥10 高共線性。常數及完全重複欄位可能得到無限大。")
        if result["rank_deficient"]:
            st.warning("選定設計矩陣秩不足；係數、VIF 與傳統 OLS 區間需格外謹慎。")
    if uploaded is None:
        with st.expander("原研究的 45 組模型比較與 VIF 圖"):
            st.caption("原研究 45 筆設定中，k=25 的五種選擇器保留相同全部特徵，並非 45 個獨立子集合。")
            show_archived_image("5115056030_step4_outputs/01_validation_top16_rmse.png", "Validation RMSE 前 16 組")
            show_archived_image("5115056030_step4_outputs/02_vif_gt10_heatmap.png", "VIF > 10")

with tab_eval:
    st.subheader("03 / Evaluation｜點預測與不確定性")
    if result is not None:
        st.caption(f"本次 Test：{result['test_n']:,} 筆；Train：{result['train_n']:,} 筆；Validation：{result['validation_n']:,} 筆。切分方式：{result['split']}")
        cols = st.columns(5)
        for col, label in zip(cols, ("MAE", "MSE", "RMSE", "R²", "Adjusted R²")):
            col.metric(label, f"{result['metrics'][label]:,.4f}")
        if result["validation_metrics"] is not None:
            st.caption(f"Validation RMSE {result['validation_metrics']['RMSE']:.2f}；本次參數由使用者指定，沒有根據 Test 自動調參。")
        cov_a, cov_b = st.columns(2)
        cov_a.metric("傳統 OLS 95% PI · Test 涵蓋率", f"{result['pi_coverage']:.2%}")
        cov_b.metric("PI 下界為負的比例", f"{result['pi_negative_pct']:.2%}")
        st.caption("CI 是條件平均值區間；PI 是單筆觀測區間。這裡的區間都由 Training 上的輔助 OLS 計算。")
        if model_name != "LinearRegression":
            st.warning("所選 Ridge/Lasso 的指標卡使用該模型；下方 95% CI/PI 與區間圖則來自相同選定特徵上的輔助 OLS，不能解讀為 Ridge/Lasso 本身的區間。")
        else:
            st.info("區間由 Training 上的 statsmodels OLS 計算；模型係數與 sklearn LinearRegression 數值等價。")
        if result["rank_deficient"]:
            st.warning("設計矩陣秩不足，傳統 OLS 區間不宜作可靠推論。")
        window = st.slider("顯示連續測試點數（預設 432；原資料為 72 小時）", 1, min(result["test_n"], 1000), min(432, result["test_n"]), step=1, on_change=lambda: st.session_state.update(active_tab=TAB_EVAL))
        if window < result["test_n"]:
            start = st.slider("視窗起點（測試集索引）", 0, result["test_n"] - window, 0, on_change=lambda: st.session_state.update(active_tab=TAB_EVAL))
        else:
            start = 0
            st.caption("目前顯示全部測試點；縮小視窗後即可移動起點。")
        end = start + window
        xs = np.arange(start, end)
        intervals = result["intervals"].iloc[start:end]
        fig = go.Figure()
        for lower, upper, fill, name in [
            ("obs_ci_lower", "obs_ci_upper", "rgba(240,182,113,.22)", "95% PI · OLS"),
            ("mean_ci_lower", "mean_ci_upper", "rgba(77,175,217,.30)", "95% mean CI · OLS"),
        ]:
            fig.add_trace(go.Scatter(x=xs, y=intervals[lower], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xs, y=intervals[upper], mode="lines", line={"width": 0}, fill="tonexty", fillcolor=fill, name=name, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=result["actual"][start:end], name="實測 Actual", line={"color": "#f5f8fa", "width": 1.5}, hovertemplate="Test #%{x}<br>實測 %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=intervals["mean"], name="OLS 點估計", line={"color": "#42d9c3", "width": 2}, hovertemplate="Test #%{x}<br>OLS %{y:.2f}<extra></extra>"))
        if model_name != "LinearRegression":
            fig.add_trace(go.Scatter(x=xs, y=result["prediction"][start:end], name=f"{model_name} 點估計", line={"color": "#d59ce2", "width": 1.5}))
        fig.add_hline(y=0, line_color="rgba(230,240,242,.4)", line_width=1)
        fig.update_layout(xaxis_title="Test 索引（時序切分時為連續時間）", yaxis_title=target, hovermode="x unified")
        st.plotly_chart(chart_style(fig, "拖曳放大、移動視窗，檢查區間是否覆蓋尖峰", 530), config={"displaylogo": False}, key="interval_chart")
        if split == RANDOM:
            st.caption("隨機切分的測試列按原 CSV 索引顯示；相鄰點不代表連續時間。")
        st.markdown("#### 分組涵蓋率｜總平均是否掩蓋局部問題？")
        coverage = result["coverage_table"]
        fig = go.Figure(go.Bar(x=coverage["群組"], y=coverage["PI涵蓋率"] * 100, marker_color=["#ef987e" if value < .9 else "#43dcc6" for value in coverage["PI涵蓋率"]], text=[f"{value:.1%} · n={n}" for value, n in zip(coverage["PI涵蓋率"], coverage["樣本數"])], textposition="outside", hovertemplate="%{x}<br>涵蓋率 %{y:.1f}%<extra></extra>"))
        fig.add_hline(y=95, line_dash="dash", line_color="#f0b671", annotation_text="名目 95%")
        fig.update_layout(yaxis_title="Test PI 實測涵蓋率 (%)", yaxis_range=[0, 110])
        st.plotly_chart(chart_style(fig, "傳統 95% PI · 依實際 Target 分組", 360), config={"displaylogo": False}, key="coverage_chart")
        st.dataframe(coverage.style.format({"PI涵蓋率": "{:.2%}"}), width="stretch", hide_index=True)
        download = pd.DataFrame({
            "test_index": np.arange(result["test_n"]), "time_or_row": result["time"].astype(str),
            "actual": result["actual"], "model_prediction": result["prediction"],
            "ols_prediction": result["intervals"]["mean"].to_numpy(),
            "mean_ci_lower": result["intervals"]["mean_ci_lower"].to_numpy(),
            "mean_ci_upper": result["intervals"]["mean_ci_upper"].to_numpy(),
            "pi_lower": result["intervals"]["obs_ci_lower"].to_numpy(),
            "pi_upper": result["intervals"]["obs_ci_upper"].to_numpy(),
        })
        st.download_button("⬇ 下載本次 Test 預測與區間 CSV", download.to_csv(index=False).encode("utf-8-sig"), file_name="interactive_test_predictions.csv", mime="text/csv", on_click="ignore")
        st.warning("傳統 OLS 區間依賴模型形式、等變異、殘差結構等假設。即使整體涵蓋率接近 95%，也不能推論每個負荷區間都已校準。")
    if uploaded is None:
        with st.expander("原研究的 72 小時圖與分負荷涵蓋率"):
            show_archived_image("5115056030_step5_outputs/02_intervals_B_lag28_72h_zoom.png", "ARX 模型前 432 筆")
            show_archived_image("5115056030_step5_outputs/05_PI_coverage_by_load.png", "PI 分負荷涵蓋率")

with tab_assumptions:
    st.subheader("04 / 五項回歸假設：模型可信度的體檢")
    if result is not None:
        st.markdown("線性形式、獨立性、等變異、常態與多重共線性，要一起看；單一 p 值不能宣告模型全面通過。")
        a, b, c = st.columns(3)
        a.metric("Train Durbin–Watson", f"{result['dw_train']:.3f}")
        b.metric("Test Durbin–Watson", f"{result['dw_test']:.3f}")
        c.metric("Train BP p-value", f"{result['bp_train_p']:.3g}")
        st.caption(f"Train JB p-value = {result['jb_train_p']:.3g}；偏態 = {result['skew_train']:.2f}；超額峰度 = {result['kurtosis_train'] - 3:.2f}。p-value 顯示 0 時通常是數值下溢。")
        fig = go.Figure(go.Scattergl(x=result["prediction"], y=result["test_residuals"], mode="markers", marker={"size": 5, "color": "#43dcc6", "opacity": .38}, hovertemplate="擬合值 %{x:.2f}<br>殘差 %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0, line_dash="dash", line_color="#f0b671")
        fig.update_layout(xaxis_title="模型預測值", yaxis_title="Test 殘差")
        st.plotly_chart(chart_style(fig, "殘差對預測值｜觀察曲線與漏斗形", 400), config={"displaylogo": False}, key="residual_scatter")
        theoretical, ordered = stats.probplot(result["test_residuals"], dist="norm", fit=False)
        slope, intercept, _ = stats.probplot(result["test_residuals"], dist="norm", fit=True)[1]
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=theoretical, y=ordered, mode="markers", name="Test 殘差分位數", marker={"size": 5, "color": "#d69ee5", "opacity": .5}))
        fig.add_trace(go.Scatter(x=[min(theoretical), max(theoretical)], y=[slope * min(theoretical) + intercept, slope * max(theoretical) + intercept], mode="lines", name="參考線", line={"color": "#f0b671", "dash": "dash"}))
        fig.update_layout(xaxis_title="常態理論分位數", yaxis_title="實測殘差分位數")
        st.plotly_chart(chart_style(fig, "常態 Q–Q｜尾端是否偏離參考線", 400), config={"displaylogo": False}, key="qq_plot")
        st.caption("BP/JB 統計使用 Training 殘差；Test 圖用於外部診斷。DW 接近 2 只表示一階自相關較少，不能證明所有延遲獨立。")
        with st.expander("查看所有選中特徵的 Training VIF"):
            st.dataframe(result["vif"], width="stretch", hide_index=True)
    if uploaded is None:
        with st.expander("原研究的五項診斷"):
            show_archived_image("5115056030_step5_outputs/01_diagnostics_A_best14.png", "主模型 A")
            show_archived_image("5115056030_step5_outputs/01_diagnostics_B_lag28.png", "ARX 對照 B")

with tab_predict:
    st.subheader("05 / 單筆預測實驗室：改變輸入，立即觀察結果")
    if result is None:
        st.info("請先在左側完成訓練，才可打開單筆實驗室。")
    else:
        st.markdown("下表以保留測試集的第一筆輸入作為起點。修改數值後按下按鈕，會沿用目前模型與 Training 時建立的轉換器；**不會重新訓練**。")
        if result["add_lags"]:
            st.warning("Lag 1–3 代表預測時已知的前三筆真實 Target。若你在此修改它們，結果只是情境模擬，不等於可用預測值遞迴推估的效果。")
        edited = st.data_editor(result["demo_row"].reset_index(drop=True), num_rows="fixed", width="stretch", hide_index=True, key="single_row_editor", on_change=lambda: st.session_state.update(active_tab=TAB_PREDICT))
        st.caption("缺值會以 Training 的中位數補齊；類別欄位由訓練時的 One-hot 編碼器處理。建議在原資料合理範圍內試驗。")
        if st.button("⚡ 估計這筆樣本", type="primary", key="predict_single", on_click=lambda: st.session_state.update(active_tab=TAB_PREDICT)):
            try:
                point, interval = predict_one(result, edited)
            except (ValueError, KeyError) as exc:
                st.error(f"單筆預測失敗：{exc}")
            else:
                a, b, c = st.columns(3)
                a.metric(f"{result['model_name']} 點預測", f"{point:,.2f}")
                b.metric("OLS 95% 平均值 CI", f"{interval['mean_ci_lower']:,.2f} ～ {interval['mean_ci_upper']:,.2f}")
                c.metric("OLS 95% 單筆 PI", f"{interval['obs_ci_lower']:,.2f} ～ {interval['obs_ci_upper']:,.2f}")
                if result["model_name"] != "LinearRegression":
                    st.warning("右側 CI/PI 來自同一選中特徵上的輔助 OLS，不是 Ridge/Lasso 點預測的區間。")
                st.caption("傳統 OLS 區間仰賴模型假設；對高耗電尖峰的涵蓋率尤其可能不足。區間下界若為負，不代表負耗電具有物理意義。")

st.divider()
st.caption("資料與原研究成果：UCI / Kaggle Appliances Energy Prediction；完整研究限制、文獻比較與 AI 協作說明請閱讀上方 PDF。此互動頁的重新訓練結果不會改寫原報告。")
