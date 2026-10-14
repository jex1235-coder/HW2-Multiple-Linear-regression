# HW2-Multiple-Linear-regression

家用電器能源分析｜Streamlit 互動平台

[GitHub 專案](https://github.com/jex1235-coder/HW2-Multiple-Linear-regression) · [自動測試](https://github.com/jex1235-coder/HW2-Multiple-Linear-regression/actions/workflows/tests.yml)

這個網站是 [正式報告](5115056030_hw2_完整報告.pdf) 的互動閱讀入口。首頁有動畫式 CRISP-DM 六階段導讀、重點數字與原研究圖表；下方提供圖示化控制台與五個分析頁面。讀者可使用預設 `KAG_energydata_complete.csv`，或上傳自己的 CSV，重新執行特徵選擇、線性回歸、VIF、殘差檢查與傳統 95% CI／PI，也可編輯單筆輸入做情境預測。網站將原研究的固定成果與訪客當次分析結果分開標示。

## 專案架構

```text
5115056030_hw2/
├── app.py                         # Streamlit 互動平台
├── assets/theme.css                # 深色響應式視覺樣式
├── assets/guide.html               # 六階段動畫導讀元件
├── .streamlit/config.toml          # 預設色彩主題
├── requirements.txt               # Streamlit Cloud Python 依賴
├── requirements-dev.txt           # 本機測試用的 pytest / Playwright
├── .github/workflows/keep_alive.yml
├── KAG_energydata_complete.csv    # 預設資料：19,735 筆
├── 5115056030_hw2.py              # 原作業六階段重現程式
├── 5115056030_hw2_完整報告.pdf     # 正式報告 + 目前可得的 GPT 對話匯出
├── 5115056030_hw2_report.md       # 正式報告文字
├── 5115056030_hw2_研究筆記_v6.md   # 各階段歷史研究紀錄
├── 5115056030_NotebookLM_研究摘要.md
├── tests/test_app.py              # 模型重現與網站功能回歸測試
├── tests/test_browser.py          # 桌機 / 手機實際瀏覽器互動測試
└── 5115056030_step{2..6}_outputs/ # 實測圖、表、模型與部署成果
```

原始報告的主模型是 Training 內用 Permutation Importance 選 14 個特徵、Validation 選定的 `LinearRegression`，Test RMSE 為 **89.20 Wh**、R² 為 **0.0366**。加入 3 個真實歷史耗電值的 28 特徵 ARX 是另行預定的對照，Test RMSE **61.40 Wh**、R² **0.5435**。其整體傳統 OLS 95% PI 涵蓋率為 **95.47%**，但對實際耗電超過 400 Wh 的 69 筆僅涵蓋 **24.64%**；不可把整體涵蓋率當作尖峰時的保證。這些數值來自現有 `step4_outputs`、`step5_outputs`，不是網站每次重訓後必然重現的指標。

## CRISP-DM 六階段

1. **Business Understanding：**以同一 10 分鐘時段的環境、氣象與照明量測估計家電耗電；這是同期估計，並非尚未發生時段的預報。
2. **Data Understanding：**檢查筆數、缺失、目標分布與 Pearson 相關。預設資料 19,735 筆、29 欄，排除日期和目標後有 27 個數值輸入。
3. **Data Preparation：**原研究移除已知隨機控制欄 `rv1`、`rv2`，保留 25 個候選特徵，依時間做 70/15/15 切分。網站對自訂資料可選時間切分或隨機 80/20；缺失數值以 Training 中位數補值，類別以 Training 內 one-hot 編碼，標準化亦只在 Training 擬合。
4. **Modeling：**可選 SelectKBest、RFE、SFS、Lasso、Permutation Importance，選定 k，再訓練 LinearRegression、Ridge 或 Lasso。SFS 為維持網站反應速度，最多使用 Training 均勻抽取的 1,200 筆選特徵，最後模型仍使用全部 Training；因此未必與原作業完整批次結果一致。
5. **Evaluation：**顯示 Test MAE、MSE、RMSE、R²、Adjusted R²，Training VIF、Train/Test Durbin–Watson、Training Breusch–Pagan/Jarque–Bera、Test 殘差圖與 Q–Q 圖。95% CI 是條件平均區間，95% PI 是單筆區間；兩者由**相同入選特徵的輔助 OLS**在 Training 上求得。若選 Ridge/Lasso，區間圖明示為輔助 OLS，不宣稱是正則化模型自身的區間。
6. **Deployment：**本網站提供互動重訓與報告導覽；原作業另以可信來源的 `joblib` Pipeline 示範載入推論。Lag 1–3 必須來自前幾筆已觀測的真實 Target，只能配合可信順序的時間切分。

## 本地執行

建議 Python 3.12。於本資料夾執行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

開啟終端顯示的本地網址，例如 `http://localhost:8501`。未上傳 CSV 時會自動載入同目錄的 `KAG_energydata_complete.csv`。自訂 CSV 需至少 80 筆資料及可轉成數值的 Target；可選數值與低基數類別欄。時間欄只用於排序。選取 14–39 個**原始候選欄**可符合作業輸入特徵數限制；one-hot 後的有效欄位數另計。類別欄超過 20 類、編碼後超過 80 欄，網站會要求先精簡資料。若資料沒有時間欄，時間切分依原始 CSV 列序進行。

互動網站為教學分析工具。現有研究筆記保留了早期「尚未完成」的歷史進度描述，請以正式報告與對應實測輸出判讀最終結果。原研究完整 45 組批次比較與部署模型，仍以 `5115056030_hw2.py` 及保存的輸出為準。

**繳交前須人工核對：**目前合併 PDF 共 50 頁，第 1–17 頁為報告、第 18–50 頁為現有 GPT 對話匯出。後者末頁仍提示須另行匯出原始全程對話，因此網站不宣稱現有附件已滿足「所有 GPT 對話」規定。請由本人核對本次後續對話並補交完整匯出；網站也不能代替學號命名 ZIP 或實際 GitHub／Streamlit Cloud 連結。

網站的 Test 圖表可用滑桿縮放連續觀測窗格，並可下載本次預測、CI/PI 結果。單筆預測實驗室沿用當次已訓練的模型與前處理器，不重新訓練；若模型是 Ridge 或 Lasso，顯示的區間仍是輔助 OLS 的區間。原研究 ARX 的尖峰涵蓋率不足，因此單筆區間不可當成實務風險保證。

## 測試與審核

於本資料夾執行 `python -m pytest tests/test_app.py -q`。測試涵蓋預設主模型與 ARX 對照數字、五種特徵選擇、自訂資料隨機切分、Ridge、Lag 限制、保留欄名防護、Streamlit 頁面渲染與單筆預測。若要重跑實際瀏覽器測試，先安裝 `requirements-dev.txt`、啟動 `streamlit run app.py`，再執行 `python -m pytest tests/test_browser.py -q`；目前測試會使用 Windows 的 Google Chrome，檢查動畫導讀、手機寬度、實際訓練及頁籤在互動後是否保持可見。線上 Demo 與 Keep-Alive 仍須部署後才可實測；不能以本地 HTTP 成功替代線上驗證。

## 部署到 Streamlit Community Cloud

本儲存庫的部署設定：Repository `jex1235-coder/HW2-Multiple-Linear-regression`、Branch `main`、Main file path `app.py`；Advanced settings 選 Python `3.12`。到 [Streamlit 部署頁](https://share.streamlit.io/deploy) 登入 GitHub 後填入這些設定。

1. 將本資料夾內容推送到你有管理權限的 GitHub repository。保留 `app.py`、`requirements.txt`、預設 CSV、正式報告 PDF 及各階段圖片。如果程式檔放在 repository 的子資料夾，部署時要選對 entrypoint 路徑。
2. 到 [Streamlit Community Cloud](https://share.streamlit.io/) 建立 App，選 repository、branch 與 `app.py`。上線後取得 `https://你的名稱.streamlit.app/`。
3. 喚醒腳本已預設使用下方公開 Demo 網址，不需建立 Secret 即可執行。若日後換網址，可在 GitHub repository 的 **Settings → Secrets and variables → Actions → New repository secret** 建立 `STREAMLIT_APP_URL` 覆寫它，值為完整的 `https://…streamlit.app/` 網址。排程瀏覽器沒有登入狀態，因此 App 需允許它直接瀏覽。
4. 在 GitHub **Actions → Visit Streamlit app → Run workflow** 手動測試一次；檢查紀錄顯示 App 已渲染、觀測到 WebSocket。

**線上 Demo：**[開啟互動式多元線性回歸平台](https://jex1235-coder-hw2-multiple-linear-regression-app-0qudfk.streamlit.app/)。

## GitHub Actions Keep-Alive

`.github/workflows/keep_alive.yml` 每 4 小時（台灣時間 00:17、04:17、08:17、12:17、16:17、20:17）用 headless Chromium 訪問 App，等待初始載入；若出現 **“Yes, get this app back up!”** 就點擊，再停留 10 秒並檢查 Streamlit 頁面、WebSocket 及應用錯誤。也可用 `workflow_dispatch` 手動執行；修改此工作流程並推送 `main` 時會立即驗證。此工作流程會訪問網址，但不能保證雲端服務永遠不休眠：GitHub 的排程可能延遲或漏跑；公開 repository 60 天沒有活動時，排程可能被停用；Streamlit Community Cloud 仍依平台規則管理休眠。請以 Actions 執行紀錄和實際 App 可用性為準。

相關平台說明：[Streamlit App 休眠與喚醒](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation)、[GitHub 排程事件](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)。

## 研究來源與限制

資料來源：[Kaggle](https://www.kaggle.com/datasets/loveall/appliances-energy-prediction)、[UCI](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)；原始文獻：Candanedo et al. (2017), [DOI](https://doi.org/10.1016/j.enbuild.2017.01.083)。原論文與本作業使用不同資料切分，文獻數字不能作同場排名。傳統 OLS 區間的獨立、同變異、常態等前提在本資料有明顯問題；不能把顯示的 PI 當成安全告警的校準保證。
