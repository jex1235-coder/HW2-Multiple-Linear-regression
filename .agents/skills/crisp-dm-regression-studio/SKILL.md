---
name: crisp-dm-regression-studio
description: >-
  Standard operating procedure and toolkit for end-to-end CRISP-DM multiple linear regression,
  five-algorithm feature selection (Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Best Subset),
  OLS statistical inference (t-test, p-values, 95% CIs, VIF collinearity), bias-variance trade-off,
  residual diagnostics, and automated deployment of interactive Vue 3 + Chart.js web dashboards.
  Use this skill whenever the user asks to analyze a tabular regression dataset, perform feature selection,
  determine the optimal parsimonious model (Occam's razor / sweet spot), diagnose overfitting, or build
  an interactive regression explorer dashboard.
---

# CRISP-DM Regression & Feature Selection Studio Skill

This skill provides an automated, scientifically rigorous workflow to take any tabular dataset from raw data to an end-to-end CRISP-DM regression analysis, comprehensive 5-algorithm feature selection evaluation, and an interactive modern web dashboard.

---

## 🧭 Workflow Overview (CRISP-DM 6 Phases)

```text
1. Business & Data Understanding  -->  EDA, Pearson correlation heatmap, target variable identification
2. Data Preparation               -->  Missing value handling, dummy variable encoding (One-Hot Encoding, drop_first=True)
3. 5-Algorithm Feature Selection   -->  Lasso (L1), Forward SFS, RFE, ElasticNet, Exhaustive Best Subset (k = 1 to min(p, 5))
4. Model Evaluation & Parsimony   -->  Test RMSE, Test R², Adjusted R², F-statistic, Occam's Razor Sweet Spot determination
5. Statistical Inference          -->  OLS beta coefficients, standard errors, t-statistics, p-values, 95% CIs, VIF collinearity
6. Deployment & Deliverables      -->  Standalone algorithm plots (algo_*.png), model_data.json/js, interactive index.html dashboard
```

---

## 🛠️ Reusable Helper Scripts

The skill provides an automated pipeline in `scripts/`:

### 1. `scripts/auto_regression_pipeline.py`
Automates the entire pipeline for any CSV dataset:
```bash
python scripts/auto_regression_pipeline.py --data path/to/dataset.csv --target Profit --output-dir ./output
```

**Parameters**:
- `--data`: Path or URL to the input CSV file.
- `--target`: Name of the target variable column (e.g., `Profit`, `Sales`, `Price`).
- `--test-size`: Proportion of test set (default: `0.2`).
- `--random-state`: Random seed for reproducibility (default: `42`).
- `--output-dir`: Output folder for figures, CSVs, JSON, and `index.html`.

---

## 🔬 Five Feature Selection Algorithms

| 演算法 | 原理策略 | 適用情境與優勢 |
|---|---|---|
| **Lasso (L1)** | $L_1$ 正則化懲罰項將不重要特徵權重壓縮至 0 | 高維稀疏特徵篩選、避免過擬合 |
| **Forward SFS** | 貪婪逐步向前（從 0 個變數開始每步挑選改善最大的特徵） | 直觀展現邊際效益反轉點 |
| **RFE** | 反向遞迴消除（由全特徵開始逐步剃除權重最小特徵） | 保留具強大聯合預測力的特徵子集 |
| **ElasticNet** | 結合 $L_1$ 稀疏性與 $L_2$ 分組效應 | 當特徵間存在中高度多重共線性時最穩健 |
| **Exhaustive** | 窮舉 $\binom{p}{k}$ 所有可能子集合 | 尋求給定特徵數 $k$ 下全域保證的最佳模型 |

---

## 📊 核心評判準則：奧坎剃刀與甜蜜點判定

1. **Optimal Sweet Spot ($k^*$ 最優解)**：
   - 測試集 RMSE 達到全域最低點（或與更複雜模型差異在 1 個標準誤以內）。
   - 變數 $p$-value 全部達統計顯著水準 ($p < 0.05$ 或具備強烈領域邏輯)。
   - 多重共線性指標 $\text{VIF} < 5.0$。
2. **Overfitting Cliff (過度擬合懸崖)**：
   - 當 $k$ 超過甜蜜點後，測試集 RMSE 逆勢反彈上升。
   - 懲罰冗餘參數的 Adjusted $R^2$ 明顯跌落。
   - 額外加入的變數 $p$-value 偏高（例如 $> 0.50$），自由度被無謂稀釋。

---

## 📚 參考文獻與深入手冊

- [奧坎剃刀原則與 VIF 診斷準則](./references/parsimony_and_vif_guide.md)
- [五大特徵選擇演算法數學原理](./references/feature_selection_algorithms.md)
- [互動儀表板模板與前端架構](./resources/dashboard_template.html)
