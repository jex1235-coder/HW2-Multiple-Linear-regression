# 五大特徵選擇演算法數學原理與選型手冊

## 1. Lasso (L1 Regularization)
目標函數加入 $L_1$ 範數懲罰：
$$\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \|\beta\|_1 \right\}$$
- **幾何特性**：$L_1$ 限制條件為菱形（多胞形），其頂點恰好落在座標軸上，促使次要特徵的迴歸係數嚴格歸零。
- **特徵重要度排序**：依據係數在正則化路徑中被壓縮至 0 的順序或其絕對值大小進行排序。

## 2. Forward Sequential Feature Selection (Forward SFS)
- **策略**：貪婪搜尋（Greedy Search）。
- **流程**：
  1. 初始化子集合 $S = \emptyset$。
  2. 每一輪遍歷所有候選特徵，挑選加入後使模型交叉驗證評估分數（如 $R^2$ 或 $-\text{MSE}$）提升幅度最大者。
  3. 重複加入直到達到目標特徵數 $k$。
- **優點**：計算速度快，直觀呈現每一變數的邊際增益。

## 3. Recursive Feature Elimination (RFE)
- **策略**：反向剃除（Backward Elimination）。
- **流程**：
  1. 以全部候選變數擬合完整模型。
  2. 依係數絕對值 $|\beta_j|$ 或權重排名，剃除權重最小的特徵。
  3. 以剩餘特徵重新擬合模型，遞迴重複直到剩下 $k$ 個特徵。
- **優點**：考慮了特徵間的交互抑制與多重關聯，不易遺漏協同特徵。

## 4. ElasticNet (L1 + L2 Hybrid)
目標函數同時結合 $L_1$ 與 $L_2$ 正則化：
$$\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2} \|\beta\|_2^2 \right\}$$
- **分組效應 (Grouping Effect)**：當多個特徵高度共線性時，Lasso 傾向於隨機選取其中一個，而 ElasticNet 則能同時保留或平滑收縮這些共線特徵。

## 5. Exhaustive Search (Best Subset Selection)
- **策略**：全域窮舉。
- **流程**：
  對指定之特徵數 $k$，計算全部組合數 $\binom{p}{k}$，評估所有可能子模型的測試集誤差，挑選指標最佳者。
- **優點**：數學上絕對保證找到給定 $k$ 下的全域最優解。
- **計算複雜度**：$\mathcal{O}(2^p)$，適合 $p \le 20$ 的特徵維度。
