#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CRISP-DM Automated Regression & Feature Selection Pipeline
Accepts any tabular CSV and produces 5-algorithm comparisons, dual-axis plots,
model exports (JSON/JS), and an interactive modern web dashboard.
"""

import os
import sys
import argparse
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_pipeline(csv_path, target_col, output_dir, test_size=0.2, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[*] Raw shape: {df.shape}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}")

    # Preprocessing & Dummy Encoding
    y = df[target_col].copy()
    X_raw = df.drop(columns=[target_col]).copy()
    
    # Identify categorical columns
    cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        print(f"[*] One-Hot Encoding categorical columns: {cat_cols}")
        X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    else:
        X = X_raw.select_dtypes(include=[np.number]).copy()

    feature_names = X.columns.tolist()
    total_features = len(feature_names)
    max_k = min(total_features, 5)
    print(f"[*] Processed features ({total_features}): {feature_names}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {
        'Lasso': [],
        'Forward SFS': [],
        'RFE': [],
        'ElasticNet': [],
        'Exhaustive': []
    }

    # 1. Lasso
    lasso = Lasso(alpha=100.0, random_state=random_state, max_iter=10000)
    lasso.fit(X_train, y_train)
    coef_abs = np.abs(lasso.coef_)
    lasso_ranking = [feature_names[i] for i in np.argsort(-coef_abs)]
    for k in range(1, max_k + 1):
        feats = lasso_ranking[:k]
        results['Lasso'].append(evaluate_subset(X_train, X_test, y_train, y_test, feats, k))

    # 2. Forward SFS
    for k in range(1, max_k + 1):
        sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=3)
        sfs.fit(X_train, y_train)
        feats = [feature_names[i] for i in range(total_features) if sfs.support_[i]]
        results['Forward SFS'].append(evaluate_subset(X_train, X_test, y_train, y_test, feats, k))

    # 3. RFE
    for k in range(1, max_k + 1):
        rfe = RFE(LinearRegression(), n_features_to_select=k)
        rfe.fit(X_train, y_train)
        feats = [feature_names[i] for i in range(total_features) if rfe.support_[i]]
        results['RFE'].append(evaluate_subset(X_train, X_test, y_train, y_test, feats, k))

    # 4. ElasticNet
    en = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state, max_iter=10000)
    en.fit(X_train, y_train)
    en_ranking = [feature_names[i] for i in np.argsort(-np.abs(en.coef_))]
    for k in range(1, max_k + 1):
        feats = en_ranking[:k]
        results['ElasticNet'].append(evaluate_subset(X_train, X_test, y_train, y_test, feats, k))

    # 5. Exhaustive Best Subset
    for k in range(1, max_k + 1):
        best_rmse = float('inf')
        best_feats = None
        for comb in itertools.combinations(feature_names, k):
            reg = LinearRegression().fit(X_train[list(comb)], y_train)
            rmse = np.sqrt(mean_squared_error(y_test, reg.predict(X_test[list(comb)])))
            if rmse < best_rmse:
                best_rmse = rmse
                best_feats = list(comb)
        results['Exhaustive'].append(evaluate_subset(X_train, X_test, y_train, y_test, best_feats, k))

    # Export model_data.json and model_data.js
    json_path = os.path.join(output_dir, 'model_data.json')
    js_path = os.path.join(output_dir, 'model_data.js')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(js_path, 'w', encoding='utf-8') as f:
        f.write(f"window.MODEL_DATA = {json.dumps(results, ensure_ascii=False, indent=2)};\n")
    print(f"[+] Saved model data to: {json_path} and {js_path}")

    # Generate Individual Dual-Axis Trajectory Plots
    generate_plots(results, max_k, output_dir)
    print(f"[+] Successfully generated standalone trajectory plots in {output_dir}")
    return results


def evaluate_subset(X_train, X_test, y_train, y_test, feats, k):
    reg = LinearRegression()
    reg.fit(X_train[feats], y_train)
    preds = reg.predict(X_test[feats])

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    n_test = len(y_test)
    adj_r2 = float(1 - (1 - r2) * (n_test - 1) / max(1, n_test - k - 1))

    # OLS Stats
    X_train_sm = sm.add_constant(X_train[feats])
    ols = sm.OLS(y_train, X_train_sm).fit()

    conf_int = ols.conf_int()
    coef_list = []
    
    # Intercept
    coef_list.append({
        'feature': 'Intercept (β₀)',
        'coef': float(ols.params['const']),
        'stderr': float(ols.bse['const']),
        'tstat': float(ols.tvalues['const']),
        'pvalue': float(ols.pvalues['const']),
        'ci_lower': float(conf_int.loc['const', 0]),
        'ci_upper': float(conf_int.loc['const', 1]),
        'vif': None
    })

    # VIFs
    vif_dict = {}
    if len(feats) > 1:
        for idx, col in enumerate(feats):
            vif_dict[col] = float(variance_inflation_factor(X_train[feats].values, idx))
    else:
        vif_dict[feats[0]] = 1.0

    for col in feats:
        coef_list.append({
            'feature': col,
            'coef': float(ols.params[col]),
            'stderr': float(ols.bse[col]),
            'tstat': float(ols.tvalues[col]),
            'pvalue': float(ols.pvalues[col]),
            'ci_lower': float(conf_int.loc[col, 0]),
            'ci_upper': float(conf_int.loc[col, 1]),
            'vif': float(vif_dict.get(col, 1.0))
        })

    # Residuals & Outliers
    res_list = []
    for test_idx, (real_y, pred_y) in enumerate(zip(y_test, preds)):
        err = real_y - pred_y
        res_list.append({
            'index': int(y_test.index[test_idx]),
            'actual': float(real_y),
            'predicted': float(pred_y),
            'residual': float(err),
            'abs_error': float(abs(err)),
            'pct_error': float(abs(err) / (abs(real_y) + 1e-9) * 100)
        })
    res_list.sort(key=lambda x: x['abs_error'], reverse=True)
    for rank, item in enumerate(res_list, start=1):
        item['rank'] = rank

    return {
        'k': k,
        'features': feats,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'adj_r2': adj_r2,
        'f_stat': float(ols.fvalue) if ols.fvalue is not None else 0.0,
        'f_pvalue': float(ols.f_pvalue) if ols.f_pvalue is not None else 0.0,
        'coefficients': coef_list,
        'outliers': res_list
    }


def generate_plots(results, max_k, output_dir):
    colors = {
        'Lasso': '#0284c7',
        'Forward SFS': '#059669',
        'RFE': '#d97706',
        'ElasticNet': '#e11d48',
        'Exhaustive': '#7c3aed'
    }

    # Standalone dual-axis plot per algorithm
    idx = 1
    for algo, records in results.items():
        ks = [r['k'] for r in records]
        rmses = [r['rmse'] for r in records]
        r2s = [r['r2'] for r in records]
        color = colors[algo]

        fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        ax2 = ax1.twinx()

        line1 = ax1.plot(ks, rmses, marker='o', linewidth=2.5, markersize=8, color=color, label='Test RMSE ($)')
        line2 = ax2.plot(ks, r2s, marker='s', linewidth=2.5, markersize=8, color='#10b981', linestyle='--', label='Test R² Score')

        ax1.set_xlabel('Number of Features (k)', fontsize=12, fontweight='bold', labelpad=8)
        ax1.set_ylabel('Test RMSE ($) [Lower is Better]', fontsize=11, fontweight='bold', color=color)
        ax2.set_ylabel('Test R² Score [Higher is Better]', fontsize=11, fontweight='bold', color='#10b981')
        ax1.set_xticks(ks)
        ax1.grid(True, linestyle=':', alpha=0.6)

        plt.title(f'Feature Selection Trajectory: {algo}', fontsize=14, fontweight='bold', pad=14)
        fig.tight_layout()
        save_name = f"algo_{idx}_{algo.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, save_name))
        plt.close()
        idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated CRISP-DM Feature Selection Pipeline')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory')
    args = parser.parse_args()

    run_pipeline(args.data, args.target, args.output_dir)
