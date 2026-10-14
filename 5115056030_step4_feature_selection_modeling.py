"""5115056030 | Step 4: leakage-aware feature selection, modeling, and AR comparison.

Dependencies: numpy, pandas, scipy, matplotlib, scikit-learn, joblib, threadpoolctl.
Run (prefer CLI):
  python 5115056030_step4_feature_selection_modeling.py \
    --csv KAG_energydata_complete.csv --out 5115056030_step4_outputs

Design:
- The same timestamp split as Step 3 (70%,15%,15%), sorted ascending.
- scaler.fit(TRAIN) only; transform Train/Val/Test, no fit on Val/Test.
- 5 feature selection algorithms operate on TRAIN alone, k=14,20,25.
- SFS uses 3-fold TimeSeriesSplit *within TRAIN*; permutation importance
  uses an 80/20 chronological holdout *within TRAIN*.
- A priori Ridge(alpha=10), Lasso(alpha=0.1) on scaled X; no test tuning.
- Model configurations are selected by validation RMSE, only the single
  winner is then scored once on the reserved test set.
- AR extension: [lag1,lag2,lag3] of Appliances, observed up to t-1 only;
  fair comparisons use the exact same aligned rows and partition boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Keep small linear algebra fits deterministic/fast. Set before NumPy import.
for _env in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_env]='1'

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

RANDOM_SEED = 42
RIDGE_ALPHA = 10.0
LASSO_ALPHA = 0.1
LASSO_SELECTOR_ALPHA = 0.1
K_VALUES = [14,20,25]
SELECTORS = ['SelectKBest','RFE','SFS','Lasso','PermutationImportance']
MODELS = ['LinearRegression','Ridge','Lasso']
TARGET = 'Appliances'
TIME = 'date'
LAG_NAMES = ['Appliances_lag1','Appliances_lag2','Appliances_lag3']
FEATURES = ['lights','T1','RH_1','T2','RH_2','T3','RH_3',
    'T4','RH_4','T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8',
    'T9','RH_9','T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    n=len(y_true)
    r2=float(r2_score(y_true,y_pred))
    return {
        'N':n,'p':p,
        'MAE':float(mean_absolute_error(y_true,y_pred)),
        'MSE':float(mean_squared_error(y_true,y_pred)),
        'RMSE':float(np.sqrt(mean_squared_error(y_true,y_pred))),
        'R2':r2,
        'Adjusted_R2':float(1-(1-r2)*(n-1)/(n-p-1)) if n>p+1 else float('nan'),
    }


def get_estimator(name:str):
    if name == 'LinearRegression': return LinearRegression()
    if name == 'Ridge': return Ridge(alpha=RIDGE_ALPHA)
    if name == 'Lasso': return Lasso(alpha=LASSO_ALPHA, max_iter=20000, tol=1e-4, selection='cyclic', random_state=RANDOM_SEED)
    raise ValueError(name)


def compute_vif(a:np.ndarray, names:list[str]) -> pd.DataFrame:
    """Train-only VIF with an intercept; VIF_j=1/(1-R²(Xj~Xother))."""
    n,p=a.shape
    results=[]
    for j, name in enumerate(names):
        x_other=np.delete(a,j,axis=1)
        r2 = r2_score(a[:,j], LinearRegression().fit(x_other,a[:,j]).predict(x_other))
        vif = np.inf if r2>=1-1e-12 else 1/(1-r2)
        results.append({'feature':name,'VIF':float(vif),'aux_R2':float(r2)})
    return pd.DataFrame(results).sort_values('VIF',ascending=False)


def safe_json_value(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,float) and not np.isfinite(x):return 'Infinity' if x>0 else '-Infinity'
    return x


def run(csv_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    df=pd.read_csv(csv_path,parse_dates=[TIME]).sort_values(TIME).reset_index(drop=True)
    if len(df)!=19735: raise ValueError('Data no longer has 19,735 rows; check file.')
    if set(df.columns)!={TIME,TARGET,'rv1','rv2',*FEATURES}:raise ValueError('Columns differ from Step 3.')
    assert np.array_equal(df.rv1.values,df.rv2.values)
    assert not df.isna().any().any()
    assert df.date.is_unique
    n=len(df);i=int(n*.7);j=int(n*.85)
    splits={'Train':slice(None,i),'Validation':slice(i,j),'Test':slice(j,None)}
    parts={name:df.iloc[sl].copy() for name,sl in splits.items()}
    assert len(parts['Train'])==13814 and len(parts['Validation'])==2960 and len(parts['Test'])==2961
    assert (parts['Train'].date.iloc[-1],parts['Validation'].date.iloc[0],parts['Test'].date.iloc[0])==(pd.Timestamp('2016-04-16 15:10:00'),pd.Timestamp('2016-04-16 15:20:00'),pd.Timestamp('2016-05-07 04:40:00'))
    # Save the exact copies from Step 3 if they aren't present in this execution environment.
    step3=out/'step3_confirmed_splits';step3.mkdir(exist_ok=True)
    for name,frame in parts.items():
        frame[[TIME,TARGET,*FEATURES]].to_csv(step3/(name.lower()+'.csv'),index=False)
    ys={name:part[TARGET].to_numpy() for name,part in parts.items()}
    scaler=StandardScaler().fit(parts['Train'][FEATURES])
    XX={name:scaler.transform(part[FEATURES]) for name,part in parts.items()}
    joblib.dump({'scaler':scaler,'features':FEATURES,'source_csv':csv_path.name},out/'environment_train_fitted_scaler.joblib')
    print('DATA: Train 13814, Validation 2960, Test 2961; scaler fit on TRAIN only',flush=True)
    Xtr=XX['Train'];ytr=ys['Train'];Xva=XX['Validation'];yva=ys['Validation']

    # Five selectors compute their rankings with Train data only.
    selector_indices={};selector_details={};
    scorer=SelectKBest(score_func=f_regression,k=25).fit(Xtr,ytr)
    best_order=np.argsort(-scorer.scores_,kind='stable')
    for k in (14,20):selector_indices[('SelectKBest',k)]=np.sort(best_order[:k])
    selector_details['SelectKBest']={'train_scores':dict(zip(FEATURES,map(float,scorer.scores_)))}
    print('Selection: SelectKBest done',flush=True)

    # RFE is fit on Train only. Different k yields real RFE fitted models.
    for k in (14,20):
        rf=RFE(LinearRegression(),n_features_to_select=k,step=1).fit(Xtr,ytr)
        selector_indices[('RFE',k)]=np.flatnonzero(rf.support_)
    print('Selection: RFE done',flush=True)

    # SFS: forward for 14, backward for 20, 3-way chronological CV entirely Train.
    cv=TimeSeriesSplit(n_splits=3)
    for k in (14,20):
        direction='forward' if k==14 else 'backward'
        print(f'Selection: SFS k={k} ({direction}) beginning ({time.perf_counter()-t0:.1f}s)',flush=True)
        sfs=SequentialFeatureSelector(make_pipeline(StandardScaler(),LinearRegression()),n_features_to_select=k,direction=direction,
                                      scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1)
        sfs.fit(parts['Train'][FEATURES].to_numpy(),ytr) # fold scaler fits on each Train CV fold
        selector_indices[('SFS',k)]=np.flatnonzero(sfs.get_support())
        print(f'Selection: SFS k={k} finished ({time.perf_counter()-t0:.1f}s)',flush=True)
    selector_details['SFS']={'cv':'TimeSeriesSplit(n_splits=3) on Train','k14':'forward','k20':'backward'}

    lsel=Lasso(alpha=LASSO_SELECTOR_ALPHA,max_iter=30000,tol=1e-4,random_state=RANDOM_SEED).fit(Xtr,ytr)
    lasso_order=np.argsort(-np.abs(lsel.coef_),kind='stable')
    for k in (14,20):selector_indices[('Lasso',k)]=np.sort(lasso_order[:k])
    selector_details['Lasso']={'alpha':LASSO_SELECTOR_ALPHA,'nonzero_count':int(np.count_nonzero(abs(lsel.coef_)>1e-7)),
                               'train_coefficient_scores':dict(zip(FEATURES,map(float,lsel.coef_)))}
    print('Selection: Lasso done',flush=True)

    # Permutation importance on the last 20% of Training ONLY (after fitting on first 80%).
    sub=int(len(ytr)*.8)
    inner_scaler=StandardScaler().fit(parts['Train'][FEATURES].iloc[:sub])
    inner_xtr=inner_scaler.transform(parts['Train'][FEATURES].iloc[:sub])
    inner_xhold=inner_scaler.transform(parts['Train'][FEATURES].iloc[sub:])
    pim=LinearRegression().fit(inner_xtr,ytr[:sub])
    prm=permutation_importance(pim,inner_xhold,ytr[sub:],scoring='neg_root_mean_squared_error',
                               n_repeats=7,random_state=RANDOM_SEED,n_jobs=1)
    perm_order=np.argsort(-prm.importances_mean,kind='stable')
    for k in (14,20):selector_indices[('PermutationImportance',k)]=np.sort(perm_order[:k])
    selector_details['PermutationImportance']={
        'inner_training_n':sub,'inner_holdout_n':len(ytr)-sub, 'repeats':7,
        'inner_scaler_fit':'First 80% of Train only, not on internal holdout',
        'importance_mean':dict(zip(FEATURES,map(float,prm.importances_mean))),
        'importance_std':dict(zip(FEATURES,map(float,prm.importances_std)))}
    for method in SELECTORS:selector_indices[(method,25)]=np.arange(25)
    print('Selection: Permutation Importance done',flush=True)

    selection_rows=[];vif_rows=[];vif_summary=[];performance=[]
    # cache identical feature subsets' training-only VIF, e.g. all 25 columns.
    vif_cache={}
    with threadpool_limits(limits=1):
        for method in SELECTORS:
            for k in K_VALUES:
                idx=selector_indices[(method,k)]
                assert len(idx)==k, f'bad k {method} {k}'
                sel_names=[FEATURES[z] for z in idx]
                selection_rows.append({'method':method,'k':k,'features':', '.join(sel_names)})
                key=tuple(idx.tolist())
                if key not in vif_cache:vif_cache[key]=compute_vif(Xtr[:,idx],sel_names)
                vis=vif_cache[key]
                vif_summary.append({'method':method,'k':k,'max_VIF':float(vis.VIF.max()),
                                    'median_VIF':float(vis.VIF.median()),'n_VIF_gt_5':int((vis.VIF>5).sum()),
                                    'n_VIF_gt_10':int((vis.VIF>10).sum()),'worst_feature':str(vis.iloc[0].feature)})
                for _, rec in vis.iterrows():
                    vif_rows.append({'method':method,'k':k,'feature':rec.feature,'VIF':float(rec.VIF),'aux_R2':float(rec.aux_R2)})
                for model_name in MODELS:
                    model=get_estimator(model_name)
                    model.fit(Xtr[:,idx],ytr)
                    pred=model.predict(Xva[:,idx]);metrics=metric_dict(yva,pred,k)
                    performance.append({'selector':method,'k':k,'model':model_name,
                                        **metrics, 'selection_features':', '.join(sel_names)})
            print(f'Modeling: {method} completed ({time.perf_counter()-t0:.1f}s)',flush=True)

    selection_df=pd.DataFrame(selection_rows)
    sel_vif=pd.DataFrame(vif_summary)
    perf=pd.DataFrame(performance).sort_values(['RMSE','MAE','selector','k','model']).reset_index(drop=True)
    vifs=pd.DataFrame(vif_rows)
    for frame,file in [(selection_df,'selected_features.csv'),(sel_vif,'vif_summary.csv'),
                       (vifs,'vif_details.csv'),(perf,'validation_model_comparison.csv')]:
        frame.to_csv(out/file,index=False,encoding='utf-8-sig',float_format='%.8f')
    (out/'selector_train_only_details.json').write_text(json.dumps(selector_details,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Validation configs complete:',len(perf),'best:',perf.iloc[0][['selector','k','model','RMSE','R2']].to_dict(),flush=True)

    # Select the single best *on VALIDATION ONLY*. Fit on Train once, evaluate Test once.
    winner=perf.iloc[0]
    method=str(winner.selector);k=int(winner.k);model_name=str(winner.model)
    idx=selector_indices[(method,k)]
    final_model=get_estimator(model_name).fit(Xtr[:,idx],ytr)
    final_pred=final_model.predict(XX['Test'][:,idx])
    final_metrics=metric_dict(ys['Test'],final_pred,k)
    final_result={'selection_from':'Validation RMSE only', 'selector':method,'k':k,'model':model_name,
                  'feature_names':[FEATURES[z] for z in idx], 'validation_metrics':
                    {name:safe_json_value(winner[name]) for name in ['N','p','MAE','MSE','RMSE','R2','Adjusted_R2']},
                  'test_metrics':final_metrics}
    (out/'final_best_test_metrics.json').write_text(json.dumps(final_result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame({'date':parts['Test'].date,'actual_Wh':ys['Test'],'predicted_Wh':final_pred}).to_csv(
        out/'final_best_test_predictions.csv',index=False,encoding='utf-8-sig')
    joblib.dump({'scaler':scaler,'feature_pool':FEATURES,'selected_features':[FEATURES[z] for z in idx],
                 'indices':idx,'estimator':final_model, 'selection_rule':'Min validation RMSE',
                 'test_metrics':final_metrics},out/'locked_final_model.joblib')
    print('Reserved Test evaluated for ONE best validation-selected model',flush=True)

    # AR comparison, precisely pre-registered (NOT selected/tuned by test).
    # Critically shift FIRST, before slicing, allowing previous-period observed y across boundaries.
    lag_df=df[[TIME,TARGET,*FEATURES]].copy()
    for lag,label in enumerate(LAG_NAMES,1):lag_df[label]=df[TARGET].shift(lag)
    lag_df=lag_df.dropna(subset=LAG_NAMES).reset_index(drop=True)
    # split based on precise *timestamp* boundaries, not new 70/15/15 after dropping first 3.
    cutoff1=parts['Validation'].date.iloc[0];cutoff2=parts['Test'].date.iloc[0]
    lag_parts={'Train':lag_df[lag_df.date<cutoff1].copy(),
               'Validation':lag_df[(lag_df.date>=cutoff1)&(lag_df.date<cutoff2)].copy(),
               'Test':lag_df[lag_df.date>=cutoff2].copy()}
    assert [len(lag_parts[k]) for k in ['Train','Validation','Test']]==[13811,2960,2961]
    assert lag_parts['Validation'].date.iloc[0]==cutoff1 and lag_parts['Test'].date.iloc[0]==cutoff2
    ar_rows=[];ar_models={};ar_scalers={}
    for label,feat in [('EnvironmentOnly_25',FEATURES),('EnvironmentPlus3Lags_28',FEATURES+LAG_NAMES)]:
        ar_scaler=StandardScaler().fit(lag_parts['Train'][feat]) # Train only
        x_train=ar_scaler.transform(lag_parts['Train'][feat]); y_train=lag_parts['Train'][TARGET].to_numpy()
        ar_model=LinearRegression().fit(x_train,y_train)
        ar_models[label]=ar_model;ar_scalers[label]=ar_scaler
        for phase in ['Validation','Test']:
            ff=lag_parts[phase]
            yp=ar_model.predict(ar_scaler.transform(ff[feat]))
            ms=metric_dict(ff[TARGET].to_numpy(),yp,len(feat))
            ar_rows.append({'experiment':label,'partition':phase,**ms})
            pd.DataFrame({'date':ff.date,'actual_Wh':ff[TARGET],
                          'predicted_Wh':yp}).to_csv(out/f'ar_{label}_{phase.lower()}_predictions.csv',index=False,encoding='utf-8-sig')
    ar_results=pd.DataFrame(ar_rows)
    ar_results.to_csv(out/'autoregression_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.8f')
    joblib.dump({'feature_pool':FEATURES,'lags':LAG_NAMES,
                 'approach':'Observed lags t-1..t-3, one-step rolling estimation only',
                 'models':ar_models,'scalers':ar_scalers},out/'ar_experiment_models.joblib')
    print('Prespecified AR Test comparison evaluated in one locked pass',flush=True)

    # Charts (all using actual results). Use authored palette in saved PNG allowed for doc formatting.
    plt.rcParams.update({'font.size':10,'axes.spines.right':False,'axes.spines.top':False})
    top=perf.iloc[:16].sort_values('RMSE',ascending=True).copy()
    labels=[f'{r.selector.replace("PermutationImportance", "Permutation")} · {r.model} · {int(r.k)}' for r in top.itertuples(index=False)]
    # Dot plot with a meaningful zoom; unlike truncated bars, dots do not imply zero baseline.
    fig,ax=plt.subplots(figsize=(11.5,8.2))
    pos=np.arange(len(top))[::-1]
    ax.scatter(top.RMSE.values,pos,s=56,color='#38677D',zorder=3)
    ax.set_yticks(pos,labels,fontsize=9)
    ax.set_xlim(float(top.RMSE.min())-.35,float(top.RMSE.max())+.45)
    for x, yy in zip(top.RMSE.values,pos):
        ax.text(x+.035,yy,f'{x:.2f}',va='center',fontsize=8)
    ax.set_ylim(-.7,len(top)-.3)
    ax.set(xlabel='Validation RMSE (Wh, lower is better)',title='Validation: Lowest RMSE Across 45 Configurations (top 16)')
    ax.grid(axis='x',alpha=.18)
    fig.tight_layout();fig.savefig(out/'01_validation_top16_rmse.png',dpi=170);plt.close(fig)

    summary_vif=sel_vif.drop_duplicates(['method','k']).copy()
    pivot=summary_vif.pivot(index='method',columns='k',values='n_VIF_gt_10').reindex(SELECTORS)[K_VALUES]
    fig,ax=plt.subplots(figsize=(8.8,3.9));im=ax.imshow(pivot.to_numpy(),cmap='YlOrBr',vmin=0)
    ax.set_xticks(range(3),[str(k) for k in K_VALUES]);ax.set_yticks(range(5),SELECTORS)
    ax.set(xlabel='Number of selected features',title='Train-only VIF: Count of Features with VIF > 10')
    for i in range(5):
        for j in range(3):ax.text(j,i,str(int(pivot.iloc[i,j])),ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Features above VIF 10');fig.tight_layout()
    fig.savefig(out/'02_vif_gt10_heatmap.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,5.7));bestplot=perf.head(10)
    ax.scatter(bestplot.RMSE,bestplot.R2,s=60,color='#38677D')
    for z in [0,1,2]:
        row=bestplot.iloc[z]
        desc=f'{row.selector.replace("PermutationImportance", "Permutation")}/{int(row.k)}/{row.model}'
        dx,dy=[(7,10),(7,-18),(7,10)][z]
        ax.annotate(desc,(row.RMSE,row.R2),xytext=(dx,dy),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Validation RMSE (Wh)',ylabel='Validation R²',title='Top 10 Models: R² vs RMSE (Validation)')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_validation_r2_vs_rmse.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,4.3));dates=parts['Test'].date.values
    # Display 5-day contiguous part of test (720 ten-minute samples) for readability.
    show=min(720,len(dates));ax.plot(dates[:show],ys['Test'][:show],label='Actual',lw=1,alpha=.8)
    ax.plot(dates[:show],final_pred[:show],label='Predicted',lw=1.1)
    ax.set(xlabel='Test timestamp',ylabel='Appliances (Wh)',
           title=f'Final Test: {method}, k={k}, {model_name} (first {show} observations)')
    ax.legend();ax.grid(alpha=.14);fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(out/'04_final_best_test_actual_vs_predicted.png',dpi=170);plt.close(fig)

    val_ar=ar_results[ar_results.partition=='Validation'].set_index('experiment')
    test_ar=ar_results[ar_results.partition=='Test'].set_index('experiment')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    experiments=['EnvironmentOnly_25','EnvironmentPlus3Lags_28']
    short_names=['25 environmental','25 + 3 observed lags']
    for idx_phase,(phase,tab) in enumerate([('Validation',val_ar),('Test',test_ar)]):
        axes[idx_phase].bar(short_names,[tab.loc[e,'RMSE'] for e in experiments],color=['#9A8B77','#267D75'])
        axes[idx_phase].set(title=f'{phase} RMSE (Wh)',ylabel='RMSE (Wh)')
        axes[idx_phase].tick_params(axis='x',labelrotation=18)
        axes[idx_phase].grid(axis='y',alpha=.13)
    fig.suptitle('Autoregressive Extension (Fair Same-row Comparison)',fontsize=12)
    fig.tight_layout();fig.savefig(out/'05_ar_rmse_comparison.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for idx_phase,(phase,tab) in enumerate([('Validation',val_ar),('Test',test_ar)]):
        axes[idx_phase].bar(short_names,[tab.loc[e,'R2'] for e in experiments],color=['#9A8B77','#267D75'])
        axes[idx_phase].set(title=f'{phase} R²',ylabel='R²')
        axes[idx_phase].tick_params(axis='x',labelrotation=18)
        axes[idx_phase].grid(axis='y',alpha=.13)
    fig.suptitle('Autoregressive Extension — R² Comparison',fontsize=12)
    fig.tight_layout();fig.savefig(out/'06_ar_r2_comparison.png',dpi=170);plt.close(fig)

    total_elapsed=time.perf_counter()-t0
    manifest={'source_csv':csv_path.name,'train_rows':13814,'validation_rows':2960,'test_rows':2961,
              'standard_scaler_fit_on':'Train only', 'feature_count_original':27,'feature_count_prepared':25,
              'selected_feature_counts':K_VALUES, 'feature_selection_methods':SELECTORS,
              'model_types':MODELS,'ridge_alpha':RIDGE_ALPHA,'lasso_alpha':LASSO_ALPHA,
              'lasso_selector_alpha':LASSO_SELECTOR_ALPHA,'time_series_cv_sfs_folds':3, 'time_series_cv_sfs_scaling':'StandardScaler is fitted within every training CV fold via Pipeline',
              'permutation_importance_inner_train_pct':80,'permutation_importance_repeats':7,
              'validation_configurations':len(perf), 'run_seconds':round(total_elapsed,2),
              'final_test_criterion':'Validation RMSE minimum (NO selection based on Test)',
              'ar_alignment':'All lag features shifted before date slicing, so use t-1,t-2,t-3 only',
              'ar_train_rows':13811, 'ar_validation_rows':2960,'ar_test_rows':2961,
              'ar_experiment':'Prespecified 25 vs 28-feature LinearRegression for both Validation/Test',
              'notes':'The original task is same-interval estimation; AR assumes historical appliances measurements are available.'}
    (out/'run_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('FINAL TEST:',final_result,flush=True)
    print('AR RESULTS:\n',ar_results.to_string(index=False),flush=True)
    print(f'Complete! runtime {total_elapsed:.1f}s; results at {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='HW2 Step 4 model comparison, trained on Train only.')
    parser.add_argument('--csv',type=Path,default=Path('KAG_energydata_complete.csv'))
    parser.add_argument('--out',type=Path,default=Path('5115056030_step4_outputs'))
    args=parser.parse_args()
    run(args.csv,args.out)
