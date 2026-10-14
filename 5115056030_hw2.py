"""5115056030 HW2 — End-to-end CRISP-DM Appliances Energy Prediction.

SINGLE FILE INTEGRATED REPRODUCTION of Steps 1–6.
Usage:
    pip install numpy pandas scipy matplotlib scikit-learn statsmodels joblib threadpoolctl
    python 5115056030_hw2.py

Expected input next to this file: KAG_energydata_complete.csv
Outputs: 5115056030_step2_outputs/ ... 5115056030_step6_outputs/

Step4 feature selectors are fitted with TRAIN only and tested on Validation.
Step4 winner chosen by Validation RMSE; Test scored once (not used to tune).
Step5 intervals are classical IID-normal OLS and require caution for spikes.
Step6 loads persisted Step4 regressors (no retraining) for sample predictions.
Note: NotebookLM is external research provided by the student; code does not call NotebookLM.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

FEATURES25 = ['lights','T1','RH_1','T2','RH_2','T3','RH_3','T4','RH_4',
    'T5','RH_5','T6','RH_6','T7','RH_7','T8','RH_8','T9','RH_9',
    'T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']


def run_steps_1_to_3(csv_path: Path, root: Path):
    """Data provenance, describe fields, confirm leak-free chronological partitions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    d=pd.read_csv(csv_path,parse_dates=['date']).sort_values('date',kind='stable').reset_index(drop=True)
    assert len(d)==19735 and set(d.columns)=={'date','Appliances','rv1','rv2',*FEATURES25}
    assert d.date.is_unique and not d.isna().any().any()
    assert np.array_equal(d.rv1,d.rv2) and len(FEATURES25)==25
    dt=d.date.diff().dropna()
    assert (dt==pd.Timedelta(minutes=10)).all()
    cut1=int(len(d)*0.7);cut2=int(len(d)*0.85)
    assert (cut1,cut2)==(13814,16774)
    s2=root/'5115056030_step2_outputs';s3=root/'5115056030_step3_outputs'
    s2.mkdir(exist_ok=True);s3.mkdir(exist_ok=True)
    d.select_dtypes('number').describe().T.to_csv(s2/'descriptive_statistics.csv',encoding='utf-8-sig')
    corr=d.select_dtypes('number').corr()
    corr.to_csv(s2/'correlation_matrix.csv',encoding='utf-8-sig')
    (corr['Appliances'].drop('Appliances').sort_values(key=abs,ascending=False)).to_csv(s2/'feature_target_correlations.csv',encoding='utf-8-sig')
    for name,frame in [('train',d.iloc[:cut1]),('validation',d.iloc[cut1:cut2]),('test',d.iloc[cut2:])]:
        frame[['date','Appliances',*FEATURES25]].to_csv(s3/f'{name}.csv',index=False)
    (s3/'data_preparation_manifest.json').write_text(json.dumps({
        'source':csv_path.name,'n':len(d),'train':cut1,'validation':cut2-cut1,'test':len(d)-cut2,
        'chosen_features':FEATURES25,'dropped':['rv1','rv2'],
        'date_role':'sort only','scenario':'same-interval energy estimation, NOT forecasting',
        'no_na':True,'rv1_equals_rv2':True,'every_gap_10min':True
    },ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4));ax.hist(d.Appliances,bins=50,color='#46758A')
    ax.set(title='EDA: Appliances Energy',xlabel='Wh per 10-minute record',ylabel='Count')
    fig.tight_layout();fig.savefig(s2/'01_eda_target_hist.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,7));ax.imshow(corr.to_numpy(),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=7)
    ax.set_yticks(range(len(corr)),corr.columns,fontsize=7)
    ax.set_title('All Numeric Correlations');fig.tight_layout()
    fig.savefig(s2/'02_eda_pearson.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));daily=d.set_index('date').Appliances.resample('D').mean()
    ax.plot(daily.index,daily.values,color='#446C84',lw=1.3)
    ax.axvspan(d.date.iloc[0],d.date.iloc[cut1],alpha=.14,color='#267D75',label='Train')
    ax.axvspan(d.date.iloc[cut1],d.date.iloc[cut2],alpha=.14,color='#D69A35',label='Validation')
    ax.axvspan(d.date.iloc[cut2],d.date.iloc[-1],alpha=.14,color='#6485B7',label='Test')
    ax.set(title='Chronological Train / Validation / Test',xlabel='Date',ylabel='Mean energy Wh')
    ax.legend();fig.autofmt_xdate();fig.tight_layout();fig.savefig(s3/'00_split_overview.png',dpi=160);plt.close(fig)
    print('Steps 1–3 confirmed: 19,735 rows, 27 original numeric inputs, 25 retained; chronological 13,814 / 2,960 / 2,961')

def run_step4(csv_path: Path, root: Path):
    """Step 4: five feature selectors × three k × three regressors, plus ARX."""
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



    run(csv_path,root/"5115056030_step4_outputs")

def run_step5(csv_path: Path, root: Path):
    """Step 5: load locked models, diagnostics, classical 95% CI/PI."""
    """5115056030 HW2 Step 5 — Five OLS diagnostics + 95% CI/PI on locked Step 4 models.

    Requires: numpy, pandas, scipy, matplotlib, scikit-learn, statsmodels, joblib
    Example:
      python 5115056030_step5_evaluation.py \
        --csv KAG_energydata_complete.csv \
        --step4 5115056030_step4_outputs \
        --out 5115056030_step5_outputs

    IMPORTANT
      - No training on Validation/Test, no model/feature re-selection.
      - statsmodels OLS trained ONLY on original Step 4 Train to reproduce existing locked
        sklearn LinearRegression predictions and calculate conventional IID-normal CI/PI.
      - The reported 95% OLS intervals are assumption-dependent and not necessarily
        calibrated for autocorrelated, skewed or heteroskedastic energy consumption.
      - Lag model uses actual historical y[t-1], y[t-2], y[t-3] only, never y[t].
    """
    import argparse
    import json
    import os
    from pathlib import Path
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest
    from statsmodels.tsa.stattools import acf
    from scipy.stats import skew, kurtosis, norm
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    LAGS=['Appliances_lag1','Appliances_lag2','Appliances_lag3']
    TARGET='Appliances'; TIME='date'
    COLORS={'A_best14':'#275D7A','Env25':'#989282','B_lag28':'#B25F35'}
    CONFIG=[('A_best14','A: Best 14 Environmental Features'),('Env25','Environment-only Control: 25 Features'),('B_lag28','B: 25 Environment + 3 Observed Lags')]


    def metrics(y,pred,features):
        n=len(y); mse=float(mean_squared_error(y,pred)); r2=float(r2_score(y,pred))
        return {'N':int(n),'p':int(features),'MAE':float(mean_absolute_error(y,pred)),
                'MSE':mse,'RMSE':float(np.sqrt(mse)),'R2':r2,
                'Adjusted_R2':float(1-(1-r2)*(n-1)/(n-features-1))}


    def vif_for(x, names):
        records=[]
        for ix,feat in enumerate(names):
            aux_pred=LinearRegression().fit(np.delete(x,ix,axis=1),x[:,ix]).predict(np.delete(x,ix,axis=1))
            r2=float(r2_score(x[:,ix],aux_pred))
            records.append({'feature':feat,'VIF':float('inf') if r2>=1-1e-12 else float(1/(1-r2)), 'aux_R2':r2})
        return sorted(records,key=lambda r:r['VIF'],reverse=True)


    def dump_csv(frame,path):
        frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.10g')


    def plot_diagnostics(key, title, data, out):
        """Residuals vs fitted, acf, histogram with normal reference, QQ plot — Test diagnostics."""
        pred=data['prediction'].to_numpy();resid=data['residual'].to_numpy()
        z=(resid-np.mean(resid))/np.std(resid,ddof=1)
        fig,axs=plt.subplots(2,2,figsize=(13.7,9.0))
        ax=axs[0,0]
        ax.scatter(pred,resid,s=5,alpha=.20,color=COLORS[key],rasterized=True)
        order=np.argsort(pred)
        smooth=pd.Series(resid[order]).rolling(window=max(40,len(resid)//15),min_periods=10,center=True).median().to_numpy()
        ax.plot(pred[order],smooth,color='#C33B31',linewidth=2,label='Rolling median trend')
        ax.axhline(0,lw=1,ls='--',color='#555555');ax.set(xlabel='OLS fitted (Wh)',ylabel='Residual (actual - fitted, Wh)',title='Residuals vs Fitted (Test)');ax.legend(loc='upper left',fontsize=8)
        ax=axs[0,1]
        acfs=acf(resid,nlags=48,fft=True)
        ax.bar(np.arange(1,49),acfs[1:],color=COLORS[key],width=.8)
        band=1.96/np.sqrt(len(resid))
        ax.axhline(+band,ls='--',lw=1,color='#AA5540',label='Naive ±1.96/√n')
        ax.axhline(-band,ls='--',lw=1,color='#AA5540')
        ax.axhline(0,color='#555555',lw=.7)
        ax.set(xlabel='Lag (10-minute intervals)',ylabel='ACF of Test residuals',title='Residual Autocorrelation (Test)');ax.legend(fontsize=8)
        ax=axs[1,0]
        ax.hist(z,bins=60,density=True,alpha=.75,color=COLORS[key],edgecolor='white',linewidth=.25)
        zz=np.linspace(-5,5,300);ax.plot(zz,norm.pdf(zz),lw=2,color='#C33B31',label='Standard normal reference')
        ax.set(xlabel='Standardized residual',ylabel='Density',title='Residual Histogram (Test)');ax.legend(fontsize=8)
        ax=axs[1,1]
        sm.qqplot(z,line='45',ax=ax,marker='.',alpha=.35,markersize=3)
        ax.set(title='Normal Q-Q Plot (Test)',xlabel='Theoretical normal quantiles',ylabel='Observed residual quantiles')
        fig.suptitle(title+' | OLS diagnostic panels',fontsize=15,fontweight='bold')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(out/f'01_diagnostics_{key}.png',dpi=165,bbox_inches='tight');plt.close(fig)


    def plot_intervals(key,title,frame,out):
        """Plot ALL Test at readable density and EXACT first 432 rows = 72 hours."""
        tt=pd.to_datetime(frame['date']); actual=frame['actual_Wh'].to_numpy()
        pred=frame['prediction'].to_numpy();lo=frame['obs_ci_lower'].to_numpy();hi=frame['obs_ci_upper'].to_numpy()
        clo=frame['mean_ci_lower'].to_numpy();chi=frame['mean_ci_upper'].to_numpy()
        for kind,window in [('overview',len(frame)),('72h_zoom',432)]:
            use=slice(0,window)
            fig,ax=plt.subplots(figsize=(15.5,5.1 if kind=='overview' else 5.8))
            xx=tt.iloc[use]
            ax.fill_between(xx,lo[use],hi[use],color='#D19B78',alpha=.23,label='95% prediction interval (PI)')
            ax.fill_between(xx,clo[use],chi[use],color='#5A89AD',alpha=.37,label='95% confidence interval (mean CI)')
            ax.plot(xx,actual[use],color='#333333',lw=.65,alpha=.82,label='Actual (Wh)',zorder=3)
            ax.plot(xx,pred[use],color=COLORS[key],lw=1.1,label='Predicted (OLS)',zorder=4)
            ax.axhline(0,color='#777777',lw=.7,ls='--')
            ax.set(title=title+' | '+('Entire Test Period' if kind=='overview' else 'First 72 Hours / 432 Contiguous Observations'),
                   ylabel='Appliances energy (Wh)',xlabel='Test timestamp')
            ax.legend(ncol=4,fontsize=9,loc='upper right');ax.grid(alpha=.12)
            fig.autofmt_xdate();fig.tight_layout()
            fig.savefig(out/f'02_intervals_{key}_{kind}.png',dpi=165,bbox_inches='tight');plt.close(fig)


    def main(csv_path,step4,out):
        out.mkdir(parents=True,exist_ok=True)
        np.random.seed(42)
        df=pd.read_csv(csv_path,parse_dates=[TIME]).sort_values(TIME).reset_index(drop=True)
        if len(df)!=19735 or not df[TIME].is_unique:
            raise ValueError('Unexpected source rows/timestamps; Step 3 split cannot be verified')
        n=len(df);cut1=int(n*.7);cut2=int(n*.85)
        if (cut1,cut2)!=(13814,16774):raise AssertionError('Step3 boundaries mismatch')
        endtrain=df[TIME].iloc[cut1];starttest=df[TIME].iloc[cut2]
        assert str(endtrain)=='2016-04-16 15:20:00' and str(starttest)=='2016-05-07 04:40:00'
        locked=joblib.load(step4/'locked_final_model.joblib')
        env=joblib.load(step4/'ar_experiment_models.joblib')
        expected=json.loads((step4/'final_best_test_metrics.json').read_text(encoding='utf-8'))
        names25=list(env['feature_pool']); names14=list(locked['selected_features'])
        assert expected['selector']=='PermutationImportance' and expected['model']=='LinearRegression' and expected['k']==14
        assert names14==expected['feature_names']
        assert names25==list(locked['feature_pool'])
        lagged=df.copy()
        for i,name in enumerate(LAGS,1):lagged[name]=lagged[TARGET].shift(i)
        # For B, lagged first 3 points are unavailable. All Validation and Test rows stay intact.
        lagged=lagged.dropna(subset=LAGS).reset_index(drop=True)
        configs={
          'A_best14':(df.iloc[:cut1].copy(),df.iloc[cut2:].copy(),names14,locked['scaler'],locked['estimator']),
          'Env25':(lagged[lagged[TIME]<endtrain].copy(),lagged[lagged[TIME]>=starttest].copy(),names25,env['scalers']['EnvironmentOnly_25'],env['models']['EnvironmentOnly_25']),
          'B_lag28':(lagged[lagged[TIME]<endtrain].copy(),lagged[lagged[TIME]>=starttest].copy(),names25+LAGS,env['scalers']['EnvironmentPlus3Lags_28'],env['models']['EnvironmentPlus3Lags_28']),
        }
        diagnostic_rows=[]; coverage_rows=[]; vif_rows=[]; vifsumm=[]; metrics_rows=[]; residual_ACF=[]
        full={};check_predictions={}
        for key,title in CONFIG:
            train,test,features,scaler,prior_estimator=configs[key]
            if key=='A_best14':
                # The locked Step4 scaler was fit on the FULL 25-feature candidate pool;
                # transform all 25 in original order, then apply the locked 14 indices.
                cols=names25
                indices=np.asarray(locked['indices'], dtype=int)
                xtr=scaler.transform(train[cols])[:,indices]
                xte=scaler.transform(test[cols])[:,indices]
            else:
                xtr=scaler.transform(train[features]);xte=scaler.transform(test[features])
            ytr=train[TARGET].to_numpy(); yte=test[TARGET].to_numpy()
            assert len(test)==2961 and len(features) in (14,25,28)
            xtrain=sm.add_constant(xtr,has_constant='add')
            xtest=sm.add_constant(xte,has_constant='add')
            ols=sm.OLS(ytr,xtrain).fit() # OLS on TRAIN only
            preds=ols.predict(xtest)
            expected_preds=prior_estimator.predict(xte)
            max_difference=float(np.max(np.abs(preds-expected_preds)))
            if max_difference>1e-5:raise ValueError(f'OLS does not match Step4 locked estimator ({key}): {max_difference}')
            check_predictions[key]=max_difference
            sf=ols.get_prediction(xtest).summary_frame(alpha=.05)
            lower=np.asarray(sf['obs_ci_lower'],float);upper=np.asarray(sf['obs_ci_upper'],float)
            cilo=np.asarray(sf['mean_ci_lower'],float);cihi=np.asarray(sf['mean_ci_upper'],float)
            residual=yte-preds
            t_residual=ytr-ols.fittedvalues
            test_data=pd.DataFrame({
              'date':test[TIME].to_numpy(),'actual_Wh':yte,'prediction':preds,'residual':residual,
              'mean_ci_lower':cilo,'mean_ci_upper':cihi,'obs_ci_lower':lower,'obs_ci_upper':upper,
              'covered_95_PI':((yte>=lower)&(yte<=upper)).astype(int),'covered_95_mean_CI':((yte>=cilo)&(yte<=cihi)).astype(int),
            })
            dump_csv(test_data,out/f'predictions_with_intervals_{key}.csv')
            full[key]=test_data
            # Residual diagnostics on Train and Test; inferential tests conditional on regression assumptions.
            for partition, obs_y, fit_y, fitted_x, diag_resid in [
                ('Train',ytr,np.asarray(ols.fittedvalues),xtrain,np.asarray(t_residual)),
                ('Test',yte,preds,xtest,residual)]:
                dw=float(durbin_watson(diag_resid))
                bplm,bpp,bpf,bpfp=het_breuschpagan(diag_resid,fitted_x)
                jb,jbp,jbsk,jbk=jarque_bera(diag_resid)
                omni,omnip=omni_normtest(diag_resid)
                ac=acf(diag_resid,nlags=48,fft=True)
                for lag,value in enumerate(ac):residual_ACF.append({'model':key,'partition':partition,'lag':lag,'residual_acf':float(value)})
                diagnostic_rows.append({'model':key,'partition':partition,'N':len(diag_resid),'features':len(features),
                     'Durbin_Watson':dw,'residual_ACF_lag1':float(ac[1]),'residual_ACF_lag6':float(ac[6]),
                     'BP_LM_stat':float(bplm),'BP_LM_pvalue':float(bpp),'BP_F_stat':float(bpf),'BP_F_pvalue':float(bpfp),
                     'residual_skewness':float(skew(diag_resid,bias=False)),
                     'residual_excess_kurtosis':float(kurtosis(diag_resid,bias=False,fisher=True)),
                     'JB_stat':float(jb),'JB_pvalue':float(jbp),
                     'Omnibus_stat':float(omni),'Omnibus_pvalue':float(omnip),
                     'residual_mean':float(np.mean(diag_resid)),'residual_std':float(np.std(diag_resid,ddof=1)),
                     'train_model_R2':float(ols.rsquared) if partition=='Train' else np.nan,
                     'OLS_condition_number':float(ols.condition_number),
                     })
            vifs=vif_for(xtr,features)
            for v in vifs:vif_rows.append({'model':key,**v})
            vifsumm.append({'model':key,'feature_count':len(features),'max_VIF':max(v['VIF'] for v in vifs),
                             'median_VIF':float(np.median([v['VIF'] for v in vifs])),
                             'n_VIF_gt_5':sum(v['VIF']>5 for v in vifs),'n_VIF_gt_10':sum(v['VIF']>10 for v in vifs),
                             'worst_feature':vifs[0]['feature']})
            coverage=float(np.mean((yte>=lower)&(yte<=upper)))
            coverage_rows.append({'model':key,'group':'All test','N':len(yte),'PI_coverage':coverage,
                'mean_PI_width_Wh':float(np.mean(upper-lower)),'median_PI_width_Wh':float(np.median(upper-lower)),
                'mean_CI_width_Wh':float(np.mean(cihi-cilo)),
                'PI_lower_negative_pct':float(100*np.mean(lower<0)),
                'n_below_PI':int(np.sum(yte<lower)),'n_above_PI':int(np.sum(yte>upper))})
            # Diagnostic/coverage by actual load, decided before looking at results (not tuning).
            groups=[('Low: <=100 Wh',yte<=100),('Medium: 100-200 Wh',(yte>100)&(yte<=200)),
                    ('High: 200-400 Wh',(yte>200)&(yte<=400)),('Peak: >400 Wh',yte>400)]
            for g,mask in groups:
                coverage_rows.append({'model':key,'group':g,'N':int(mask.sum()),
                    'PI_coverage':float(np.mean((yte[mask]>=lower[mask])&(yte[mask]<=upper[mask]))),
                    'mean_PI_width_Wh':float(np.mean((upper-lower)[mask])),
                    'median_PI_width_Wh':float(np.median((upper-lower)[mask])),
                    'mean_CI_width_Wh':float(np.mean((cihi-cilo)[mask])),
                    'PI_lower_negative_pct':float(100*np.mean(lower[mask]<0)),
                    'n_below_PI':int(np.sum(yte[mask]<lower[mask])),
                    'n_above_PI':int(np.sum(yte[mask]>upper[mask]))})
            metrics_rows.append({'model':key,**metrics(yte,preds,len(features)),
                      'PI_coverage':coverage,'PI_mean_width':float(np.mean(upper-lower)),
                      'CI_mean_width':float(np.mean(cihi-cilo)),
                      'negative_PI_lower_count':int(np.sum(lower<0))})
            plot_diagnostics(key,title,test_data,out)
            plot_intervals(key,title,test_data,out)
            print(f'{key}: DW train/test = {diagnostic_rows[-2]["Durbin_Watson"]:.4f}/{diagnostic_rows[-1]["Durbin_Watson"]:.4f}; '
                  f'PI coverage={coverage:.4%}; RMSE={metrics_rows[-1]["RMSE"]:.3f}; max VIF={vifsumm[-1]["max_VIF"]:.2f}',flush=True)
        for name,rows in [('diagnostics_statistics.csv',diagnostic_rows),('vif_train_details.csv',vif_rows),
                          ('vif_model_summary.csv',vifsumm),('coverage_by_load.csv',coverage_rows),
                          ('model_metrics_and_intervals.csv',metrics_rows),('residual_acf.csv',residual_ACF)]:
            dump_csv(pd.DataFrame(rows),out/name)
        # Visualization of actual difference across A14, Env25, B28.
        tbl=pd.DataFrame(metrics_rows)
        fig,axs=plt.subplots(1,2,figsize=(10.8,4.6))
        x=np.arange(3);labels=['A: 14 env','25 env','B: 25+3 lag'];cols=[COLORS[k] for k,_ in CONFIG]
        axs[0].bar(x,tbl['RMSE'],color=cols)
        axs[0].set(title='Test RMSE (lower is better)',ylabel='RMSE (Wh)',xticks=x,xticklabels=labels)
        axs[1].bar(x,tbl['R2'],color=cols)
        axs[1].set(title='Test R² (higher is better)',ylabel='R²',xticks=x,xticklabels=labels)
        for ax in axs:ax.grid(axis='y',alpha=.15)
        fig.suptitle('Locked Step 4 Models: Test Comparison')
        fig.tight_layout();fig.savefig(out/'03_test_metrics_comparison.png',dpi=170);plt.close(fig)
        diagnostics=pd.DataFrame(diagnostic_rows)
        fig,axs=plt.subplots(1,2,figsize=(11.3,4.7));i=np.arange(3);w=.35
        for j,partition in enumerate(['Train','Test']):
            dd=diagnostics[diagnostics.partition==partition].set_index('model').loc[[k for k,_ in CONFIG]]
            axs[0].bar(i+(j-.5)*w,dd.Durbin_Watson.values,w,label=partition,alpha=.88)
            axs[1].bar(i+(j-.5)*w,dd.residual_ACF_lag1.values,w,label=partition,alpha=.88)
        axs[0].axhline(2,color='#A54F32',lw=1.5,ls='--',label='DW=2 (reference)')
        axs[0].set(ylabel='Durbin–Watson',title='Residual independence: DW',xticks=i,xticklabels=['A14','Env25','B28'])
        axs[1].axhline(0,color='#555555',lw=.9)
        axs[1].set(ylabel='Lag-1 residual ACF',title='Residual dependence: 10-minute lag',xticks=i,xticklabels=['A14','Env25','B28'])
        for ax in axs:ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15)
        fig.tight_layout();fig.savefig(out/'04_residual_independence_comparison.png',dpi=170);plt.close(fig)
        cov=pd.DataFrame(coverage_rows)
        groups=['Low: <=100 Wh','Medium: 100-200 Wh','High: 200-400 Wh','Peak: >400 Wh']
        fig,ax=plt.subplots(figsize=(11.6,4.7));ind=np.arange(4);width=.25
        for i,(key,_) in enumerate(CONFIG):
            cs=cov[(cov.model==key)&(cov.group!='All test')].set_index('group').loc[groups]
            ax.bar(ind+(i-1)*width,100*cs.PI_coverage,width,color=COLORS[key],label=key)
        ax.axhline(95,color='#C44D34',ls='--',lw=1.5,label='Nominal 95%')
        ax.set(xticks=ind,xticklabels=['<=100 Wh','100-200 Wh','200-400 Wh','>400 Wh'],ylim=(0,105),
               xlabel='Observed appliance energy load category (Test)',ylabel='Actual 95% PI coverage (%)',
               title='Out-of-sample OLS PI coverage by actual load (diagnostic only)')
        ax.legend(ncol=4,fontsize=8);ax.grid(axis='y',alpha=.13)
        fig.tight_layout();fig.savefig(out/'05_PI_coverage_by_load.png',dpi=170);plt.close(fig)
        vs=pd.DataFrame(vifsumm)
        fig,ax=plt.subplots(figsize=(10,4.6)); bars=ax.bar([k for k,_ in CONFIG],vs.max_VIF, color=[COLORS[k] for k,_ in CONFIG])
        ax.axhline(10,color='#C44D34',ls='--',label='Common VIF flag = 10')
        ax.set(yscale='log',ylabel='Maximum Train VIF (log scale)',title='Multicollinearity by Locked Model')
        for b,value in zip(bars,vs.max_VIF):ax.text(b.get_x()+b.get_width()/2,value*1.16,f'{value:.1f}',ha='center',fontsize=10)
        ax.legend();ax.grid(axis='y',alpha=.16)
        fig.tight_layout();fig.savefig(out/'06_VIF_comparison.png',dpi=170);plt.close(fig)
        manifest={'data_csv':csv_path.name,'step4_directory':step4.name,'split':'chronological Train=13814, Validation=2960, Test=2961',
                  'train_only_ols_refit':'Uses SAME Train data, SAME Step4 StandardScaler, SAME selected features as locked LinearRegression models; only for OLS inference.',
                  'max_abs_prediction_differences_vs_locked_step4':check_predictions,
                  'fixed_models':{'A_best14':names14,'Env25':names25,'B_lag28':names25+LAGS},
                  'lag_availability':'lag1/2/3 are observed y values for t-1/2/3, shifted before time slicing.',
                  'CI_PI':'statsmodels OLS.get_prediction().summary_frame(alpha=0.05), nonrobust normal-IID standard errors',
                  'test_window':'first 432 records=72 hours',
                  'final_model_selection':'Step4 Validation only. Step5 does not choose new model using Test diagnostics.',
                  'warning':'Classical OLS CI/PI validity is conditional on assumptions; significant residual autocorrelation, skewness or heteroskedasticity undermines nominal interpretation.'}
        (out/'step5_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        print('\n==TEST METRICS AND COVERAGE==\n'+pd.DataFrame(metrics_rows).to_string(index=False),flush=True)
        print('\n==DIAGNOSTICS==\n'+pd.DataFrame(diagnostic_rows).to_string(index=False),flush=True)
        print('\n==VIF==\n'+pd.DataFrame(vifsumm).to_string(index=False),flush=True)
        print('\n==COVERAGE BY GROUP==\n'+pd.DataFrame(coverage_rows).to_string(index=False),flush=True)


    main(csv_path,root/"5115056030_step4_outputs",root/"5115056030_step5_outputs")

def run_step6(csv_path: Path, root: Path):
    """Step 6: persisted pipelines and reproducible example inference."""
    """Step 6 | Load the LOCKED Step4 estimators and demonstrate HEMS inference.

    Usage:
     python 5115056030_step6_deployment.py --csv KAG_energydata_complete.csv \
         --step4 5115056030_step4_outputs --out 5115056030_step6_outputs

    Point predictions use the ORIGINAL Step4 model weights. OLS intervals are a
    train-only, numerically equivalent OLS re-fit (traditional IID-normal PI).
    Never load joblib models from untrusted sources.
    """
    import argparse, json
    from pathlib import Path
    import joblib
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error

    LAGS=['Appliances_lag1','Appliances_lag2','Appliances_lag3']


    def deploy(csv_path:Path,step4:Path,out:Path):
        out.mkdir(parents=True,exist_ok=True)
        original=pd.read_csv(csv_path,parse_dates=['date']).sort_values('date').reset_index(drop=True)
        assert len(original)==19735 and original.date.is_unique
        a=int(len(original)*.7);b=int(len(original)*.85)
        assert (a,b)==(13814,16774)
        locked=joblib.load(step4/'locked_final_model.joblib')
        ar=joblib.load(step4/'ar_experiment_models.joblib')
        selected=list(locked['selected_features']); pool=list(locked['feature_pool']);idx=list(locked['indices'])
        assert [pool[i] for i in idx]==selected and len(selected)==14 and len(pool)==25
        # Assemble a sklearn Pipeline with the *already trained* scaler and predictor.
        # ColumnTransformer is fit on TRAIN-transformed features to fix column selection.
        # Pipeline.predict(raw DataFrame in pool order) performs scale -> selection -> regression.
        subselector=ColumnTransformer([('chosen','passthrough',idx)],remainder='drop')
        subselector.fit(locked['scaler'].transform(original.loc[:a-1,pool]))
        pipe_A=Pipeline([('standard_scaler',locked['scaler']),('selected_14_features',subselector),('linear_regression',locked['estimator'])])
        features_B=pool+LAGS
        pipe_B=Pipeline([('standard_scaler',ar['scalers']['EnvironmentPlus3Lags_28']),
                         ('linear_regression',ar['models']['EnvironmentPlus3Lags_28'])])
        joblib.dump(pipe_A,out/'HEMS_environment_14_pipeline.joblib')
        joblib.dump(pipe_B,out/'HEMS_lag_28_pipeline.joblib')
        (out/'deployment_schema.json').write_text(json.dumps({
          'target':'Appliances (Wh per 10-minute observation)',
          'A_feature_pool_order':pool,'A_selected_features':selected,
          'B_features_order':features_B,
          'B_lag_sources':'observed appliances t-1,t-2,t-3 ONLY, never current y',
          'training_boundary':str(original.date.iloc[a-1]),'test_start':str(original.date.iloc[b]),
          'valid_input':'all required numeric columns, finite, complete, same units as training',
          'interval_warning':'statsmodels OLS 95% classical IID normal intervals; not calibrated for time dependence/heteroscedastic peaks',
          'pi_lower_may_be_negative':True,
          'scaler_fit_scope':'Train only; pipeline contains prefit Step4 transformers and estimators'
        },ensure_ascii=False,indent=2),encoding='utf-8')

        # Lag construction before splitting, never taking target at the query timestamp.
        frame=original.copy()
        for lag,name in enumerate(LAGS,1):frame[name]=frame.Appliances.shift(lag)
        frame=frame.dropna(subset=LAGS).reset_index(drop=True)
        test=frame[frame.date>=original.date.iloc[b]].copy()
        train_A=original.iloc[:a];train_B=frame[frame.date<original.date.iloc[a]]
        assert len(test)==2961 and len(train_B)==13811

        # Demonstrations picked by true observed load only for AFTER-THE-FACT illustration;
        # it does not form part of the feature-selection, model tuning or deployment rule.
        low=test[(test.Appliances<=60)&(test.lights==0)].head(3)
        peak=test[(test.Appliances>450)&(test.lights>0)].head(3)
        samples=pd.concat([low,peak]).copy(); assert len(samples)==6
        types=['low/base']*3+['observed_high_peak']*3

        # Classic 95% OLS intervals are fit on original train, never test.
        matA=locked['scaler'].transform(train_A[pool])[:,idx]
        matB=ar['scalers']['EnvironmentPlus3Lags_28'].transform(train_B[features_B])
        mA=sm.OLS(train_A.Appliances.to_numpy(),sm.add_constant(matA,has_constant='add')).fit()
        mB=sm.OLS(train_B.Appliances.to_numpy(),sm.add_constant(matB,has_constant='add')).fit()
        # pipelines are checked against the locked models and train-only OLS, not silently refitted.
        sA=locked['scaler'].transform(samples[pool])[:,idx]
        sB=ar['scalers']['EnvironmentPlus3Lags_28'].transform(samples[features_B])
        pyA=pipe_A.predict(samples[pool]); pyB=pipe_B.predict(samples[features_B])
        ciA=mA.get_prediction(sm.add_constant(sA,has_constant='add')).summary_frame(alpha=.05)
        ciB=mB.get_prediction(sm.add_constant(sB,has_constant='add')).summary_frame(alpha=.05)
        assert np.allclose(pyA,ciA['mean'].to_numpy(),rtol=0,atol=1e-7)
        assert np.allclose(pyB,ciB['mean'].to_numpy(),rtol=0,atol=1e-7)
        assert np.allclose(pyA,locked['estimator'].predict(sA),atol=1e-10)
        assert np.allclose(pyB,ar['models']['EnvironmentPlus3Lags_28'].predict(sB),atol=1e-10)
        outputs=[];contrib=[]
        for j,(pos,row) in enumerate(samples.iterrows()):
            for tag,pipe,preds,ci,xstd,model,features in [
                ('A_environment_14',pipe_A,pyA,ciA,sA,mA,selected),
                ('B_environment_plus_lag_28',pipe_B,pyB,ciB,sB,mB,features_B)]:
                y=float(preds[j]);frame_ci=ci.iloc[j]
                outputs.append({'case':types[j],'date':str(row.date),'model':tag,'actual_Wh':float(row.Appliances),
                    'lights_Wh':float(row.lights),'prediction_Wh':y,'95pct_mean_CI_low':float(frame_ci.mean_ci_lower),
                    '95pct_mean_CI_high':float(frame_ci.mean_ci_upper),
                    '95pct_PI_low':float(frame_ci.obs_ci_lower),'95pct_PI_high':float(frame_ci.obs_ci_upper),
                    'PI_covers_actual':int(frame_ci.obs_ci_lower<=row.Appliances<=frame_ci.obs_ci_upper)})
                values=xstd[j];coefficients=np.asarray(model.params[1:]);contributions=values*coefficients
                intercept=float(model.params[0])
                assert np.isclose(intercept+np.sum(contributions),y,atol=1e-7)
                for name,original_val,z,coef,val in zip(features,[row[k] for k in features],values,coefficients,contributions):
                    contrib.append({'case':types[j],'date':str(row.date),'model':tag,'feature':name,
                        'raw_value':float(original_val),'standardized_value':float(z),
                        'coefficient_per_one_training_SD_Wh':float(coef),
                        'contribution_from_training_mean_Wh':float(val),'intercept_at_training_feature_means_Wh':intercept})
        output=pd.DataFrame(outputs)
        output.to_csv(out/'demo_predictions_95PI.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(contrib).to_csv(out/'demo_feature_contributions.csv',index=False,encoding='utf-8-sig')
        top=pd.DataFrame(contrib)
        # contributions vary by sample; illustration for one high-load event using top 10 abs values.
        sample_date=str(peak.iloc[0].date)
        fig,axes=plt.subplots(1,2,figsize=(13.5,5))
        for ax,tag,short in zip(axes,['A_environment_14','B_environment_plus_lag_28'],['A: Environmental 14','B: Environment + 3 Lags']):
            t=top[(top.date==sample_date)&(top.model==tag)].copy()
            t=t.reindex(t.contribution_from_training_mean_Wh.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
            ax.barh(t.feature,t.contribution_from_training_mean_Wh,color=['#B86650' if z<0 else '#2F7881' for z in t.contribution_from_training_mean_Wh])
            ax.axvline(0,lw=.9,color='#555555')
            ax.set(title=short,xlabel='Coefficient × standardized feature (Wh)')
            ax.grid(axis='x',alpha=.18)
        fig.suptitle('Explanatory Linear Contributions — One High-Load Example (not causal effects)',fontsize=12)
        fig.tight_layout();fig.savefig(out/'01_typical_highload_contributions.png',dpi=160,bbox_inches='tight');plt.close(fig)
        fig,ax=plt.subplots(figsize=(10.5,5))
        x=np.arange(len(samples));width=.35
        aa=output[output.model=='A_environment_14'];bb=output[output.model=='B_environment_plus_lag_28']
        ax.bar(x-width/2,aa.prediction_Wh,width,color='#5286A8',label='A point estimate')
        ax.bar(x+width/2,bb.prediction_Wh,width,color='#D18E60',label='B point estimate')
        ax.scatter(x,samples.Appliances.to_numpy(),color='#222222',marker='D',s=46,zorder=3,label='Actual')
        ax.set(xticks=x,xticklabels=['Low1','Low2','Low3','Peak1','Peak2','Peak3'],ylabel='Appliances (Wh)',title='HEMS Sample Predictions vs Observed Energy')
        ax.legend();ax.grid(axis='y',alpha=.17);fig.tight_layout();fig.savefig(out/'02_demo_point_predictions.png',dpi=160);plt.close(fig)
        # Saved pipelines can be loaded and run with exact same numeric result.
        again_A=joblib.load(out/'HEMS_environment_14_pipeline.joblib')
        again_B=joblib.load(out/'HEMS_lag_28_pipeline.joblib')
        assert np.allclose(again_A.predict(samples[pool]),pyA,rtol=0,atol=1e-9)
        assert np.allclose(again_B.predict(samples[features_B]),pyB,rtol=0,atol=1e-9)
        summary={'sample_selection':'3 first Test entries with actual <=60 and lights=0; 3 first with actual >450 and lights>0; for explanation ONLY, not model selection',
          'scaler_refit':False,'models_refit':False,'interval_models_train_fit_only':True,
          'sample_count':len(samples),'A_features':selected,'B_features':features_B,
          'max_pipeline_vs_statsmodels_diff_A_Wh':float(np.max(abs(pyA-ciA['mean'].to_numpy()))),
          'max_pipeline_vs_statsmodels_diff_B_Wh':float(np.max(abs(pyB-ciB['mean'].to_numpy()))),
          'pipeline_reloaded_verified':True,
          'avoid_CAUSAL_inference':True,'B_observed_lags_required':True}
        (out/'deployment_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(output[['case','date','model','actual_Wh','prediction_Wh','95pct_PI_low','95pct_PI_high']].to_string(index=False))
        return output


    deploy(csv_path,root/"5115056030_step4_outputs",root/"5115056030_step6_outputs")

def main():
    ap=argparse.ArgumentParser(description='CRISP-DM HW2 all six steps; original Kaggle CSV required.')
    ap.add_argument('--csv',type=Path,default=None)
    ap.add_argument('--skip-modeling',action='store_true',help='Use previously saved, locked Step4 models instead of retraining.')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    csv_path=args.csv or (root/'KAG_energydata_complete.csv')
    if not csv_path.is_file(): raise FileNotFoundError(csv_path)
    run_steps_1_to_3(csv_path,root)
    if not args.skip_modeling: run_step4(csv_path,root)
    else:
        if not (root/'5115056030_step4_outputs'/'locked_final_model.joblib').exists():
            raise FileNotFoundError('Cannot skip modeling without trained models.')
    run_step5(csv_path,root)
    run_step6(csv_path,root)
    print('HW2 COMPLETE — see research report for citations, assumptions and academic limitations.')

if __name__=='__main__':
    main()
