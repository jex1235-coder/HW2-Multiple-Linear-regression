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
from __future__ import annotations
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

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',type=Path,default=Path('KAG_energydata_complete.csv'))
    ap.add_argument('--step4',type=Path,default=Path('5115056030_step4_outputs'))
    ap.add_argument('--out',type=Path,default=Path('5115056030_step5_outputs'))
    args=ap.parse_args()
    main(args.csv,args.step4,args.out)
