"""Step 6 | Load the LOCKED Step4 estimators and demonstrate HEMS inference.

Usage:
 python 5115056030_step6_deployment.py --csv KAG_energydata_complete.csv \
     --step4 5115056030_step4_outputs --out 5115056030_step6_outputs

Point predictions use the ORIGINAL Step4 model weights. OLS intervals are a
train-only, numerically equivalent OLS re-fit (traditional IID-normal PI).
Never load joblib models from untrusted sources.
"""
from __future__ import annotations
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

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--csv',type=Path,default=Path('KAG_energydata_complete.csv'))
    ap.add_argument('--step4',type=Path,default=Path('5115056030_step4_outputs'))
    ap.add_argument('--out',type=Path,default=Path('5115056030_step6_outputs'))
    args=ap.parse_args();deploy(args.csv,args.step4,args.out)
