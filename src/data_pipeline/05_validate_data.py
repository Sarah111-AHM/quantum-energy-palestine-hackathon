"""05_validate_data.py
Validate data quality and consistency
"""

import pandas as pd, numpy as np, os, json
from datetime import datetime

def load_master_dataset():
    p='data/processed/master_dataset.csv'
    if not os.path.exists(p):print(f"✗ Dataset missing:{p}");return None
    try:df=pd.read_csv(p);print(f"✓ Loaded {len(df)} rows");return df
    except Exception as e:print(f"✗ Load failed:{e}");return None

def check_integrity(df):
    print("\n"+"="*55);print("DATA INTEGRITY CHECK");print("="*55)
    c={}
    miss=df.isnull().sum();miss_cols=miss[miss>0]
    c['missing']={'total':int(miss.sum()),'columns':len(miss_cols),'percent':round(miss.sum()/(len(df)*len(df.columns))*100,2)}
    print(f"Missing values:{c['missing']['total']}")
    dup=df.duplicated().sum()
    c['duplicates']={'rows':int(dup),'percent':round(dup/len(df)*100,2)}
    print(f"Duplicate rows:{dup}")
    ranges=[]
    ranges.append(("Coordinates inside Gaza",df['latitude'].between(31.2,31.6).all() and df['longitude'].between(34.2,34.55).all()))
    for col in ['base_risk_score','base_access_score','base_priority_score','weather_score','total_score']:
        if col in df.columns:ranges.append((f"{col} in [0,1]",df[col].between(0,1).all()))
    for col in ['population_served','estimated_cost_usd']:
        if col in df.columns:ranges.append((f"{col} positive",(df[col]>=0).all()))
    c['ranges']={'total':len(ranges),'passed':sum(p for _,p in ranges),'details':ranges}
    for n,p in ranges:print(f"{'✓' if p else '✗'} {n}")
    cons=[]
    if all(k in df.columns for k in ['base_risk_score','base_access_score','base_priority_score','base_score']):
        w=0.35*(1-df['base_risk_score'])+0.25*df['base_access_score']+0.30*df['base_priority_score']
        cons.append(("Base score consistency",(abs(df['base_score']-w).mean()<0.1)))
    if all(k in df.columns for k in ['rank','total_score']):
        cons.append(("Ranking correctness",(df.sort_values('total_score',ascending=False)['rank'].values==np.arange(1,len(df)+1)).all()))
    c['consistency']={'total':len(cons),'passed':sum(p for _,p in cons),'details':cons}
    for n,p in cons:print(f"{'✓' if p else '✗'} {n}")
    return c

def generate_report(df,c):
    print("\n"+"="*55);print("DATA QUALITY REPORT");print("="*55)
    total_checks=c['ranges']['total']+c['consistency']['total']
    passed=c['ranges']['passed']+c['consistency']['passed']
    score=round((passed/total_checks)*100,1) if total_checks>0 else 0
    r={'generated_at':datetime.now().isoformat(),'rows':len(df),'columns':len(df.columns),'quality_score':score,'missing_percent':c['missing']['percent'],'duplicate_percent':c['duplicates']['percent'],'recommendations':[]}
    if c['missing']['percent']>5:r['recommendations'].append("Reduce missing values")
    if c['duplicates']['rows']>0:r['recommendations'].append("Remove duplicate rows")
    if score<80:r['recommendations'].append("Full data review recommended")
    if not r['recommendations']:r['recommendations'].append("Data quality is solid. Ready for next stage.")
    os.makedirs('data/processed',exist_ok=True)
    with open('data/processed/data_quality_report.json','w',encoding='utf-8') as f:json.dump(r,f,indent=2)
    print(f"Quality score:{score}%")
    print("Top 3 sites:")
    print(df.nlargest(3,'total_score')[['site_id','name','total_score','region']].to_string(index=False))
    return r

def clean_dataset(df,c):
    d=df.copy()
    if c['duplicates']['rows']>0:d=d.drop_duplicates()
    for col in d.columns[d.isnull().any()]:
        if d[col].dtype in ['float64','int64']:d[col]=d[col].fillna(d[col].median())
        else:d[col]=d[col].fillna(d[col].mode()[0] if not d[col].mode().empty else 'unknown')
    d.to_csv('data/processed/master_dataset_clean.csv',index=False,encoding='utf-8-sig')
    print(f"✓ Clean dataset saved | rows:{len(d)}")
    return d

def main():
    print("="*55);print("DATA VALIDATION | Gaza AI Pipeline");print("="*55)
    df=load_master_dataset()
    if df is None:return
    checks=check_integrity(df)
    report=generate_report(df,checks)
    if checks['missing']['percent']>5 or checks['duplicates']['rows']>0:clean_dataset(df,checks)
    print("="*55);print("Validation completed");print("="*55)
    if report['quality_score']>=90:print("🎉 Excellent data quality")
    elif report['quality_score']>=75:print("👍 Good quality")
    elif report['quality_score']>=60:print("⚠️ Acceptable but needs improvement")
    else:print("❌ Data quality not acceptable")

if __name__=="__main__":
    main()
