"""04_merge_datasets.py
Merge everything into one master decision dataset
"""

import pandas as pd, numpy as np, os
from datetime import datetime

def load_all_datasets():
    print("Loading datasets...📦")
    d={}
    p_sites='data/processed/candidates_enhanced.csv'
    if not os.path.exists(p_sites):print("✗ Enhanced sites missing");return None
    d['sites']=pd.read_csv(p_sites);print(f"✓ Sites loaded:{len(d['sites'])}")
    p_weather='data/processed/weather_summary.csv'
    d['weather']=pd.read_csv(p_weather) if os.path.exists(p_weather) else None
    print("✓ Weather loaded" if d['weather'] is not None else "⚠️ Weather missing")
    p_dist='data/processed/distance_matrix.npy'
    d['distances']=np.load(p_dist) if os.path.exists(p_dist) else None
    print("✓ Distance matrix loaded" if d['distances'] is not None else "⚠️ Distance matrix missing")
    return d

def calculate_scores(df):
    W={'risk':0.35,'access':0.25,'priority':0.30,'weather':0.10}
    df['base_score']=(W['risk']*(1-df['base_risk_score'])+W['access']*df['base_access_score']+W['priority']*df['base_priority_score']).round(3)
    df['total_score']=(df['base_score']*0.9+df['weather_score']*0.1).round(3) if 'weather_score' in df.columns else df['base_score']
    df['rank']=df['total_score'].rank(ascending=False,method='min').astype(int)
    return df

def add_distance_features(df,mat):
    n=len(df);rows=[]
    for i in range(n):
        d=mat[i];od=d[d>0]
        if len(od)>=5:m=np.sort(od)[:5];rows.append({'avg_distance_km':round(m.mean(),2),'nearest_distance_km':round(m[0],2)})
        else:rows.append({'avg_distance_km':round(od.mean(),2) if len(od)>0 else 10,'nearest_distance_km':round(od.min(),2) if len(od)>0 else 10})
    return pd.concat([df,pd.DataFrame(rows)],axis=1)

def add_derived_features(df):
    df['cost_efficiency']=(df['population_served']/df['estimated_cost_usd']).round(6)
    df['risk_priority_ratio']=(df['base_priority_score']/(df['base_risk_score']+0.01)).round(3)
    df['readiness_index']=(df['base_access_score']*0.4+df['weather_score']*0.3+(1-df['base_risk_score'])*0.3).round(3)
    df['score_band']=pd.cut(df['total_score'],bins=[0,0.4,0.6,0.8,1.0],labels=['low','medium','high','excellent'])
    df['last_processed']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return df

def organize(df):
    order=['site_id','name','region','zone','site_type','latitude','longitude','base_risk_score','risk_level','base_access_score','access_level','base_priority_score','weather_score','temp_mean','wind_max','base_score','total_score','rank','score_band','urgency_index','readiness_index','risk_priority_ratio','avg_distance_km','nearest_distance_km','population_served','estimated_cost_usd','cost_efficiency','is_critical','needs_special_access','data_source','creation_date','last_processed']
    keep=[c for c in order if c in df.columns];extra=[c for c in df.columns if c not in keep]
    return df[keep+extra]

def main():
    print("="*55);print("MASTER DATASET BUILD | Gaza AI Pipeline");print("="*55)
    d=load_all_datasets()
    if d is None:print("Pipeline stopped");return
    m=d['sites'].copy()
    if d['weather'] is not None:m=pd.merge(m,d['weather'],on='site_id',how='left')
    else:m[['weather_score','temp_mean','wind_max']]=[0.7,20.0,6.0]
    m=calculate_scores(m)
    if d['distances'] is not None:m=add_distance_features(m,d['distances'])
    m=add_derived_features(m);m=organize(m)
    os.makedirs('data/processed',exist_ok=True)
    m.to_csv('data/processed/master_dataset.csv',index=False,encoding='utf-8-sig')
    m.to_excel('data/processed/master_dataset.xlsx',index=False)
    print(f"✓ Master dataset ready | rows:{len(m)} cols:{len(m.columns)}")
    print(f"Top 5 sites:\n{m.nlargest(5,'total_score')[['site_id','name','total_score','region','site_type']].to_string(index=False)}")

if __name__=="__main__":
    main()
