"""03_enrich_locations.py
Enrich site data with smart features and distances
"""

import pandas as pd, numpy as np, os
from geopy.distance import geodesic

def calculate_distances(df):
    """Compute distance matrix between all sites"""
    print("Calculating site-to-site distances...📏")
    n=len(df);dist=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            c1=(df.iloc[i]['latitude'],df.iloc[i]['longitude'])
            c2=(df.iloc[j]['latitude'],df.iloc[j]['longitude'])
            d=geodesic(c1,c2).km;dist[i,j]=d;dist[j,i]=d
    return dist

def add_site_features(df):
    """Add decision-ready features"""
    print("Enriching site features...✨")
    e=df.copy()
    e['urgency_index']=(e['base_priority_score']*0.6+e['base_risk_score']*0.4).round(3)
    e['risk_level']=pd.cut(e['base_risk_score'],bins=[0,0.3,0.6,1.0],labels=['low','medium','high'])
    e['access_level']=pd.cut(e['base_access_score'],bins=[0,0.4,0.7,1.0],labels=['hard','medium','easy'])
    e['is_critical']=e['site_type'].isin(['hospital','camp'])
    e['needs_special_access']=e['access_level']=='hard'
    return e

def main():
    sites_path='data/raw/candidates_raw.csv';weather_path='data/processed/weather_summary.csv'
    if not os.path.exists(sites_path):print("✗ Base sites file missing");return
    sites=pd.read_csv(sites_path);weather=pd.read_csv(weather_path) if os.path.exists(weather_path) else None
    enriched=add_site_features(sites)
    if weather is not None:enriched=pd.merge(enriched,weather,on='site_id',how='left')
    dist=calculate_distances(enriched)
    os.makedirs('data/processed',exist_ok=True)
    enriched.to_csv('data/processed/candidates_enhanced.csv',index=False,encoding='utf-8-sig')
    np.save('data/processed/distance_matrix.npy',dist)
    print(f"✓ Enriched {len(enriched)} sites | Features + Distances ready")

if __name__=="__main__":
    main()
