"""01_fetch_osm_data.py
Create realistic locations in Gaza
"""

import pandas as pd, numpy as np, os
from datetime import datetime

def create_gaza_locations(output_path='data/raw/candidates_raw.csv'):
    """Generate 50 realistic sites in Gaza"""
    print("Creating location data...🚀")
    regions=[('North_Gaza',31.55,34.50,'north'),('Gaza_City',31.50,34.46,'central'),('Central',31.45,34.40,'central'),('Khan_Younis',31.34,34.30,'south'),('Rafah',31.29,34.25,'south')]
    site_types=[('hospital','Hospital',0.9,0.6,200000),('school','School',0.8,0.7,80000),('camp','Camp',0.9,0.4,150000),('aid_center','Aid Center',0.7,0.5,100000),('water_station','Water Station',0.8,0.6,120000)]
    data=[]
    for i in range(50):
        r_idx,t_idx=i%len(regions),i%len(site_types)
        r_name,base_lat,base_lon,zone=regions[r_idx]
        type_key,type_name,base_prio,base_acc,base_cost=site_types[t_idx]
        lat=base_lat+np.random.uniform(-0.01,0.01);lon=base_lon+np.random.uniform(-0.01,0.01)
        risk,access=(np.random.uniform(0.6,0.9),np.random.uniform(0.3,0.6)) if zone=='north' else ((np.random.uniform(0.5,0.8),np.random.uniform(0.4,0.7)) if zone=='south' else (np.random.uniform(0.3,0.6),np.random.uniform(0.5,0.8)))
        priority=base_prio+np.random.uniform(-0.1,0.1);cost=base_cost+np.random.uniform(-30000,30000)
        data.append({'site_id':i+1,'name':f"{type_name} in {r_name.replace('_',' ')}",'latitude':round(lat,6),'longitude':round(lon,6),'region':r_name,'site_type':type_key,'zone':zone,'risk_score':round(max(0.1,min(0.95,risk)),3),'access_score':round(max(0.1,min(0.95,access)),3),'priority_score':round(max(0.3,min(1.0,priority)),3),'estimated_cost_usd':int(cost),'population_served':np.random.randint(1000,20000),'data_source':'synthetic_realistic','creation_date':datetime.now().strftime('%Y-%m-%d')})
    df=pd.DataFrame(data);os.makedirs(os.path.dirname(output_path),exist_ok=True);df.to_csv(output_path,index=False,encoding='utf-8-sig')
    print(f"✓ Created {len(df)} sites at {output_path} | Regions:{df['region'].nunique()} | Types:{df['site_type'].nunique()}")
    return df

if __name__=="__main__":
    df=create_gaza_locations()
    print("\nSample data:")
    print(df[['site_id','name','region','site_type','risk_score','access_score']].head())
