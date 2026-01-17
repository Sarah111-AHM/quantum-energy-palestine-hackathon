"""02_collect_nasa_data.py
Generate realistic weather data (no API, just smart simulation)
"""

import pandas as pd, numpy as np, os
from datetime import datetime, timedelta

def simulate_weather_data(sites_df):
    """Simulate 7 days of realistic weather per site"""
    print("Simulating weather data...☁️")
    data=[];today=datetime.now()
    for _,site in sites_df.iterrows():
        zone=site['zone'];site_id=site['site_id']
        base_temp,base_wind=(18+np.random.uniform(-2,2),7+np.random.uniform(-1,1)) if zone=='north' else ((22+np.random.uniform(-2,2),6+np.random.uniform(-1,1)) if zone=='south' else (20+np.random.uniform(-2,2),6.5+np.random.uniform(-1,1)))
        for d in range(7):
            date=today-timedelta(days=d)
            temp=base_temp+np.random.uniform(-1,1);wind=max(1,base_wind+np.random.uniform(-0.5,0.5))
            data.append({'site_id':site_id,'date':date.strftime('%Y-%m-%d'),'temperature_c':round(temp,1),'wind_speed_mps':round(wind,1),'humidity_percent':round(60+np.random.uniform(-10,10),1),'precipitation_mm':round(np.random.uniform(0,3),1)})
    return pd.DataFrame(data)

def create_weather_summary(df):
    """Create clean weather indicators per site"""
    s=df.groupby('site_id').agg({'temperature_c':['mean','std'],'wind_speed_mps':['mean','max'],'humidity_percent':'mean','precipitation_mm':'sum'}).round(2)
    s.columns=['temp_mean','temp_std','wind_mean','wind_max','humidity_mean','precip_total'];s=s.reset_index()
    s['weather_score']=(0.4*(1-abs(20-s['temp_mean'])/20)+0.3*(1-np.minimum(s['wind_max']/15,1))+0.2*(1-abs(60-s['humidity_mean'])/60)+0.1*(1-np.minimum(s['precip_total']/20,1))).round(3)
    return s

def main():
    sites_path='data/raw/candidates_raw.csv'
    if not os.path.exists(sites_path):print("✗ Base sites file missing. Run 01 first.");return
    sites_df=pd.read_csv(sites_path)
    weather_df=simulate_weather_data(sites_df);summary=create_weather_summary(weather_df)
    os.makedirs('data/raw/nasa_power_raw',exist_ok=True);os.makedirs('data/processed',exist_ok=True)
    weather_df.to_csv('data/raw/nasa_power_raw/weather_daily.csv',index=False)
    summary.to_csv('data/processed/weather_summary.csv',index=False)
    print(f"✓ Weather data ready for {len(sites_df)} sites | Raw + Summary saved")

if __name__=="__main__":
    main()
