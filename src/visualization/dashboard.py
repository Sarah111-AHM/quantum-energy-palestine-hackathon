# dashboard_simple.py
# Humanitarian AI + Quantum Dashboard (Gaza)

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import folium_static
import folium
import os

st.set_page_config(page_title="Gaza AI+Quantum Sites", layout="wide")

st.title("📍 Humanitarian AI + Quantum Dashboard (Gaza)")
st.markdown("System helps pick best sites for aid using AI & Quantum computing.")

# --- Load Data ---
@st.cache_data
def load_data():
    sites = pd.read_csv('data/processed/master_dataset.csv') if os.path.exists('data/processed/master_dataset.csv') else pd.DataFrame()
    scored = pd.read_csv('data/processed/scored_sites.csv') if os.path.exists('data/processed/scored_sites.csv') else pd.DataFrame()
    selected_q = pd.read_csv('results/quantum_selections/selected_sites.csv') if os.path.exists('results/quantum_selections/selected_sites.csv') else pd.DataFrame()
    return sites, scored, selected_q

sites_df, scored_df, selected_q_df = load_data()

st.sidebar.header("Filters & Weights")
region_filter = st.sidebar.selectbox("Region", ['All'] + sorted(sites_df['region'].unique().tolist()) if not sites_df.empty else ['All'])
site_type_filter = st.sidebar.selectbox("Site Type", ['All'] + sorted(sites_df['site_type'].unique().tolist()) if not sites_df.empty else ['All'])

risk_w = st.sidebar.slider("Risk Weight", 0.0, 1.0, 0.35)
access_w = st.sidebar.slider("Access Weight", 0.0, 1.0, 0.25)
priority_w = st.sidebar.slider("Priority Weight", 0.0, 1.0, 0.30)
weather_w = st.sidebar.slider("Weather Weight", 0.0, 1.0, 0.10)

# --- Overview ---
st.subheader("📊 Overview")
if not sites_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sites", len(sites_df))
    col2.metric("Avg Score", f"{sites_df['final_score'].mean():.2f}" if 'final_score' in sites_df else 0)
    col3.metric("High Risk Sites", len(sites_df[sites_df['base_risk_score']>0.7]) if 'base_risk_score' in sites_df else 0)
    col4.metric("High Priority Sites", len(sites_df[sites_df['base_priority_score']>0.8]) if 'base_priority_score' in sites_df else 0)

# --- Map ---
st.subheader("🗺️ Interactive Map")
if not sites_df.empty:
    map_df = sites_df.copy()
    if region_filter != 'All':
        map_df = map_df[map_df['region']==region_filter]

    map_center = [map_df['latitude'].mean(), map_df['longitude'].mean()] if not map_df.empty else [31.5, 34.5]
    m = folium.Map(location=map_center, zoom_start=12)

    for _, site in map_df.iterrows():
        color = 'red' if site['base_risk_score']>0.7 else 'orange' if site['base_risk_score']>0.4 else 'green'
        folium.CircleMarker(
            [site['latitude'], site['longitude']],
            radius=10,
            color=color,
            fill=True,
            fill_color=color,
            popup=f"{site['name_ar']} | Score: {site['final_score']:.2f}" if 'final_score' in site else site['name_ar']
        ).add_to(m)

    folium_static(m, width=1200, height=600)

# --- Top Sites ---
st.subheader("🏆 Top 10 Sites")
if not sites_df.empty and 'final_score' in sites_df.columns:
    top_sites = sites_df.nlargest(10, 'final_score')[['name_ar', 'region', 'site_type', 'final_score']]
    st.dataframe(top_sites)

st.markdown("---")
st.markdown("⚡ Dashboard simplified for presentation. Use sidebar to filter and adjust weights.")
