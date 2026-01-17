"""
map_visualizer.py
Cool tools for interactive maps 🌍
"""

import folium
from folium import plugins
import pandas as pd
import plotly.graph_objects as go
import os

class MapVisualizer:
    """Advanced Humanitarian Map Visualizer"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.base_map = None

    def create_map(self, center=None, zoom=11):
        """Basic interactive map"""
        if center is None:
            center = [self.df['latitude'].mean(), self.df['longitude'].mean()]
        m = folium.Map(location=center, zoom_start=zoom, control_scale=True)
        
        folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
        folium.TileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            name='Satellite',
            attr='Esri',
            overlay=False
        ).add_to(m)
        
        plugins.MeasureControl(primary_length_unit='kilometers').add_to(m)
        plugins.Fullscreen().add_to(m)
        self.base_map = m
        return m

    def add_markers(self, m, color_by='risk', size_by='priority'):
        """Add site markers"""
        def color(val, metric):
            if metric == 'risk':
                return 'red' if val>0.7 else 'orange' if val>0.4 else 'green'
            if metric == 'access':
                return 'green' if val>0.7 else 'yellow' if val>0.4 else 'red'
            return 'blue'
        def size(val, metric):
            if metric=='priority':
                return 15 if val>0.8 else 12 if val>0.6 else 8
            return 10
        for _, s in self.df.iterrows():
            cv = s.get(f'base_{color_by}_score',0.5)
            sv = s.get(f'base_{size_by}_score',0.5)
            popup = self._popup_html(s)
            folium.CircleMarker(
                location=[s['latitude'], s['longitude']],
                radius=size(sv, size_by),
                popup=folium.Popup(popup, max_width=300),
                color=color(cv,color_by),
                fill=True,
                fill_color=color(cv,color_by),
                fill_opacity=0.7
            ).add_to(m)
        return m

    def _popup_html(self, s):
        """Simple popup HTML"""
        return f"""
        <div style="width:250px;font-family:sans-serif;">
        <b>{s.get('name_ar','Site')}</b><br>
        📍 {s.get('region','Unknown')}<br>
        🏥 {s.get('site_type','Unknown')}<br>
        ⚠️ Risk: {s.get('base_risk_score',0)*100:.0f}%<br>
        🚀 Access: {s.get('base_access_score',0)*100:.0f}%<br>
        ⭐ Priority: {s.get('base_priority_score',0)*100:.0f}%<br>
        👥 Pop: {s.get('population_served',0):,}<br>
        💰 Cost: ${s.get('estimated_cost_usd',0):,}<br>
        {'🌟 Score: {:.3f}'.format(s.get('final_score',0)) if 'final_score' in s else ''}
        </div>
        """

    def add_heatmap(self, m, col='final_score'):
        """Add heatmap"""
        data = [[r['latitude'], r['longitude'], r.get(col,1)] for _,r in self.df.iterrows()
                if pd.notna(r['latitude']) and pd.notna(r['longitude'])]
        plugins.HeatMap(data, radius=15, blur=10,
                        gradient={0.4:'blue',0.65:'lime',1:'red'}).add_to(m)
        return m

    def add_clusters(self, m):
        """Add clustered markers"""
        from folium.plugins import MarkerCluster
        cluster = MarkerCluster(name="Clusters").add_to(m)
        for _, s in self.df.iterrows():
            folium.Marker(
                [s['latitude'],s['longitude']],
                popup=folium.Popup(self._popup_html(s),max_width=300),
                icon=folium.Icon(color='blue',icon='info-sign')
            ).add_to(cluster)
        return m

    def add_selected(self, m, df_sel, name="Selected Sites"):
        """Highlight selected sites"""
        if df_sel is None or len(df_sel)==0: return m
        fg = folium.FeatureGroup(name=name)
        for _, s in df_sel.iterrows():
            folium.Circle([s['latitude'],s['longitude']], radius=300,
                          color='#FF5733', fill=True, fill_opacity=0.3,
                          popup=f"✅ {s.get('name_ar','Selected')}").add_to(fg)
        fg.add_to(m)
        return m

    def save_map(self, m, path='results/map.html'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        m.save(path)
        print(f"Map saved: {path}")
        return path

    def create_3d_map(self, path=None):
        """3D map"""
        fig = go.Figure(data=[go.Scatter3d(
            x=self.df['longitude'], y=self.df['latitude'],
            z=self.df.get('final_score',0.5),
            mode='markers',
            marker=dict(size=8,color=self.df.get('final_score',0.5),
                        colorscale='Viridis',opacity=0.8,colorbar=dict(title='Score')),
            text=self.df['name_ar'], hoverinfo='text'
        )])
        fig.update_layout(scene=dict(xaxis_title='Lon',yaxis_title='Lat',zaxis_title='Score'),height=600)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.write_html(path)
            print(f"3D map saved: {path}")
        return fig

# Quick test
def test_map():
    print("Testing map visualizer...")
    path='data/processed/scored_sites.csv'
    if not os.path.exists(path): print("✗ CSV missing"); return
    df=pd.read_csv(path)
    mv=MapVisualizer(df)
    m=mv.create_map(); mv.add_markers(m); mv.add_clusters(m); mv.add_heatmap(m)
    mv.save_map(m)
    mv.create_3d_map('results/3d_map.html')
    print("✅ All done!")

if __name__=="__main__":
    test_map()
