"""
Enhanced Map Visualization Module for Gaza Strip Energy Sites.
Creates interactive maps with risk zones, selected sites, and infrastructure.
"""

import folium
from folium import plugins
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import branca.colormap as cm
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MapConfig:
    """Configuration for map visualization."""
    center_lat: float = 31.4167
    center_lon: float = 34.3333
    zoom_start: int = 10
    tiles: str = 'CartoDB positron'
    api_key: Optional[str] = None  # For satellite tiles
    
    # Visual settings
    marker_scale: float = 1.0
    opacity: float = 0.7
    show_legend: bool = True
    show_scale: bool = True
    
    # Risk zones
    show_risk_heatmap: bool = True
    risk_threshold_high: float = 7.0
    risk_threshold_medium: float = 4.0
    
    # Clustering
    cluster_markers: bool = False
    max_cluster_radius: int = 80
    
    # Gaza boundaries (approximate)
    gaza_bounds = {
        'min_lat': 31.25,
        'max_lat': 31.58,
        'min_lon': 34.20,
        'max_lon': 34.55
    }


class GazaMapVisualizer:
    """
    Professional map visualizer for Gaza Strip energy infrastructure.
    
    Features:
    - Interactive site markers with popup info
    - Risk zone heatmaps
    - Selected site highlighting
    - Infrastructure overlays
    - Distance measurements
    - Export to HTML
    """
    
    def __init__(self, config: Optional[MapConfig] = None):
        """
        Initialize map visualizer.
        
        Args:
            config: Map configuration (uses defaults if None)
        """
        self.config = config or MapConfig()
        self.map = None
        self.markers = []
        self.layers = {}
        
        logger.info(f"Initialized GazaMapVisualizer at ({self.config.center_lat}, {self.config.center_lon})")
    
    def create_base_map(self) -> folium.Map:
        """
        Create base map with specified configuration.
        
        Returns:
            Folium map object
        """
        # Select tiles based on configuration
        if self.config.tiles == 'satellite' and self.config.api_key:
            # Use satellite tiles with API key
            self.map = folium.Map(
                location=[self.config.center_lat, self.config.center_lon],
                zoom_start=self.config.zoom_start,
                tiles=f'https://api.mapbox.com/v4/mapbox.satellite/{{z}}/{{x}}/{{y}}.png?access_token={self.config.api_key}',
                attr='Mapbox'
            )
        else:
            # Use standard tiles
            self.map = folium.Map(
                location=[self.config.center_lat, self.config.center_lon],
                zoom_start=self.config.zoom_start,
                tiles=self.config.tiles
            )
        
        # Add scale bar
        if self.config.show_scale:
            plugins.MeasureControl(
                position='bottomleft',
                primary_length_unit='kilometers'
            ).add_to(self.map)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(self.map)
        
        # Add draw tools for custom measurements
        plugins.Draw(
            export=True,
            filename='measurement.geojson',
            position='topleft',
            draw_options={
                'polyline': {'allowIntersection': False},
                'polygon': False,
                'circle': False,
                'rectangle': False,
                'marker': False,
                'circlemarker': False
            }
        ).add_to(self.map)
        
        logger.info("Base map created")
        return self.map
    
    def add_site_markers(
        self,
        df: pd.DataFrame,
        selected_indices: Optional[List[int]] = None,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude',
        name_col: str = 'Region_ID',
        risk_col: str = 'Risk_Score',
        solar_col: str = 'Solar_Irradiance',
        wind_col: str = 'Wind_Speed',
        access_col: str = 'Accessibility',
        score_col: Optional[str] = None
    ):
        """
        Add site markers to map with color coding.
        
        Args:
            df: DataFrame with site data
            selected_indices: Indices of selected sites
            lat_col: Latitude column name
            lon_col: Longitude column name
            name_col: Region ID column name
            risk_col: Risk score column
            solar_col: Solar irradiance column
            wind_col: Wind speed column
            access_col: Accessibility column
            score_col: Suitability score column (optional)
        """
        if self.map is None:
            self.create_base_map()
        
        # Create feature groups for different types
        selected_group = folium.FeatureGroup(name='✅ Selected Sites')
        candidate_group = folium.FeatureGroup(name='⚡ Candidate Sites')
        restricted_group = folium.FeatureGroup(name='🚫 Restricted Zones')
        high_risk_group = folium.FeatureGroup(name='⚠️ High Risk Areas')
        
        selected_set = set(selected_indices) if selected_indices else set()
        
        # Color maps
        risk_cmap = cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=0,
            vmax=10,
            caption='Risk Level'
        )
        
        for idx, row in df.iterrows():
            # Determine marker type and style
            if idx in selected_set:
                color = '#10b981'  # Emerald green
                icon = 'check-circle'
                group = selected_group
                prefix = 'fa'
                radius = 10
            elif row[access_col] == 0:
                color = '#9ca3af'  # Gray
                icon = 'ban'
                group = restricted_group
                prefix = 'fa'
                radius = 6
            elif row[risk_col] > self.config.risk_threshold_high:
                color = '#ef4444'  # Red
                icon = 'exclamation-triangle'
                group = high_risk_group
                prefix = 'fa'
                radius = 7
            else:
                color = '#3b82f6'  # Blue
                icon = 'solar-panel'
                group = candidate_group
                prefix = 'fa'
                radius = 8
            
            # Create popup content
            popup_html = self._create_popup_content(
                row, name_col, risk_col, solar_col, wind_col, score_col, color
            )
            
            popup = folium.Popup(popup_html, max_width=300)
            
            # Create tooltip
            tooltip = f"{row[name_col]} - Risk: {row[risk_col]}/10"
            
            if idx in selected_set:
                tooltip = f"✅ SELECTED: {tooltip}"
            
            # Add marker
            if self.config.cluster_markers:
                # Use circle markers for clustering
                marker = folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=self.config.opacity,
                    popup=popup,
                    tooltip=tooltip
                )
            else:
                # Use FontAwesome icons
                marker = folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=popup,
                    tooltip=tooltip,
                    icon=folium.Icon(
                        color=self._get_folium_color(color),
                        icon=icon,
                        prefix=prefix
                    )
                )
            
            marker.add_to(group)
            self.markers.append(marker)
        
        # Add groups to map
        selected_group.add_to(self.map)
        candidate_group.add_to(self.map)
        high_risk_group.add_to(self.map)
        restricted_group.add_to(self.map)
        
        # Store layers
        self.layers['selected'] = selected_group
        self.layers['candidate'] = candidate_group
        self.layers['high_risk'] = high_risk_group
        self.layers['restricted'] = restricted_group
        
        # Add layer control
        folium.LayerControl().add_to(self.map)
        
        # Add colorbar if requested
        if self.config.show_legend:
            risk_cmap.add_to(self.map)
        
        logger.info(f"Added {len(df)} markers to map")
    
    def _create_popup_content(
        self,
        row: pd.Series,
        name_col: str,
        risk_col: str,
        solar_col: str,
        wind_col: str,
        score_col: Optional[str],
        color: str
    ) -> str:
        """Create HTML popup content for marker."""
        
        # Risk level class
        risk = row[risk_col]
        if risk > 7:
            risk_class = 'risk-high'
        elif risk > 4:
            risk_class = 'risk-medium'
        else:
            risk_class = 'risk-low'
        
        # Build HTML
        html = f"""
        <div style="font-family: 'Helvetica Neue', sans-serif; min-width: 220px;">
            <h4 style="margin:0 0 10px 0; color:{color}; border-bottom:2px solid {color}; padding-bottom:5px;">
                {row[name_col]}
            </h4>
            <table style="width:100%; font-size:13px;">
                <tr>
                    <td style="padding:3px 0;">☀️ Solar:</td>
                    <td style="padding:3px 0; font-weight:bold;">{row[solar_col]:.2f} kWh</td>
                </tr>
                <tr>
                    <td style="padding:3px 0;">💨 Wind:</td>
                    <td style="padding:3px 0; font-weight:bold;">{row[wind_col]:.2f} m/s</td>
                </tr>
                <tr>
                    <td style="padding:3px 0;">⚠️ Risk:</td>
                    <td style="padding:3px 0; font-weight:bold;" class="{risk_class}">{risk}/10</td>
                </tr>
        """
        
        if score_col and score_col in row:
            html += f"""
                <tr>
                    <td style="padding:3px 0;">📊 Score:</td>
                    <td style="padding:3px 0; font-weight:bold;">{row[score_col]:.1f}%</td>
                </tr>
            """
        
        html += """
            </table>
            <div style="margin-top:8px; font-size:11px; color:#666; text-align:center;">
                Click for details
            </div>
        </div>
        """
        
        return html
    
    def _get_folium_color(self, hex_color: str) -> str:
        """Convert hex color to folium color name."""
        color_map = {
            '#10b981': 'green',
            '#3b82f6': 'blue',
            '#ef4444': 'red',
            '#9ca3af': 'gray',
            '#f59e0b': 'orange',
            '#8b5cf6': 'purple'
        }
        return color_map.get(hex_color, 'blue')
    
    def add_risk_heatmap(
        self,
        df: pd.DataFrame,
        risk_col: str = 'Risk_Score',
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ):
        """
        Add risk heatmap layer.
        
        Args:
            df: DataFrame with site data
            risk_col: Risk score column
            lat_col: Latitude column
            lon_col: Longitude column
        """
        if self.map is None:
            self.create_base_map()
        
        # Prepare data for heatmap
        heat_data = []
        for _, row in df.iterrows():
            heat_data.append([
                row[lat_col],
                row[lon_col],
                row[risk_col]  # Weight by risk
            ])
        
        # Create heatmap
        heatmap = plugins.HeatMap(
            heat_data,
            name='Risk Heatmap',
            min_opacity=0.3,
            max_zoom=12,
            radius=15,
            blur=10,
            gradient={
                0.0: 'green',
                0.5: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        )
        
        heatmap.add_to(self.map)
        self.layers['heatmap'] = heatmap
        
        logger.info("Added risk heatmap layer")
    
    def add_infrastructure_overlay(
        self,
        infrastructure_file: Optional[str] = None,
        infrastructure_data: Optional[Dict] = None
    ):
        """
        Add existing infrastructure overlay (power lines, roads, etc.).
        
        Args:
            infrastructure_file: GeoJSON file path
            infrastructure_data: GeoJSON data dictionary
        """
        if self.map is None:
            self.create_base_map()
        
        # Load infrastructure data
        if infrastructure_file and os.path.exists(infrastructure_file):
            with open(infrastructure_file, 'r') as f:
                infrastructure_data = json.load(f)
        
        if infrastructure_data:
            # Add power lines
            if 'power_lines' in infrastructure_data:
                folium.GeoJson(
                    infrastructure_data['power_lines'],
                    name='Power Lines',
                    style_function=lambda x: {
                        'color': '#f59e0b',
                        'weight': 3,
                        'opacity': 0.7,
                        'dashArray': '5, 5'
                    }
                ).add_to(self.map)
            
            # Add roads
            if 'roads' in infrastructure_data:
                folium.GeoJson(
                    infrastructure_data['roads'],
                    name='Road Network',
                    style_function=lambda x: {
                        'color': '#6b7280',
                        'weight': 2,
                        'opacity': 0.5
                    }
                ).add_to(self.map)
            
            # Add hospitals/schools
            if 'critical_facilities' in infrastructure_data:
                for facility in infrastructure_data['critical_facilities']:
                    folium.Marker(
                        location=[facility['lat'], facility['lon']],
                        popup=facility['name'],
                        icon=folium.Icon(
                            color='red',
                            icon='plus-square',
                            prefix='fa'
                        )
                    ).add_to(self.map)
            
            logger.info("Added infrastructure overlay")
    
    def add_distance_circles(
        self,
        center_idx: int,
        df: pd.DataFrame,
        radii_km: List[float] = [5, 10, 20],
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ):
        """
        Add distance circles around a center point.
        
        Args:
            center_idx: Index of center site
            df: DataFrame with site data
            radii_km: List of radii in kilometers
            lat_col: Latitude column
            lon_col: Longitude column
        """
        if self.map is None:
            self.create_base_map()
        
        center = df.iloc[center_idx]
        center_loc = [center[lat_col], center[lon_col]]
        
        for radius in radii_km:
            folium.Circle(
                location=center_loc,
                radius=radius * 1000,  # Convert to meters
                color='#3b82f6',
                weight=1,
                fill=False,
                popup=f"{radius}km radius",
                opacity=0.3
            ).add_to(self.map)
        
        logger.info(f"Added distance circles around site {center_idx}")
    
    def add_region_boundaries(self):
        """Add Gaza Strip region boundaries."""
        if self.map is None:
            self.create_base_map()
        
        # Draw approximate Gaza boundary
        bounds = self.config.gaza_bounds
        
        # Create boundary polygon
        boundary_coords = [
            [bounds['min_lat'], bounds['min_lon']],
            [bounds['min_lat'], bounds['max_lon']],
            [bounds['max_lat'], bounds['max_lon']],
            [bounds['max_lat'], bounds['min_lon']],
            [bounds['min_lat'], bounds['min_lon']]
        ]
        
        folium.Polygon(
            locations=boundary_coords,
            color='#374151',
            weight=2,
            fill=False,
            dashArray='5, 5',
            popup='Gaza Strip Boundary',
            tooltip='Gaza Strip'
        ).add_to(self.map)
        
        # Add region labels
        regions = {
            'North Gaza': [31.55, 34.35],
            'Gaza City': [31.50, 34.45],
            'Deir al-Balah': [31.42, 34.40],
            'Khan Younis': [31.35, 34.30],
            'Rafah': [31.28, 34.25]
        }
        
        for region, loc in regions.items():
            folium.Marker(
                location=loc,
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10pt; color:#374151; font-weight:bold;">{region}</div>'
                )
            ).add_to(self.map)
        
        logger.info("Added region boundaries")
    
    def add_minimap(self):
        """Add minimap for navigation."""
        if self.map is None:
            self.create_base_map()
        
        minimap = plugins.MiniMap(
            toggle_display=True,
            position='bottomright',
            zoom_level_offset=-5
        )
        self.map.add_child(minimap)
        
        logger.info("Added minimap")
    
    def add_legend(self):
        """Add custom legend to map."""
        if self.map is None:
            self.create_base_map()
        
        legend_html = '''
        <div style="
            position: fixed;
            bottom: 50px;
            right: 50px;
            z-index: 1000;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 12px;
        ">
            <h4 style="margin:0 0 10px 0; color:#111827;">Legend</h4>
            <p><span style="color:#10b981;">●</span> Selected Sites</p>
            <p><span style="color:#3b82f6;">●</span> Candidate Sites</p>
            <p><span style="color:#ef4444;">●</span> High Risk Areas</p>
            <p><span style="color:#9ca3af;">●</span> Restricted Zones</p>
            <hr style="margin:10px 0;">
            <p><span style="color:#f59e0b;">━━</span> Power Lines</p>
            <p><span style="color:#6b7280;">━━</span> Roads</p>
        </div>
        '''
        
        self.map.get_root().html.add_child(folium.Element(legend_html))
        
        logger.info("Added legend")
    
    def save_map(self, filename: str = 'gaza_energy_map.html'):
        """
        Save map to HTML file.
        
        Args:
            filename: Output filename
        """
        if self.map:
            self.map.save(filename)
            logger.info(f"Map saved to {filename}")
            return filename
        return None
    
    def get_map(self) -> Optional[folium.Map]:
        """Get the current map object."""
        return self.map


# Utility function for quick map creation
def create_energy_map(
    df: pd.DataFrame,
    selected_indices: Optional[List[int]] = None,
    show_heatmap: bool = True,
    show_infrastructure: bool = False,
    save: bool = False,
    filename: str = 'energy_map.html'
) -> folium.Map:
    """
    Quick map creation utility.
    
    Example:
        >>> m = create_energy_map(df, selected_indices=[0,1,2])
    """
    visualizer = GazaMapVisualizer()
    visualizer.create_base_map()
    
    # Add sites
    visualizer.add_site_markers(df, selected_indices)
    
    # Add heatmap
    if show_heatmap:
        visualizer.add_risk_heatmap(df)
    
    # Add boundaries
    visualizer.add_region_boundaries()
    
    # Add minimap
    visualizer.add_minimap()
    
    # Add legend
    visualizer.add_legend()
    
    if save:
        visualizer.save_map(filename)
    
    return visualizer.get_map()


# Example usage
if __name__ == "__main__":
    # Create sample data
    import pandas as pd
    
    # Load your data
    try:
        df = pd.read_csv('../../data/gaza_energy_data.csv')
        
        # Create map
        visualizer = GazaMapVisualizer()
        visualizer.create_base_map()
        visualizer.add_site_markers(df, selected_indices=[0, 5, 10])
        visualizer.add_risk_heatmap(df)
        visualizer.add_region_boundaries()
        visualizer.add_minimap()
        visualizer.add_legend()
        
        # Save
        visualizer.save_map('sample_map.html')
        print("Map created successfully!")
        
    except FileNotFoundError:
        print("Data file not found. Please check the path.")
