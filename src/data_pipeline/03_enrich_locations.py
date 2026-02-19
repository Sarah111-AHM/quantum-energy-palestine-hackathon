"""
Enrich location data with risk zones and population information.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import logging
from typing import Optional, Dict, List
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocationEnricher:
    """
    Enrich location data with additional attributes.
    """
    
    def __init__(self, data_dir: str = '../../data'):
        """
        Initialize enricher.
        
        Args:
            data_dir: Directory with data files
        """
        self.data_dir = Path(data_dir)
        self.risk_zones = None
        self.population_data = None
        self.infrastructure = None
        
    def load_risk_zones(self, file_path: Optional[str] = None) -> bool:
        """
        Load risk zone GeoJSON.
        
        Args:
            file_path: Path to risk zones file
            
        Returns:
            True if loaded successfully
        """
        if file_path is None:
            file_path = self.data_dir / 'risk_zones.geojson'
        
        try:
            self.risk_zones = gpd.read_file(file_path)
            logger.info(f"Loaded {len(self.risk_zones)} risk zones")
            return True
        except Exception as e:
            logger.error(f"Failed to load risk zones: {e}")
            
            # Create synthetic risk zones
            self._create_synthetic_risk_zones()
            return False
    
    def _create_synthetic_risk_zones(self):
        """Create synthetic risk zones for Gaza."""
        logger.info("Creating synthetic risk zones...")
        
        # Define Gaza bounding box
        min_lat, max_lat = 31.25, 31.58
        min_lon, max_lon = 34.20, 34.55
        
        # Create grid of zones
        n_zones = 20
        lats = np.linspace(min_lat, max_lat, 5)
        lons = np.linspace(min_lon, max_lon, 4)
        
        zones = []
        for i, lat in enumerate(lats[:-1]):
            for j, lon in enumerate(lons[:-1]):
                # Create polygon
                polygon = gpd.GeoDataFrame({
                    'geometry': gpd.points_from_xy(
                        [lon, lons[j+1], lons[j+1], lon, lon],
                        [lat, lat, lats[i+1], lats[i+1], lat]
                    )
                })
                
                # Risk level based on location
                # North and borders = higher risk
                if lat > 31.5 or lat < 31.3 or lon > 34.5 or lon < 34.25:
                    risk = np.random.randint(7, 11)
                elif lat > 31.45 or lat < 31.35:
                    risk = np.random.randint(4, 8)
                else:
                    risk = np.random.randint(2, 6)
                
                zones.append({
                    'zone_id': f'Z{i}_{j}',
                    'risk_level': risk,
                    'geometry': polygon.geometry.iloc[0]
                })
        
        self.risk_zones = gpd.GeoDataFrame(zones)
        logger.info(f"Created {len(self.risk_zones)} synthetic risk zones")
    
    def load_population_data(self, file_path: Optional[str] = None) -> bool:
        """
        Load population data.
        
        Args:
            file_path: Path to population CSV
            
        Returns:
            True if loaded successfully
        """
        if file_path is None:
            file_path = self.data_dir / 'population.csv'
        
        try:
            self.population_data = pd.read_csv(file_path)
            logger.info(f"Loaded population data for {len(self.population_data)} regions")
            return True
        except Exception as e:
            logger.error(f"Failed to load population data: {e}")
            
            # Create synthetic population data
            self._create_synthetic_population()
            return False
    
    def _create_synthetic_population(self):
        """Create synthetic population data for Gaza."""
        logger.info("Creating synthetic population data...")
        
        regions = [
            'North_Gaza', 'Gaza_City', 'Deir_al_Balah',
            'Khan_Younis', 'Rafah'
        ]
        
        populations = [350000, 600000, 250000, 400000, 220000]
        areas = [61, 70, 58, 108, 64]  # km²
        
        data = []
        for region, pop, area in zip(regions, populations, areas):
            data.append({
                'region': region,
                'population': pop,
                'area_km2': area,
                'density': pop / area,
                'households': pop / 5.5  # Average household size
            })
        
        self.population_data = pd.DataFrame(data)
        logger.info(f"Created synthetic population data for {len(data)} regions")
    
    def load_infrastructure(self, file_path: Optional[str] = None) -> bool:
        """
        Load infrastructure data.
        
        Args:
            file_path: Path to infrastructure GeoJSON
            
        Returns:
            True if loaded successfully
        """
        if file_path is None:
            file_path = self.data_dir / 'infrastructure.geojson'
        
        try:
            self.infrastructure = gpd.read_file(file_path)
            logger.info(f"Loaded {len(self.infrastructure)} infrastructure features")
            return True
        except Exception as e:
            logger.error(f"Failed to load infrastructure: {e}")
            return False
    
    def enrich_with_risk_zones(
        self,
        df: pd.DataFrame,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ) -> pd.DataFrame:
        """
        Add risk zone information to locations.
        
        Args:
            df: DataFrame with locations
            lat_col: Latitude column name
            lon_col: Longitude column name
            
        Returns:
            Enriched DataFrame
        """
        if self.risk_zones is None:
            self.load_risk_zones()
        
        result = df.copy()
        
        # Create point geometries
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(result, geometry=geometry, crs='EPSG:4326')
        
        # Spatial join with risk zones
        joined = gpd.sjoin(gdf, self.risk_zones, how='left', predicate='within')
        
        # Add risk score
        if 'risk_level' in joined.columns:
            result['Risk_Score'] = joined['risk_level'].fillna(5).astype(int)
        else:
            result['Risk_Score'] = 5
        
        logger.info(f"Enriched {len(result)} locations with risk zones")
        return result
    
    def enrich_with_population(
        self,
        df: pd.DataFrame,
        region_col: str = 'Region_ID'
    ) -> pd.DataFrame:
        """
        Add population data to locations.
        
        Args:
            df: DataFrame with locations
            region_col: Column with region names
            
        Returns:
            Enriched DataFrame
        """
        if self.population_data is None:
            self.load_population_data()
        
        result = df.copy()
        
        # Extract region from Region_ID
        if region_col in df.columns:
            result['region'] = df[region_col].str.extract(r'([A-Za-z_]+)')
            
            # Merge with population data
            result = result.merge(
                self.population_data,
                on='region',
                how='left'
            )
            
            logger.info(f"Enriched with population data")
        
        return result
    
    def calculate_grid_distances(
        self,
        df: pd.DataFrame,
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude'
    ) -> pd.DataFrame:
        """
        Calculate distance to nearest grid infrastructure.
        
        Args:
            df: DataFrame with locations
            lat_col: Latitude column
            lon_col: Longitude column
            
        Returns:
            DataFrame with Grid_Distance column
        """
        if self.infrastructure is None:
            self.load_infrastructure()
        
        result = df.copy()
        
        if self.infrastructure is not None and len(self.infrastructure) > 0:
            # Calculate distances to nearest power line
            from scipy.spatial import KDTree
            
            # Extract line points (simplified)
            coords = []
            for geom in self.infrastructure.geometry:
                if geom.geom_type == 'LineString':
                    coords.extend(list(geom.coords))
            
            if coords:
                tree = KDTree([(c[1], c[0]) for c in coords])  # lat, lon
                
                distances = []
                for idx, row in df.iterrows():
                    dist, _ = tree.query([row[lat_col], row[lon_col]])
                    distances.append(dist * 111)  # Convert degrees to km approx
                
                result['Grid_Distance'] = distances
            else:
                result['Grid_Distance'] = np.random.randint(100, 5000, len(df))
        else:
            # Synthetic distances
            result['Grid_Distance'] = np.random.randint(100, 5000, len(df))
        
        logger.info(f"Calculated grid distances")
        return result

def enrich_with_risk_zones(
    df: pd.DataFrame,
    data_dir: str = '../../data'
) -> pd.DataFrame:
    """
    Quick function to add risk zones.
    
    Args:
        df: Input DataFrame
        data_dir: Data directory
        
    Returns:
        Enriched DataFrame
    """
    enricher = LocationEnricher(data_dir)
    return enricher.enrich_with_risk_zones(df)

def add_population_data(
    df: pd.DataFrame,
    data_dir: str = '../../data'
) -> pd.DataFrame:
    """
    Quick function to add population data.
    
    Args:
        df: Input DataFrame
        data_dir: Data directory
        
    Returns:
        Enriched DataFrame
    """
    enricher = LocationEnricher(data_dir)
    return enricher.enrich_with_population(df)

if __name__ == "__main__":
    # Create sample data
    df = pd.DataFrame({
        'Region_ID': [f'Site_{i}' for i in range(20)],
        'Latitude': np.random.uniform(31.25, 31.58, 20),
        'Longitude': np.random.uniform(34.20, 34.55, 20)
    })
    
    # Enrich
    enricher = LocationEnricher()
    df = enricher.enrich_with_risk_zones(df)
    df = enricher.calculate_grid_distances(df)
    
    print(df[['Region_ID', 'Risk_Score', 'Grid_Distance']].head())
