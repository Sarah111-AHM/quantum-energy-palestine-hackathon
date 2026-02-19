"""
Merge multiple datasets into master Gaza energy dataset.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetMerger:
    """
    Merge multiple data sources into master dataset.
    """
    
    def __init__(self, data_dir: str = '../../data'):
        """
        Initialize merger.
        
        Args:
            data_dir: Directory with data files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary of dataset name to DataFrame
        """
        # Define expected datasets
        expected = {
            'energy': 'gaza_energy_data.csv',
            'nasa': 'nasa_weather_data.csv',
            'osm_power': 'osm_power.geojson',
            'osm_roads': 'osm_highway.geojson',
            'population': 'population.csv',
            'risk_zones': 'risk_zones.geojson'
        }
        
        for name, filename in expected.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    if filename.endswith('.geojson'):
                        import geopandas as gpd
                        df = gpd.read_file(file_path)
                        # Convert to regular DataFrame
                        if 'geometry' in df.columns:
                            df['Latitude'] = df.geometry.y
                            df['Longitude'] = df.geometry.x
                            df = df.drop(columns='geometry')
                    else:
                        df = pd.read_csv(file_path)
                    
                    self.datasets[name] = df
                    logger.info(f"Loaded {name}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")
            else:
                logger.warning(f"Dataset not found: {file_path}")
        
        return self.datasets
    
    def create_base_locations(self, n_sites: int = 45) -> pd.DataFrame:
        """
        Create base location grid for Gaza.
        
        Args:
            n_sites: Number of sites to generate
            
        Returns:
            DataFrame with base locations
        """
        # Gaza bounds
        min_lat, max_lat = 31.25, 31.58
        min_lon, max_lon = 34.20, 34.55
        
        # Generate grid points
        n_per_side = int(np.sqrt(n_sites))
        lats = np.linspace(min_lat, max_lat, n_per_side)
        lons = np.linspace(min_lon, max_lon, n_per_side)
        
        points = []
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                region = self._get_region(lat)
                points.append({
                    'Region_ID': f"{region}_{i:02d}{j:02d}",
                    'Latitude': round(lat, 6),
                    'Longitude': round(lon, 6)
                })
        
        df = pd.DataFrame(points[:n_sites])
        logger.info(f"Created {len(df)} base locations")
        return df
    
    def _get_region(self, latitude: float) -> str:
        """Determine region based on latitude."""
        if latitude > 31.50:
            return "North_Gaza"
        elif latitude > 31.45:
            return "Gaza_City"
        elif latitude > 31.38:
            return "Deir_al_Balah"
        elif latitude > 31.30:
            return "Khan_Younis"
        else:
            return "Rafah"
    
    def merge_with_nasa(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge with NASA weather data.
        
        Args:
            base_df: Base locations DataFrame
            
        Returns:
            Merged DataFrame
        """
        if 'nasa' not in self.datasets:
            logger.warning("NASA data not available, generating synthetic data")
            return self._add_synthetic_nasa(base_df)
        
        nasa_df = self.datasets['nasa']
        result = base_df.copy()
        
        # For each location, find nearest NASA grid point
        from scipy.spatial import KDTree
        
        nasa_coords = nasa_df[['latitude', 'longitude']].values
        tree = KDTree(nasa_coords)
        
        solar_values = []
        wind_values = []
        
        for idx, row in base_df.iterrows():
            dist, nasa_idx = tree.query([row['Latitude'], row['Longitude']])
            solar_values.append(nasa_df.iloc[nasa_idx]['solar_irradiance'])
            wind_values.append(nasa_df.iloc[nasa_idx]['wind_speed'])
        
        result['Solar_Irradiance'] = solar_values
        result['Wind_Speed'] = wind_values
        
        logger.info("Merged with NASA data")
        return result
    
    def _add_synthetic_nasa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add synthetic NASA data."""
        result = df.copy()
        
        # Solar irradiance (4.5-6.0 kWh/m²/day)
        result['Solar_Irradiance'] = np.random.uniform(4.5, 6.0, len(df))
        
        # Wind speed (2.5-6.5 m/s) - higher near coast
        is_coastal = df['Longitude'] < 34.35
        result.loc[is_coastal, 'Wind_Speed'] = np.random.uniform(4.0, 6.5, is_coastal.sum())
        result.loc[~is_coastal, 'Wind_Speed'] = np.random.uniform(2.5, 4.5, (~is_coastal).sum())
        
        logger.info("Added synthetic NASA data")
        return result
    
    def add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk scores based on location.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk scores
        """
        result = df.copy()
        
        if 'risk_zones' in self.datasets:
            # Use actual risk zones
            from .03_enrich_locations import enrich_with_risk_zones
            result = enrich_with_risk_zones(result, str(self.data_dir))
        else:
            # Synthetic risk scores
            # Higher risk in north and south borders
            risk = []
            for idx, row in df.iterrows():
                if row['Latitude'] > 31.50 or row['Latitude'] < 31.30:
                    risk.append(np.random.randint(6, 11))
                else:
                    risk.append(np.random.randint(2, 8))
            
            result['Risk_Score'] = risk
        
        # Add accessibility (80% accessible)
        result['Accessibility'] = np.random.choice([0, 1], len(df), p=[0.2, 0.8])
        
        logger.info("Added risk scores and accessibility")
        return result
    
    def add_grid_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add distances to grid infrastructure.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with grid distances
        """
        result = df.copy()
        
        if 'osm_power' in self.datasets:
            from .03_enrich_locations import LocationEnricher
            enricher = LocationEnricher(str(self.data_dir))
            enricher.infrastructure = self.datasets['osm_power']
            result = enricher.calculate_grid_distances(result)
        else:
            # Synthetic distances (100-5000m)
            result['Grid_Distance'] = np.random.randint(100, 5000, len(df))
        
        logger.info("Added grid distances")
        return result
    
    def create_master_dataset(
        self,
        n_sites: int = 45,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Create master dataset from all sources.
        
        Args:
            n_sites: Number of sites to generate
            save: Whether to save to CSV
            
        Returns:
            Master DataFrame
        """
        logger.info("Creating master dataset...")
        
        # Load all available datasets
        self.load_all_datasets()
        
        # Create base locations
        df = self.create_base_locations(n_sites)
        
        # Merge with NASA data
        df = self.merge_with_nasa(df)
        
        # Add risk scores
        df = self.add_risk_scores(df)
        
        # Add grid distances
        df = self.add_grid_distances(df)
        
        # Round values
        df['Solar_Irradiance'] = df['Solar_Irradiance'].round(2)
        df['Wind_Speed'] = df['Wind_Speed'].round(2)
        
        logger.info(f"Master dataset created with {len(df)} rows and {len(df.columns)} columns")
        
        # Save
        if save:
            output_path = self.data_dir / 'gaza_energy_data.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Saved to {output_path}")
        
        return df

def merge_all_datasets(data_dir: str = '../../data') -> Dict[str, pd.DataFrame]:
    """
    Quick function to load all datasets.
    
    Args:
        data_dir: Data directory
        
    Returns:
        Dictionary of datasets
    """
    merger = DatasetMerger(data_dir)
    return merger.load_all_datasets()

def create_master_dataset(
    n_sites: int = 45,
    data_dir: str = '../../data',
    save: bool = True
) -> pd.DataFrame:
    """
    Quick function to create master dataset.
    
    Args:
        n_sites: Number of sites
        data_dir: Data directory
        save: Whether to save
        
    Returns:
        Master DataFrame
    """
    merger = DatasetMerger(data_dir)
    return merger.create_master_dataset(n_sites, save)

if __name__ == "__main__":
    # Create master dataset
    df = create_master_dataset(n_sites=45, save=True)
    
    print(f"Master dataset created with {len(df)} rows")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Save summary
    summary = {
        'total_sites': len(df),
        'accessible_sites': df['Accessibility'].sum(),
        'avg_solar': df['Solar_Irradiance'].mean(),
        'avg_wind': df['Wind_Speed'].mean(),
        'avg_risk': df['Risk_Score'].mean(),
        'avg_grid_distance': df['Grid_Distance'].mean()
    }
    
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
