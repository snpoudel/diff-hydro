"""
CAMELS-US Dataset Preprocessing Script

This script processes the CAMELS-US dataset and compiles static and dynamic features
for each basin into unified CSV files.

Author: Sandeep Poudel
Date: February 04, 2026

Setup Instructions:
1. Download CAMELS-US dataset from: https://zenodo.org/records/15529996
2. Extract all zip files into a folder with this structure:
   camels/
   ├── static/
   │   ├── camels_topo.txt
   │   ├── camels_clim.txt
   │   ├── camels_vege.txt
   │   ├── camels_soil.txt
   │   └── camels_geol.txt
   └── dynamic/
       ├── daymet/
       │   ├── 01/, 02/, ..., 18/
       └── usgs_streamflow/
           ├── 01/, 02/, ..., 18/
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional


# ==================== CONFIGURATION ====================

# Update this path to your CAMELS folder
CAMELS_DIR = Path('C:/Sandeep/camels/')

# Feature configurations
STATIC_FEATURES = {
    'topo': ['gauge_id', 'elev_mean', 'slope_mean', 'area_gages2'],
    'clim': ['gauge_id', 'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 
             'frac_snow', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur'],
    'vege': ['gauge_id', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 
             'gvf_diff', 'dom_land_cover_frac', 'root_depth_50'],
    'soil': ['gauge_id', 'soil_depth_pelletier', 'soil_depth_statsgo', 
             'soil_porosity', 'soil_conductivity', 'max_water_content', 
             'sand_frac', 'silt_frac', 'clay_frac'],
    'geol': ['gauge_id', 'glim_1st_class_frac', 'glim_2nd_class_frac', 
             'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
}

DYNAMIC_INPUT_COLS = ['Year', 'Mnth', 'Day', 'dayl(s)', 'srad(W/m2)', 
                      'vp(Pa)', 'tmax(C)', 'tmin(C)', 'prcp(mm/day)']

OUTPUT_COLS = ['Year', 'Mnth', 'Day', 'qobs(mm/day)']

# Constants
CFS_TO_MM_PER_DAY = 2.446575  # Conversion factor from cfs to mm/day per km²


# ==================== HELPER FUNCTIONS ====================

def consolidate_streamflow_files(camels_dir: Path) -> None:
    """Consolidate all USGS streamflow files into a single directory."""
    print("Consolidating streamflow files...")
    
    source_dir = camels_dir / 'dynamic' / 'usgs_streamflow'
    target_dir = source_dir / 'all'
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each numbered subdirectory (01-18)
    for i in range(1, 19):
        subdir = source_dir / f'{i:02d}'
        if not subdir.exists():
            continue
            
        for file in subdir.glob('*.txt'):
            df = pd.read_csv(file)
            df.to_csv(target_dir / file.name, index=False)
    
    print(f"  ✓ Streamflow files consolidated to {target_dir}")


def load_basin_list(camels_dir: Path) -> pd.DataFrame:
    """Load basin list or use all basins from camels_topo.txt."""
    basin_file = camels_dir / 'camels_531.txt'
    
    if basin_file.exists():
        print("Loading specified basin list from camels_531.txt...")
        basins = pd.read_csv(basin_file, dtype={'gauge_id': str})
        basins = basins[['gauge_id']]
    else:
        print("Basin list not found, using all basins from camels_topo.txt...")
        topo = pd.read_csv(camels_dir / 'static' / 'camels_topo.txt', 
                          sep=';', dtype={'gauge_id': str})
        basins = topo[['gauge_id']]
    
    # Ensure gauge_id is 8 digits
    basins['gauge_id'] = basins['gauge_id'].str.zfill(8)
    print(f"  ✓ Loaded {len(basins)} basins")
    
    return basins


def load_static_features(camels_dir: Path) -> dict:
    """Load all static feature files."""
    print("Loading static features...")
    
    static_dir = camels_dir / 'static'
    static_dfs = {}
    
    for feature_type, columns in STATIC_FEATURES.items():
        filepath = static_dir / f'camels_{feature_type}.txt'
        df = pd.read_csv(filepath, sep=';', dtype={'gauge_id': str})
        static_dfs[feature_type] = df[columns]
    
    print(f"  ✓ Loaded {len(static_dfs)} static feature types")
    return static_dfs


def convert_discharge_units(qobs: pd.Series, area_km2: float) -> pd.Series:
    """Convert discharge from cfs to mm/day."""
    qobs = qobs.replace(-999.0, np.nan)  # Replace missing value flag
    return qobs * CFS_TO_MM_PER_DAY / area_km2


def process_basin(gauge_id: str, camels_dir: Path, static_dfs: dict) -> pd.DataFrame:
    """Process a single basin and return merged DataFrame."""
    
    # Load dynamic input (forcing data)
    input_file = camels_dir / 'dynamic' / 'daymet' / 'all' / f'{gauge_id}_lump_cida_forcing_leap.txt'
    input_df = pd.read_csv(input_file, sep='\\s+', skiprows=3, usecols=DYNAMIC_INPUT_COLS)
    
    # Load dynamic output (streamflow)
    output_file = camels_dir / 'dynamic' / 'usgs_streamflow' / 'all' / f'{gauge_id}_streamflow_qc.txt'
    output_df = pd.read_csv(
        output_file, 
        sep='\\s+', 
        header=None,
        names=['gauge_id', 'Year', 'Mnth', 'Day', 'qobs', 'flag']
    )
    
    # Get basin area for unit conversion
    area_km2 = static_dfs['topo'][static_dfs['topo']['gauge_id'] == gauge_id]['area_gages2'].values[0]
    
    # Convert discharge units
    output_df['qobs(mm/day)'] = convert_discharge_units(output_df['qobs'], area_km2)
    output_df = output_df[OUTPUT_COLS]
    
    # Merge dynamic features
    merged_df = pd.merge(input_df, output_df, on=['Year', 'Mnth', 'Day'], how='left')
    
    # Add static features by repeating for each row
    for feature_type, df in static_dfs.items():
        filtered = df[df['gauge_id'] == gauge_id].drop(columns=['gauge_id'])
        
        # Repeat static values for all timesteps
        for col in filtered.columns:
            merged_df[col] = filtered[col].values[0]
    
    return merged_df.round(4)


def process_all_basins(camels_dir: Path, basin_list: pd.DataFrame, 
                       static_dfs: dict, output_dir: Path) -> None:
    """Process all basins and save compiled files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gauge_ids = basin_list['gauge_id'].tolist()
    total = len(gauge_ids)
    
    print(f"\nProcessing {total} basins...")
    
    for idx, gauge_id in enumerate(gauge_ids, 1):
        try:
            merged_df = process_basin(gauge_id, camels_dir, static_dfs)
            output_file = output_dir / f'input_{gauge_id}.csv'
            merged_df.to_csv(output_file, index=False)
            
            if idx % 50 == 0 or idx == total:
                print(f"  Progress: {idx}/{total} basins processed")
                
        except Exception as e:
            print(f"  ✗ Error processing basin {gauge_id}: {e}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("CAMELS-US Dataset Preprocessing")
    print("=" * 60)
    
    # Consolidate streamflow files
    consolidate_streamflow_files(CAMELS_DIR)
    
    # Load basin list
    basin_list = load_basin_list(CAMELS_DIR)
    
    # Load static features
    static_dfs = load_static_features(CAMELS_DIR)
    
    # Process all basins
    output_dir = CAMELS_DIR / 'compiled'
    process_all_basins(CAMELS_DIR, basin_list, static_dfs, output_dir)
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing completed successfully!")
    print(f"✓ Compiled files saved to: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()


