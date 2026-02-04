"""
First, CAMELS-US dataset is downloaded from: https://zenodo.org/records/15529996 and compiled in desired format using the script '01preprocess_camels_data.py'.

Now this scripts modifies the originally created input files in two ways:
1) Only use basic basin attributes and forcings summary statistics.
2) Create a mixed-up dataset where all basin statics are kept but shuffled between basins.

Author: Sandeep Poudel (Feb 04, 2026)
"""
import pandas as pd
import os
from pathlib import Path

# set working directory to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(SCRIPT_DIR))

##---------------------------------------------------####---------------------------------------------------####---------------------------------------------------##
## Input modification 01: Create a thinned dataset with only few features and forcings summary.
# Keeps: date, precip, tmax, tmin, daylenhr, qobs.
# Adds per-basin summary stats for dynamic features: mean_precip, sd_precip, mean_tmax, sd_tmax, ...
# Also appends static basin attributes (drainage area, elevation, slope, lat, lon).
# Writes thinned CSVs to the 'data_thinned' directory.

basin_list = pd.read_csv(f'camels531.csv', dtype={'name': str})

thinned_features = ['date', 'precip', 'tmax', 'tmin', 'daylenhr', 'qobs']
output_file_path = 'data_thinned'

for id in basin_list['name']:
    file_path = f'data/input_{id.zfill(8)}.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=['date'])
        df = df[thinned_features]
        # add mean_precip, sd_precip, mean_tmax, sd_tmax, mean_tmin, sd_tmin, mean_daylenhr, sd_daylenhr
        df['mean_precip'] = df['precip'].mean()
        df['sd_precip'] = df['precip'].std()
        df['mean_tmax'] = df['tmax'].mean()
        df['sd_tmax'] = df['tmax'].std()
        df['mean_tmin'] = df['tmin'].mean()
        df['sd_tmin'] = df['tmin'].std()
        df['mean_daylenhr'] = df['daylenhr'].mean()
        df['sd_daylenhr'] = df['daylenhr'].std()
        # further add statics from basin_list
        basin_statics = basin_list[basin_list['name'] == id].iloc[0]
        for feature in basin_statics.index:
            if feature != 'name':
                df[feature] = basin_statics[feature]
        # round to 4 decimal places
        df = df.round(4)
        df.to_csv(f'{output_file_path}/input_{id.zfill(8)}.csv', index=False)
    else:
        print(f"File {file_path} does not exist.")


##---------------------------------------------------####---------------------------------------------------####---------------------------------------------------##
## Input modification 02: Create a mixed-up dataset where statics are shuffled between basins.

# Make another dataset which is the comprehensive one but except for the dynamic features
# all other statics are shuffled between basins, so a basin has statics from other basins
dynamic_features = ['date','precip','tmax','tmin','daylenhr','srad(W/m2)','vp(Pa)','qobs']

# Setup paths
input_folder = Path('data')
output_folder = Path('data_mixedup')
output_folder.mkdir(exist_ok=True)

# Process each CSV file
all_csv_files = list(input_folder.glob('*.csv'))
randomized_list = all_csv_files.copy()
import random
# set seed for reproducibility
random.seed(42)
random.shuffle(randomized_list)

for csv_file in randomized_list:
    # Read CSV
    df_original = pd.read_csv(csv_file)
    # get statics from next files since they are randomized
    next_index = (randomized_list.index(csv_file) + 1) % len(randomized_list)
    df_next = pd.read_csv(randomized_list[next_index])
    # replace statics in df_original with those from df_next
    for column in df_original.columns:
        if column not in dynamic_features:
            df_original[column] = df_next[column].values
    # round df to 4 decimal places
    df_original = df_original.round(4)
    # Save to output folder
    output_path = output_folder / csv_file.name
    df_original.to_csv(output_path, index=False)
    print(f"Processed: {csv_file.name}")
    
print(f"\nAll files processed! Saved to {output_folder}")
    