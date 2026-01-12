"""
Thin input features to a minimal, commonly available set and add simple basin-level stats.
Keeps: date, precip, tmax, tmin, daylenhr, qobs.
Adds per-basin summary stats for dynamic features: mean_precip, sd_precip, mean_tmax, sd_tmax, ...
Also appends static basin attributes (drainage area, elevation, slope, lat, lon).
Writes thinned CSVs to the 'data_thinned' directory.
"""
import pandas as pd
import os

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