'''
Script to visualize spatial variability and temporal dynamics of HBV model parameters
produced by the LSTM model. These parameters are dynamic during training but we run the LSTM-HBV
model with full sequence length per basin and use last step parameters during inference making them
static per basin. This script basically analyzes the temporal variability of HBV parameters during training
which are often used as 'static' parameters during inference.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import os

# --- Settings
param_names = [ "fc", "beta", "pwp", "l", "ks", "ki", "kb", "kperc",
                "coeff_pet", "ddf", "scf", "ts", "tm", "tti", "whc",
                "crf", "d_shape", "d_scale", "b_shape", "b_scale"]

param_bounds = {
    "fc": [1.0, 1000.0],
    "beta": [0.5, 5.0],
    "pwp": [0.01, 0.99],
    "l": [1.0, 999.0],
    "ks": [0.01, 0.99],
    "ki": [0.01, 0.99],
    "kb": [0.001, 0.99],
    "kperc": [0.0001, 0.99],
    "coeff_pet": [0.5, 2.0],
    "ddf": [0.05, 10.0],
    "scf": [0.5, 2.0],
    "ts": [-4.0, 4.0],
    "tm": [-4.0, 4.0],
    "tti": [0.1, 4.0],
    "whc": [0.05, 0.2],
    "crf": [0.1, 1.0],
    "d_shape": [1.0, 4.0],
    "d_scale": [0.5, 4.0],
    "b_shape": [1.0, 6.0],
    "b_scale": [1.0, 7.0],
}

basin_list = pd.read_csv("camels531.csv")
basin_list['name'] = basin_list['name'].apply(lambda x: str(x).zfill(8))

shapefile_path = 'Z:/LSTM-Attention4PUB/data/us_shapefile/ne_110m_admin_1_states_provinces_lakes.shp'
us = gpd.read_file(shapefile_path)
us = us[us['admin'] == 'United States of America']

output_dir = "figures/hbv_parameters"
os.makedirs(output_dir, exist_ok=True)

# --- Function to plot map + timeseries for a single parameter
def plot_param(param):
    # Compute coefficient of variation per basin
    param_var = {}
    for basin in basin_list['name']:
        param_file = f"output/best_lstm_1hbv_parameters/input_{basin}_parameters.csv"
        params = pd.read_csv(param_file)
        mean_val = np.mean(params[param].values)
        std_val = np.std(params[param].values)
        param_var[basin] = std_val / mean_val if mean_val != 0 else 0
        # also print warning if mean is zero
        if mean_val == 0:
            print(f"Mean value for basin {basin} is zero, setting coefficient of variation to 0.")

    param_var_df = pd.DataFrame.from_dict(param_var, orient='index', columns=[param]).reset_index().rename(columns={'index': 'basin'})
    param_var_df = param_var_df.merge(basin_list[['name', 'lat', 'lon']], left_on='basin', right_on='name', how='left')

    param_var_gdf = gpd.GeoDataFrame(
        param_var_df,
        geometry=gpd.points_from_xy(param_var_df.lon, param_var_df.lat),
        crs="EPSG:4326"
    )

    # Identify basin with highest CV
    highest_var_basin = param_var_df.loc[param_var_df[param].idxmax(), 'basin']

    # --- Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios':[4,1]})

    # Top panel: map
    us.boundary.plot(ax=ax1, color='black', linewidth=0.8)
    g = param_var_gdf.plot(column=param, ax=ax1, cmap='plasma', markersize=15, legend=False)

    # Horizontal colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', 
                               norm=plt.Normalize(vmin=param_var_gdf[param].min(),
                                                  vmax=param_var_gdf[param].max()))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', fraction=0.03, pad=0.10)

    # Highlight highest CV basin
    highest_var_point = param_var_gdf[param_var_gdf['basin'] == highest_var_basin]
    highest_var_point.plot(ax=ax1, color='green', marker='+', markersize=50, label='Highest value')
    ax1.legend(frameon=True, loc='lower left')

    ax1.set_title(f'Coefficient of Variation (std/mean) of {param} Across Basins', fontsize=12, pad=12)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_xlim([-130, -65])
    ax1.set_ylim([24, 50])
    ax1.set_aspect('equal')

    # Bottom panel: time series
    param_file = f"output/best_lstm_1hbv_parameters/input_{highest_var_basin}_parameters.csv"
    params = pd.read_csv(param_file)
    params['date'] = pd.to_datetime(params['date'])

    sns.lineplot(data=params, x='date', y=param, ax=ax2, color='tab:blue')
    sns.scatterplot(data=params, x='date', y=param, ax=ax2, color='tab:blue', s=10)
    low, high = param_bounds[param]
    ax2.axhline(low, color='red', linestyle='--', linewidth=1, label='Lower bound')
    ax2.axhline(high, color='orange', linestyle='--', linewidth=1, label='Upper bound')
    ax2.legend(frameon=False, loc='upper right', fontsize=8, ncol=2, columnspacing=0.5, handletextpad=0.3)
    ax2.set_title(f'Time Series of {param} for Basin {highest_var_basin}', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel(param)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/plot_{param}.jpeg", dpi=300, bbox_inches="tight")
    # plt.close(fig)  # close to save memory when looping

# --- Loop over all parameters
for param in param_names:
    plot_param(param)
