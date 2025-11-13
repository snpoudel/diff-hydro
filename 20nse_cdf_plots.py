"""
Create CDF plots of NSE for out-of-sample predictions in time and space.
Model trained on CAMELS-US basins (1990-2005) and validated on 2006-2015.
20% of basins were withheld for spatial testing; temporal testing uses 1980-1989.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def nse(observed, simulated):
    """Calculate Nash-Sutcliffe Efficiency (NSE)"""
    observed_mean = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2)
    return 1 - (numerator / denominator)


# NSE cdf plot for ungauged basins predictions
model_name = 'best_lstm_16hbv'

## Out of sample in Time
all_basin_list = pd.read_csv(f'camels531.csv')

ungauged_basins = pd.read_csv(f'output/{model_name}/test_basins.csv')
gauged_basins = all_basin_list[~all_basin_list['name'].isin(ungauged_basins['name'])]

gauged_basins = gauged_basins["name"].values
# add a leading zero to gauge_id if length is 7
gauge_id = [str(gid).zfill(8) for gid in gauged_basins]
median_nse_values = []
for file in gauge_id:
    try:
        df = pd.read_csv(f'output/{model_name}/pred_input_{file}.csv')
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= pd.to_datetime('1980-01-01')) & (df['date'] <= pd.to_datetime('1989-12-31'))]
        df = df.dropna(subset=['qobs', 'qsim_1'])
        median_nse = nse(df['qobs'].values, df['qsim_1'].values)
        median_nse_values.append(median_nse)
    except:
        continue
median_nse_values = np.array(median_nse_values)
median_nse_values = median_nse_values[~np.isnan(median_nse_values)]


##-- Out of sample in Space
pub_gauge_id = ungauged_basins["name"].values
# add a leading zero to gauge_id if length is 7
pub_gauge_id = [str(gid).zfill(8) for gid in pub_gauge_id]

pub_median_nse_values = []
for file in pub_gauge_id:
    try:
        df = pd.read_csv(f'output/{model_name}/pred_input_{file}.csv')
        df['date'] = pd.to_datetime(df['date'])
        # remove any rows with NaN values in qobs or qsim_1
        df = df.dropna(subset=['qobs', 'qsim_1'])
        median_nse = nse(df['qobs'].values, df['qsim_1'].values)
        pub_median_nse_values.append(median_nse)
    except:
        continue
pub_median_nse_values = np.array(pub_median_nse_values)
pub_median_nse_values = pub_median_nse_values[~np.isnan(pub_median_nse_values)]


# Plot CDF of median NSE values (curve and points)
sorted_nse = np.sort(median_nse_values)
sorted_pub_nse = np.sort(pub_median_nse_values)

cdf = np.arange(1, len(sorted_nse) + 1) / len(sorted_nse)
cdf_pub = np.arange(1, len(sorted_pub_nse) + 1) / len(sorted_pub_nse)

# Figure
plt.figure(figsize=(6, 4))
plt.plot(sorted_nse, cdf, color='blue', lw=1.5, label=f'DiffModel out-of-sample in time\n Count={len(median_nse_values)} | Median NSE={np.median(median_nse_values):.2f}')
plt.scatter(sorted_nse, cdf, color='blue', s=5)
plt.plot(sorted_pub_nse, cdf_pub, color='orange', lw=1.5, label=f'DiffModel out-of-sample in space\n Count={len(pub_median_nse_values)} | Median NSE={np.median(pub_median_nse_values):.2f}')
plt.scatter(sorted_pub_nse, cdf_pub, color='orange', s=5)
plt.axhline(y=0.5, color='brown', linestyle='--', lw=1.5)
plt.xlabel('Nash-Sutcliffe Efficiency (NSE)')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.title(f"Summary Performance for: {model_name.replace('best_', '').upper().replace('_', ' + ')} Model")
plt.legend(fontsize=10)
plt.xlim([-1, 1])
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'figures/{model_name}_NSE.png', dpi=300)
plt.show()