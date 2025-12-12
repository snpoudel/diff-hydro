import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def nse(observed, simulated):
    """Calculate Nash-Sutcliffe Efficiency."""
    observed_mean = observed.mean()
    numerator = ((observed - simulated) ** 2).sum()
    denominator = ((observed - observed_mean) ** 2).sum()
    return 1 - (numerator / denominator)

# input_file = pd.read_csv(f'output/best_lstm_1hbv/pred_input_12035000.csv')
# print('NSE: ', nse(input_file['qobs'], input_file['qsim_1']))

#read equifinality test results
basin_id = "12035000"
use_par = 'fc' # parameter to test equifinality on

# read parameter sets
param_sets = pd.read_csv(f"output/best_lstm_1hbv_parameters/input_{basin_id}_parameters.csv")
highest_coeff_pet = np.round(param_sets[use_par].max(), 2)
lowest_coeff_pet = np.round(param_sets[use_par].min(), 2)

# read results from equifinality test
results_df = pd.read_csv(f'output/best_lstm_1hbv_equifinality/equifinality_test_{basin_id}_{use_par}.csv')
# train valid test periods: 1990-2005 (train), 2006-2015 (valid), 1980-1989 (test)
results_df['date'] = pd.to_datetime(results_df['date'])
train_mask = (results_df['date'] >= '1990-01-01') & (results_df['date'] <= '2005-12-31')
valid_mask = (results_df['date'] >= '2006-01-01') & (results_df['date'] <= '2015-12-31')
test_mask = (results_df['date'] >= '1980-01-01') & (results_df['date'] <= '1989-12-31')

results_df_test = results_df[test_mask]
#calculate NSE for both parameter sets
nse_high = nse(results_df_test['qobs'], results_df_test[f'qsim_high_{use_par}'])
nse_low = nse(results_df_test['qobs'], results_df_test[f'qsim_low_{use_par}'])
print(f"NSE for high {use_par} parameter set: {nse_high:.4f}")
print(f"NSE for low {use_par} parameter set: {nse_low:.4f}")

# plot observed vs simulated for both parameter sets
results_df_plot = results_df[(results_df['date'] >= '1980-01-01') & (results_df['date'] <= '1980-12-31')]
plt.figure(figsize=(8, 5))
plt.plot(results_df_plot['date'], results_df_plot['qobs'], label='Obs', color='black', linewidth=1)
plt.scatter(results_df_plot['date'], results_df_plot['qobs'],color='black', s=10)
plt.plot(results_df_plot['date'], results_df_plot[f'qsim_high_{use_par}'], label=f'Sim high {use_par}: {highest_coeff_pet}\n NSE: {nse_high:.2f}')
plt.plot(results_df_plot['date'], results_df_plot[f'qsim_low_{use_par}'], label=f'Sim low {use_par}: {lowest_coeff_pet}\n NSE: {nse_low:.2f}')
plt.xlabel('Date')
plt.ylabel('Streamflow')
plt.title(f'Basin {basin_id}: parameter sets with the highest and lowest values of {use_par}', fontsize=11)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'figures/equifinality/equifinality_plot_{basin_id}_{use_par}.png', dpi=300)
plt.show()