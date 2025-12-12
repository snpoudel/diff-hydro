import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def nse(observed, simulated):
    """Calculate Nash-Sutcliffe Efficiency."""
    observed_mean = observed.mean()
    numerator = ((observed - simulated) ** 2).sum()
    denominator = ((observed - observed_mean) ** 2).sum()
    return 1 - (numerator / denominator)

##-- Correlation plot------------------------------------------------------------------------------------------------------------------------------------
#read equifinality test results
basin_id = "10336660" #12035000

# read parameter sets
param_sets = pd.read_csv(f"output/best_lstm_1hbv_parameters/input_{basin_id}_parameters.csv")
param_sets = param_sets.drop(columns=['date'])

# use only selected parameters
# use_pars = ['fc', 'beta', 'pwp', 'coeff_pet', 'scf']
use_pars = ['fc', 'pwp', 'kperc', 'coeff_pet', 'ddf', 'scf']
param_sets = param_sets[use_pars]

#plor pairplot of parameters
g = sns.pairplot(param_sets, height=1.5, aspect=1)
g.fig.suptitle(f'Pairplot of Parameters for Basin {basin_id}', y=0.98)
g.fig.tight_layout()
g.fig.subplots_adjust(top=0.92)
# plt.savefig(f'figures/pars_correlation/parameter_pairplot_{basin_id}.png', dpi=300)
plt.show()


##-- Parameter variability vs model nse performance------------------------------------------------------------------------------------------------------------------------------------
param_varibility = pd.read_csv(f'figures/hbv_parameters/avg_cv_all_parameters_per_basin.csv')
param_varibility['nse'] = None
basin_ids = param_varibility['basin'].unique()
#zfill basin ids to 8 digits
basin_ids = [str(basin_id).zfill(8) for basin_id in basin_ids]
#read each basin calculate nse and add to dataframe
for basin_id in basin_ids:
    # read results from equifinality test
    results_df = pd.read_csv(f'output/best_lstm_1hbv/pred_input_{basin_id}.csv')
    # train valid test periods: 1990-2005 (train), 2006-2015 (valid), 1980-1989 (test)
    results_df['date'] = pd.to_datetime(results_df['date'])
    test_mask = (results_df['date'] >= '1980-01-01') & (results_df['date'] <= '1989-12-31')
    results_df_test = results_df[test_mask]
    #calculate NSE for both parameter sets
    nse_val = nse(results_df_test['qobs'], results_df_test['qsim_1'])
    #add nse to dataframe
    param_varibility.loc[param_varibility['basin'] == int(basin_id), 'nse'] = nse_val

#plot parameter variability vs nse
plt.figure(figsize=(8, 5))
plt.scatter(param_varibility['avg_cv'], param_varibility['nse'], color='blue', alpha=0.6)
plt.xlabel('Average Coefficient of Variation of Parameters')
plt.ylabel('Nash-Sutcliffe Efficiency (NSE)')
plt.title('Parameter Variability vs Model NSE Performance | correlation: {:.2f}'.format(param_varibility['avg_cv'].corr(param_varibility['nse'])))
plt.grid(True)
plt.ylim(-0.5, 1)
plt.xlim(0, 0.075)
plt.savefig('figures/pars_correlation/parameter_variability_vs_nse.png', dpi=300)
plt.show()