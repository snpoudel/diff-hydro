"""
Equifinality test for multiple basins and parameters using LSTM + 1HBV model.
Loops through selected basins and parameters, generates simulations for high/low
parameter sets, and saves results as CSV files.

Author: Sandeep Poudel (1/12/2026)
"""
import pandas as pd
import torch
from models.multi_hbv import DifferentiableMHBV

# Settings
# ------------------------
basins = [
    ('01532000', 'High variability'),
    ('10336660', 'Medium variability'),
    ('02465493', 'Low variability')
]

parameters = ['fc', 'coeff_pet', 'kperc', 'ddf']

# Loop over basins and parameters
# ------------------------
for basin_id, variability in basins:
    print(f"\nProcessing Basin {basin_id} ({variability}) ...")
    
    # Load input file
    input_file = pd.read_csv(f'data/input_{basin_id}.csv')
    input_file['temp'] = (input_file['tmax'] + input_file['tmin']) / 2  # average temperature
    
    # Load parameter sets
    param_sets = pd.read_csv(f"output/best_lstm_1hbv_parameters/input_{basin_id}_parameters.csv")
    param_sets_no_date = param_sets.drop(columns=['date'])
    
    # Model
    hbv = DifferentiableMHBV(num_hbv_units=1)
    
    # Convert inputs to torch tensors
    precip = torch.tensor(input_file['precip'].values.reshape(1, -1), dtype=torch.float32)
    temp = torch.tensor(input_file['temp'].values.reshape(1, -1), dtype=torch.float32)
    daylenhr = torch.tensor(input_file['daylenhr'].values.reshape(1, -1), dtype=torch.float32)
    
    for param in parameters:
        # Get parameter sets for highest and lowest value
        high_row = param_sets_no_date[param_sets_no_date[param] == param_sets_no_date[param].max()].iloc[0].values
        low_row  = param_sets_no_date[param_sets_no_date[param] == param_sets_no_date[param].min()].iloc[0].values
        
        param_set_high = torch.tensor(high_row, dtype=torch.float32).reshape(1, 1, -1)
        param_set_low  = torch.tensor(low_row, dtype=torch.float32).reshape(1, 1, -1)
        
        # Run HBV model
        qsim_high = hbv(param_set_high, precip, temp, daylenhr)
        qsim_low  = hbv(param_set_low, precip, temp, daylenhr)
        
        # Save results
        results_df = pd.DataFrame({
            'date': input_file['date'],
            'qobs': input_file['qobs'].values,
            f'qsim_high_{param}': qsim_high.detach().numpy().flatten(),
            f'qsim_low_{param}': qsim_low.detach().numpy().flatten()
        })
        
        out_file = f'output/best_lstm_1hbv_equifinality/equifinality_test_{basin_id}_{param}.csv'
        results_df.to_csv(out_file, index=False)
        
        print(f"Saved results for Basin {basin_id}, Parameter {param}")
        print(f"High {param}: {param_set_high.flatten()[param_sets_no_date.columns.get_loc(param)]}")
        print(f"Low {param}: {param_set_low.flatten()[param_sets_no_date.columns.get_loc(param)]}")

