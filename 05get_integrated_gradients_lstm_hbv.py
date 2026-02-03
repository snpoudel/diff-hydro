'''
Run integrated gradients analysis to save attributions of input features to LSTM+HBV predicted HBV parameters.
Saves attributions as CSV files for each basin and each HBV parameter.

Author: Sandeep Poudel (1/12/2026)
'''

import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from models.multi_hbv import LSTMParameterNet, DifferentiableMHBV, constrain_multi_parameters   # custom imports
from captum.attr import IntegratedGradients

#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#
# Configuration

static_feats_names = [
    "elev_mean", "slope_mean", "area_gages2", "p_mean", "pet_mean", "aridity",
    "p_seasonality", "frac_snow", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "frac_forest", "lai_max", "lai_diff",
    "gvf_max", "gvf_diff", "dom_land_cover_frac", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity", "max_water_content",
    "sand_frac", "silt_frac", "clay_frac", "glim_1st_class_frac", "glim_2nd_class_frac",
    "carbonate_rocks_frac", "geol_permeability",
]
param_names = [ "fc", "beta", "pwp", "l", "ks", "ki", "kb", "kperc",
                "coeff_pet", "ddf", "scf", "ts", "tm", "tti", "whc",
                "crf", "d_shape", "d_scale", "b_shape", "b_scale"]

num_hbv_units = 1 # predict 1 set of HBV parameters per basin
hidden_dim = 512 # 512 LSTM hidden dimension
print(f"Using hidden dim: {hidden_dim} with hbv unit: {num_hbv_units}")
data_dir = "data"
output_dir = f"output/integrated_gradients/lstm_{num_hbv_units}hbv"
os.makedirs(output_dir, exist_ok=True)

scaler_path = f"{data_dir}/scaler_camels_lstm_hbv.pt"
model_path = f"output/tune_lstm_hbv/best_lstm_model_{num_hbv_units}hbv_{hidden_dim}hiddensize.pt"

input_dim = len(static_feats_names) + 3 # 14 Number of static features
output_dim = 20 # Number of HBV parameters
batch_size = 128 # batch size
epochs = 1000 # 100 Maximum number of training epochs
lr = 1e-4 # Learning rate
dropout = 0.4 # Dropout rate for LSTM
spinup_days = 365*2 # Spin-up days for HBV model
lstm_lookback = 365 # LSTM lookback days
stride_length = 60 # sliding window of stride length when creating sequences
early_stopping_patience = 5 # Patience for early stopping
lr_patience = 2 # Patience for learning rate reduction
test_batch_size = 128 #Number of basins to run in parallel during inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get file list from data directory
basin_list = pd.read_csv("camels531.csv")

#randomly select 20% of basins as test set
test_basin = basin_list.sample(frac=0.2, random_state=42).reset_index(drop=True)

train_basin = basin_list[~basin_list['name'].isin(test_basin['name'])].reset_index(drop=True)

gauge_id = train_basin["name"].values
# add a leading zero if gauge_id is numeric and has length 7
gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]
file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]

#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#
# Inference to save parameters for all basins
start_time = time.time()
lstm = LSTMParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout)
lstm.load_state_dict(torch.load(model_path, map_location=device))
lstm.to(device)
lstm.eval()  # ‼️‼️run lstm in train model to enable MC dropout

scaler = torch.load(scaler_path, weights_only=False)

hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units).to(device)
hbv.eval()  # HBV is deterministic, eval mode is fine

basin_list = pd.read_csv("camels531.csv") 
gauge_id = basin_list["name"].values
# add a leading zero if gauge_id is numeric and has length 7
gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]


# Iterate basin by basin to get parameters using integrated gradients
use_basins = ['01532000', '10336660', '02465493'] #basin with highes, medium, lowest parameter variance
for basin_id in use_basins:
    file_path = os.path.join(data_dir, f"input_{basin_id}.csv")
    if not os.path.exists(file_path):
        continue

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Only use years 2006–2009 for testing period
    df = df[df['date'].dt.year.isin(range(1990, 2006))].reset_index(drop=True)

    # Loop through output parameters
    for par_index in range(output_dim):  # HBV unit = 1 output here
        ig = IntegratedGradients(lambda x: lstm(x)[:, par_index])

        all_attributions = []
        all_dates = []

        # Build sequences
        for start in range(0, len(df) - lstm_lookback + 1, stride_length):
            end = start + lstm_lookback

            # Build features
            static_feats = df[static_feats_names].iloc[0].values.astype("float32")
            precip = df["precip"].values.astype("float32")
            temp = ((df["tmax"] + df["tmin"]) / 2).values.astype("float32")
            daylen = df["daylenhr"].values.astype("float32")

            static_repeated = np.tile(static_feats, (lstm_lookback, 1))
            dynamic = np.stack([precip[start:end], temp[start:end], daylen[start:end]], axis=1)
            concat_feats = np.concatenate([static_repeated, dynamic], axis=1)

            # Scale
            concat_feats_scaled = scaler.transform(concat_feats).astype(np.float32)
            concat_feats_tensor = torch.tensor(concat_feats_scaled).unsqueeze(0).to(device)

            baseline = torch.zeros_like(concat_feats_tensor)

            # Compute IG
            attributions, delta = ig.attribute(concat_feats_tensor, baseline, return_convergence_delta=True)

            all_attributions.append(attributions.cpu().numpy())

            # date logic: repeat final date for seq_len rows
            seq_date = df['date'].iloc[end - 1]
            all_dates.extend([seq_date] * lstm_lookback)


        # ---- SAVE SECTION ---- #
        attributions = np.concatenate(all_attributions, axis=0)
        num_samples, seq_len, input_dim = attributions.shape

        feature_names = static_feats_names + ["precip", "temp", "daylenhr"]

        df_save = pd.DataFrame(
            attributions.reshape(-1, input_dim),
            columns=feature_names
        )

        # Add lag and date
        df_save["lag"] = np.tile(np.arange(seq_len), num_samples) # Lag 0 is the most recent and lag seq_len-1 is the oldest observation
        df_save["date"] = all_dates

        # Save CSV
        output_file_path = os.path.join(output_dir, f"ig_lstm_hbv_{basin_id}_par_{param_names[par_index]}.csv")
        df_save.to_csv(output_file_path, index=False)

        print(f"Saved IG attributions for basin {basin_id} in {(time.time() - start_time)/60:.2f} minutes.")
