'''
Use best (from hyperparameter tuning) LSTM-HBV model and run to save HBV parameters for all input sequences.
Runs only for 1HBV unit model.

Author: Sandeep Poudel (1/12/2026)
'''
import pandas as pd
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from models.multi_hbv import LSTMParameterNet, DifferentiableMHBV, constrain_multi_parameters   # custom imports

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

num_hbv_units = 1 # predict 1 set of HBV parameters per basin
hidden_dim = 512 # 512 LSTM hidden dimension
print(f"Using hidden dim: {hidden_dim} with hbv unit: {num_hbv_units}")
data_dir = "data"
output_dir = f"output/best_lstm_{num_hbv_units}hbv_parameters"
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
sequence_length = spinup_days + 365  # Includes HBV spinup + HBV loss calculation period
lstm_lookback = 365 # LSTM lookback days - this is extracted from latest part of sequence_length
stride_length = 60 # sliding window of stride length when creating sequences
early_stopping_patience = 10 # Patience for early stopping
lr_patience = 5 # Patience for learning rate reduction
test_batch_size = 128 #Number of basins to run in parallel during inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get file list from data directory
basin_list = pd.read_csv("camels531.csv")

#randomly select 20% of basins as test set
test_basin = basin_list.sample(frac=0.2, random_state=42).reset_index(drop=True)
#save this as a csv file
test_basin.to_csv(os.path.join(output_dir, "test_basins.csv"), index=False)
train_basin = basin_list[~basin_list['name'].isin(test_basin['name'])].reset_index(drop=True)

gauge_id = train_basin["name"].values
# add a leading zero if gauge_id is numeric and has length 7
gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]
file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]

#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#
# Dataset and DataLoader
class HBVDataset(Dataset):
    """
    Dataset for HBV model.
    Loads multiple basin csv files and selects specified years.
    Each item returns (static_features, precip, temp, daylen, qobs) of 2-year sequences.
    Static features can be scaled with StandardScaler.
    """
    def __init__(self, file_list, years, scaler=None, fit_scaler=False):
        self.data = []
        self.scaler = scaler
        concat_feats_all = []
        for f in file_list:
            df = pd.read_csv(f)
            # Filter rows only in the desired years
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year.isin(years)].reset_index(drop=True)

            static_feats = df[static_feats_names].iloc[0].values.astype("float32")
            precip = df["precip"].values.astype("float32")
            temp = ((df["tmax"] + df["tmin"]) / 2).values.astype("float32")
            qobs = df["qobs"].values.astype("float32")
            daylen = (df["daylenhr"]).values.astype("float32")

            total_days = len(df)
            for start in range(0, total_days - sequence_length + 1, stride_length): # step by stride_length
                end = start + sequence_length

                # repeat static and dynamic across timesteps
                static_repeated = np.tile(static_feats, (sequence_length, 1))
                dynamic = np.stack([precip[start:end], temp[start:end], daylen[start:end]], axis=1)
                concat_feats = np.concatenate([static_repeated, dynamic], axis=1)

                self.data.append({
                    "concat_feats": concat_feats,
                    "precip": precip[start:end],
                    "temp": temp[start:end],
                    "qobs": qobs[start:end],
                    "daylen": daylen[start:end],
                })
                concat_feats_all.append(concat_feats)
        if fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            # stack across all sequences
            all_concat = np.concatenate(concat_feats_all, axis=0)  # (N_total_timesteps, input_dim)
            self.scaler.fit(all_concat)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx] # get the idx-th sample
        concat_feats = d["concat_feats"]
        if self.scaler is not None:
            concat_feats = self.scaler.transform(concat_feats).astype(np.float32)
        return (
            torch.tensor(concat_feats, dtype=torch.float32),
            torch.tensor(d["precip"], dtype=torch.float32),
            torch.tensor(d["temp"], dtype=torch.float32),
            torch.tensor(d["daylen"], dtype=torch.float32),
            torch.tensor(d["qobs"], dtype=torch.float32),
        )

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
file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]


for i in range(0, len(file_list), test_batch_size):
    hbv.reset_state() # Reset HBV states before each batch
    batch_files = file_list[i:i + test_batch_size]

    dfs, concat_feats_list, precip_list, temp_list, daylen_list, qobs_list, basin_ids = [], [], [], [], [], [], []
    for fpath in batch_files:
        basin_id = os.path.basename(fpath).replace(".csv", "")
        df = pd.read_csv(fpath)
        dfs.append(df)
        basin_ids.append(basin_id)

        # Concat features
        df["temp"] = (df["tmax"] + df["tmin"]) / 2 # add avg temp column
        concat_feats = df[static_feats_names + ["precip", "temp", "daylenhr"]]
        concat_feats = concat_feats.values.astype("float32")
        concat_feats = scaler.transform(concat_feats)
        concat_feats_list.append(concat_feats)

        # Dynamic inputs
        precip_list.append(df["precip"].values.astype("float32"))
        temp_list.append(((df["tmax"] + df["tmin"]) / 2).values.astype("float32"))
        daylen_list.append(df["daylenhr"].values.astype("float32"))
        qobs_list.append(df["qobs"].values)

    # Convert to tensors
    concat_feats_tensor = torch.tensor(np.stack(concat_feats_list), dtype=torch.float32).to(device)       # [B, T, input_dim]
    precip = torch.tensor(np.stack(precip_list), dtype=torch.float32).to(device)                # [B, T]
    temp   = torch.tensor(np.stack(temp_list), dtype=torch.float32).to(device)                  # [B, T]
    daylen = torch.tensor(np.stack(daylen_list), dtype=torch.float32).to(device)                # [B, T]


    with torch.no_grad():
        all_pars = []  # will store [num_sequences, B, P]
        for stride in range(0, concat_feats_tensor.size(1) - lstm_lookback + 1, stride_length):
            end = stride + lstm_lookback
            seq = concat_feats_tensor[:, stride:end, :]  # [B, T, input_dim]

            pars = lstm(seq)                            # [B, P]
            pars = constrain_multi_parameters(pars, num_hbv_units)  # [B, P]

            all_pars.append(pars.cpu().numpy())

        # Shape: [num_sequences, B, P] → transpose to [B, num_sequences, P]
        all_pars = np.stack(all_pars, axis=0).transpose(1, 0, 2)


    # Save per basin parameters
    for b in range(len(batch_files)):
        df = dfs[b].reset_index(drop=True)
        basin_id = basin_ids[b]

        pars_basin = all_pars[b]  # [num_sequences, P]
        num_sequences = pars_basin.shape[0]

        dates = pd.to_datetime(df["date"]).reset_index(drop=True)

        # match each sequence to its end date
        seq_dates = []
        for s in range(num_sequences):
            stride = s * stride_length
            end_idx = stride + lstm_lookback - 1
            end_idx = min(end_idx, len(dates) - 1)
            seq_dates.append(dates.iloc[end_idx].strftime("%Y-%m-%d"))

        # build final dataframe
        pars_df = pd.DataFrame(
            pars_basin,
            columns=[f"param_{j+1}" for j in range(pars_basin.shape[1])]
        )
        pars_df.insert(0, "date", seq_dates)

        # save
        fname = f"{basin_id}_lstm_{num_hbv_units}hbv_parameters.csv"
        pars_df.to_csv(os.path.join(output_dir, fname), index=False)

print(f"Inference complete in {(time.time() - start_time)/60:.2f} minutes")
