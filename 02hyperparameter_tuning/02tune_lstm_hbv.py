'''
This is a script to tune hyperparameters for LSTM-HBV model on CAMELS dataset.
It mainly tunes the hidden layer size for single HBV model and ensemble HBV model.
The following submission script dynamically sets hidden layer size and number of HBV units.

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

# set working directory to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(SCRIPT_DIR))

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

data_dir = "data"
output_dir = "output/tune_lstm_hbv" 
os.makedirs(output_dir, exist_ok=True)

scaler_path = f"{data_dir}/scaler_camels_lstm_hbv.pt"
input_dim = len(static_feats_names) + 3 # 14 Number of static features
hidden_dim = int(os.environ.get("HIDDEN_DIM", 256)) # Hidden layer size from environment variable or default to 256
output_dim = 20 # Number of HBV parameters
batch_size = 128 # batch size
epochs = 1000 # 100 Maximum number of training epochs
lr = 1e-4 # Learning rate
dropout = 0.4 # Dropout rate for LSTM
spinup_days = 365*2 # Spin-up days for HBV model
sequence_length = spinup_days + 365  # Includes HBV spinup + HBV loss calculation period
lstm_lookback = 365 # LSTM lookback days - this is extracted from latest part of sequence_length
stride_length = 60 # sliding window of stride length when creating sequences
# num_ensemble = 5  # 5 number of MC dropout samples, not used currently
early_stopping_patience = 5 # Patience for early stopping
lr_patience = 2 # Patience for learning rate reduction
num_hbv_units = int(os.environ.get("NUM_HBV_UNITS", 1)) # Number of HBV units from environment variable or default to 1
# test_batch_size = 128 #Number of basins to run in parallel during inference
model_path = f"{output_dir}/best_lstm_model_{num_hbv_units}hbv_{hidden_dim}hiddensize.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'Using hidden size: {hidden_dim}, num_hbv_units: {num_hbv_units}, device: {device}')

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

            # Concat static and dynamic features into sequences of length `sequence_length`

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
# Training and Validation
start_time = time.time()

def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=target.device, requires_grad=True)
    return ((pred[mask] - target[mask])**2).mean()

# Datasets and Loaders
#use total of 24 years of data: 1995 to 2018; 16 years for training, 8 years for validation
train_ds = HBVDataset(file_list, years=list(range(1990, 2006)), fit_scaler=True) # ‼️ Use years 1990-2005 for training
scaler = train_ds.scaler # Get the scaler from the training dataset
torch.save(scaler, scaler_path) # save the scaler to a file

valid_ds = HBVDataset(file_list, years=list(range(2006, 2015)), scaler=scaler) # ‼️ Use years 2006-2014 for validation

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# Models
lstm = LSTMParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout).to(device)
hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units).to(device)

# Optimizer & Loss
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)
loss_fn = masked_mse_loss

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience, min_lr=1e-6)

best_val_loss = float("inf")
patience = early_stopping_patience # Early stopping patience
epochs_no_improvement = 0
# torch.autograd.set_detect_anomaly(True) # Enable anomaly detection during debugging NaNs
for epoch in range(1, epochs + 1):
    #train
    lstm.train()
    total_loss = 0.0

    for concat_feats, precip, temp, daylen, qobs in train_loader: # For each batch in the training set; each batch has B basins
        concat_feats = concat_feats.to(device) # [B, T, input_dim]
        precip = precip.to(device) # [B, T] where T is the number of days
        temp = temp.to(device) # [B, T]
        daylen = daylen.to(device) # [B, T]
        qobs = qobs.to(device) # [B, T]
        pars = lstm(concat_feats)  # [B, 51] -> [B, 17] HBV parameters
        # --- LSTM predicts HBV parameters ---
        pars = lstm(concat_feats[:, -lstm_lookback:, :]) # only use lstm_lookback days for parameter prediction
        pars = constrain_multi_parameters(pars, num_hbv_units)

        # --- Spinup ---
        hbv_states = hbv.run_spinup(pars, precip[:, :spinup_days],
                                    temp[:, :spinup_days], daylen[:, :spinup_days])

        # --- Main period with gradients ---
        hbv.set_state(hbv_states)
        qsim = hbv(pars, precip[:, spinup_days:], temp[:, spinup_days:], daylen[:, spinup_days:])

        # --- Loss ---
        loss = loss_fn(qsim, qobs[:, spinup_days:])

        if any(torch.isnan(v).any() for v in [pars, qsim, loss]):
            print("NaNs detected in parameters, states, or loss — skipping batch.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Reset states for next batch ---
        hbv.reset_state()

        total_loss += loss.detach().cpu().item()

    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    lstm.eval()
    val_loss = 0.0
    with torch.no_grad():
        for concat_feats, precip, temp, daylen, qobs in valid_loader:
            concat_feats = concat_feats.to(device)
            precip = precip.to(device)
            temp = temp.to(device)
            daylen = daylen.to(device)
            qobs = qobs.to(device)

            # pars = lstm(concat_feats) # original code
            pars = lstm(concat_feats[:, -lstm_lookback:, :]) # only use lstm_lookback days for parameter prediction
            pars = constrain_multi_parameters(pars, num_hbv_units)

            hbv_states = hbv.run_spinup(pars, precip[:, :spinup_days], temp[:, :spinup_days], daylen[:, :spinup_days])
            hbv.set_state(hbv_states)
            qsim = hbv(pars, precip[:, spinup_days:], temp[:, spinup_days:], daylen[:, spinup_days:])
            loss = loss_fn(qsim, qobs[:, spinup_days:])

            if torch.isnan(loss):
                continue
            hbv.reset_state()

            val_loss += loss.detach().cpu().item()

    avg_val_loss = val_loss / len(valid_loader)
    print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

    # Step the scheduler
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(lstm.state_dict(), model_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f'Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs.')
        break
print(f"Training complete in {(time.time() - start_time)/60:.2f} minutes")
print(f"Best validation loss with hidden size of {hidden_dim}  with hbv unit of {num_hbv_units} is {best_val_loss:.4f} at epoch {epoch - epochs_no_improve}")