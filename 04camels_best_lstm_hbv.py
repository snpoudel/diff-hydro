'''
Use best LSTM-HBV model saved from tuning experiment and make predictions on all CAMELS basins.
Run both 1hbv unit and 16 hbv units models.
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

#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
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

num_hbv_units = 1 # predict 16 sets of HBV parameters per basin
hidden_dim = 384 # 384 LSTM hidden dimension
print(f"Using hidden dim: {hidden_dim} with hbv unit: {num_hbv_units}")
data_dir = "data"
output_dir = f"output/best_lstm_{num_hbv_units}hbv"
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
num_ensemble = 1  #  number of MC dropout samples during inference
early_stopping_patience = 5 # Patience for early stopping
lr_patience = 2 # Patience for learning rate reduction
test_batch_size = 128 #Number of basins to run in parallel during inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get file list from data directory
basin_list = pd.read_csv("camels531.csv")# ‼️‼️check correct list file

#randomly select 20% of basins as test set
test_basin = basin_list.sample(frac=0.2, random_state=42).reset_index(drop=True)
#save this as a csv file
test_basin.to_csv(os.path.join(output_dir, "test_basins.csv"), index=False)
train_basin = basin_list[~basin_list['name'].isin(test_basin['name'])].reset_index(drop=True)

gauge_id = train_basin["name"].values
# add a leading zero if gauge_id is numeric and has length 7
gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]
file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]

#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
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
            # for start in range(0, total_days - sequence_length + 1, (sequence_length-spinup_days)):  # step by (sequence_length - spinup_days)
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


#-- Comment out training code below if using saved model from tuning experiment --##
#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
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
        # pars = lstm(concat_feats) # original code
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


#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
# Inference with MC Dropout
start_time = time.time()
lstm = LSTMParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout)
lstm.load_state_dict(torch.load(model_path, map_location=device))
lstm.to(device)
lstm.eval()  # ‼️‼️run lstm in train model to enable MC dropout

scaler = torch.load(scaler_path, weights_only=False)

hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units).to(device)
hbv.eval()  # HBV is deterministic, eval mode is fine

basin_list = pd.read_csv("camels531.csv") # ‼️‼️check if this is correct dataset
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

    # Convert to tensors (assuming all basins have same T)
    concat_feats_tensor = torch.tensor(np.stack(concat_feats_list), dtype=torch.float32).to(device)       # [B, T, input_dim]
    precip = torch.tensor(np.stack(precip_list), dtype=torch.float32).to(device)                # [B, T]
    temp   = torch.tensor(np.stack(temp_list), dtype=torch.float32).to(device)                  # [B, T]
    daylen = torch.tensor(np.stack(daylen_list), dtype=torch.float32).to(device)                # [B, T]

    with torch.no_grad():
        ensemble_qsim = []
        for _ in range(num_ensemble):
            pars = lstm(concat_feats_tensor)  # [B, P]
            pars = constrain_multi_parameters(pars, num_hbv_units)
            qsim = hbv(pars, precip, temp, daylen)  # [B, T]
            ensemble_qsim.append(qsim.cpu().numpy())

    ensemble_qsim = np.stack(ensemble_qsim, axis=2)  # [B, T, num_ensemble]

    # Save per basin
    for b in range(len(batch_files)):
        df = dfs[b]
        qobs = qobs_list[b]
        out_df = pd.DataFrame({
            "date": df["date"],
            "qobs": np.round(qobs, 3)
        })
        for j in range(num_ensemble):
            out_df[f"qsim_{j+1}"] = np.round(ensemble_qsim[b, :, j], 3)
        out_df.to_csv(os.path.join(output_dir, f"pred_{basin_ids[b]}.csv"), index=False)

print(f"Inference complete in {(time.time() - start_time)/60:.2f} minutes")
