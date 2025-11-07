# run_mlp_hbv.py
import pandas as pd
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from models.multi_hbv import MLPParameterNet, DifferentiableMHBV, constrain_multi_parameters   # custom imports

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

data_dir = "data/og_camels_compiled_data"
output_dir = "output/og_camels_multi_mlp_hbv"
scaler_path = "data/og_camels_mlp_static_scaler.pt"
model_path = "output/og_camels_multi_mlp_hbv/best_mlp_model.pt"
input_dim = len(static_feats_names) # 14 Number of static features
hidden_dim = 4056 # 4056 MLP hidden dimension
output_dim = 20 # Number of HBV parameters
batch_size = 128 # batch size
epochs = 1000 # 1000 Maximum number of training epochs
lr = 1e-4 # Learning rate
dropout = 0.4 # Dropout rate for MLP
spinup_days = 365*2 # Spin-up days for HBV model
sequence_length = spinup_days + 365  # Length of the input sequence for HBV model
stride_length = 60 # sliding window of stride length when creating sequences
num_ensemble = 3  # 3 number of MC dropout samples
early_stopping_patience = 4 # Patience for early stopping
lr_patience = 2 # Patience for learning rate reduction
num_hbv_units = 16 # predict 16 sets of HBV parameters per basin
test_batch_size = 64 #Number of basins to run in parallel during inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get file list from data directory
basin_list = pd.read_csv("data/camels531.csv")# ‼️‼️check correct list file
# basin_list = basin_list[basin_list['flag']=='diagnostic_sites']
# remove_diagnostic_sites = [2196000, 12010000, 6280300, 1144000, 3070500, 3182500, 12143600, 11148900]  # ‼️these sites have questionable flow data so remove
# basin_list = basin_list[~basin_list['name'].isin(remove_diagnostic_sites)].reset_index(drop=True)

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
        static_feats_all = []
        for f in file_list:
            df = pd.read_csv(f)
            # Filter rows only in the desired years
            # df = df[df["Year"].isin(years)]
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'].dt.year.isin(years)].reset_index(drop=True)

            static_feats = df[static_feats_names].iloc[0].values.astype("float32")
            precip = df["precip"].values.astype("float32")
            temp = ((df["tmax"] + df["tmin"]) / 2).values.astype("float32")
            qobs = df["qobs"].values.astype("float32")
            daylen = (df["daylenhr"]).values.astype("float32")
            total_days = len(df)
            # for start in range(0, total_days - sequence_length + 1, (sequence_length-spinup_days)):  # step by (sequence_length - spinup_days)
            for start in range(0, total_days - sequence_length + 1, stride_length): # step by stride_length
                end = start + sequence_length
                self.data.append({
                    "static": static_feats,
                    "precip": precip[start:end],
                    "temp": temp[start:end],
                    "qobs": qobs[start:end],
                    "daylen": daylen[start:end],
                })
                static_feats_all.append(static_feats)
        if fit_scaler and scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(np.stack(static_feats_all))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        static = d["static"]
        if self.scaler is not None:
            static = self.scaler.transform(static.reshape(1, -1)).squeeze(0).astype(np.float32)
        return (
            torch.tensor(static, dtype=torch.float32),
            torch.tensor(d["precip"], dtype=torch.float32),
            torch.tensor(d["temp"], dtype=torch.float32),
            torch.tensor(d["daylen"], dtype=torch.float32),
            torch.tensor(d["qobs"], dtype=torch.float32),
        )

# #-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
# # Training and Validation
# start_time = time.time()

# def masked_mse_loss(pred, target):
#     mask = ~torch.isnan(target)
#     if mask.sum() == 0:
#         return torch.tensor(0.0, device=target.device, requires_grad=True)
#     return ((pred[mask] - target[mask])**2).mean()

# # Datasets and Loaders
# #use total of 24 years of data: 1995 to 2018; 16 years for training, 8 years for validation
# train_ds = HBVDataset(file_list, years=list(range(1990, 2006)), fit_scaler=True) # ‼️ Use years 1990-2005 for training
# scaler = train_ds.scaler # Get the scaler from the training dataset
# torch.save(scaler, scaler_path) # save the scaler to a file

# valid_ds = HBVDataset(file_list, years=list(range(2006, 2015)), scaler=scaler) # ‼️ Use years 2006-2014 for validation

# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

# # Models
# mlp = MLPParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout).to(device)
# hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units).to(device)

# # Optimizer & Loss
# optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
# loss_fn = masked_mse_loss

# # Learning rate scheduler
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_patience, min_lr=1e-6)

# best_val_loss = float("inf")
# patience = early_stopping_patience # Early stopping patience
# epochs_no_improvement = 0
# # torch.autograd.set_detect_anomaly(True) # Enable anomaly detection during debugging NaNs
# for epoch in range(1, epochs + 1):
#     mlp.train()
#     total_loss = 0.0

#     for static, precip, temp, daylen, qobs in train_loader:
#         static = static.to(device)
#         precip = precip.to(device)
#         temp = temp.to(device)
#         daylen = daylen.to(device)
#         qobs = qobs.to(device)

#         # --- MLP predicts HBV parameters ---
#         pars = mlp(static)
#         pars = constrain_multi_parameters(pars, num_hbv_units)

#         # --- Spinup ---
#         hbv_states = hbv.run_spinup(pars, precip[:, :spinup_days],
#                                     temp[:, :spinup_days], daylen[:, :spinup_days])

#         # --- Main period with gradients ---
#         hbv.set_state(hbv_states)
#         qsim = hbv(pars, precip[:, spinup_days:], temp[:, spinup_days:], daylen[:, spinup_days:])

#         # --- Loss ---
#         loss = loss_fn(qsim, qobs[:, spinup_days:])

#         if any(torch.isnan(v).any() for v in [pars, qsim, loss]):
#             print("NaNs detected in parameters, states, or loss — skipping batch.")
#             continue

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
#         optimizer.step()

#         # --- Reset states for next batch ---
#         hbv.reset_state()

#         total_loss += loss.detach().cpu().item()

#     avg_train_loss = total_loss / len(train_loader)
    
#     # Validation
#     mlp.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for static, precip, temp, daylen, qobs in valid_loader:
#             static = static.to(device)
#             precip = precip.to(device)
#             temp = temp.to(device)
#             daylen = daylen.to(device)
#             qobs = qobs.to(device)

#             pars = mlp(static)
#             pars = constrain_multi_parameters(pars, num_hbv_units)

#             hbv_states = hbv.run_spinup(pars, precip[:, :spinup_days], temp[:, :spinup_days], daylen[:, :spinup_days])
#             hbv.set_state(hbv_states)
#             qsim = hbv(pars, precip[:, spinup_days:], temp[:, spinup_days:], daylen[:, spinup_days:])
#             loss = loss_fn(qsim, qobs[:, spinup_days:])

#             if torch.isnan(loss):
#                 continue
#             hbv.reset_state()

#             val_loss += loss.detach().cpu().item()

#     avg_val_loss = val_loss / len(valid_loader)
#     print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | lr: {optimizer.param_groups[0]['lr']:.6f}")

#     # Step the scheduler
#     scheduler.step(avg_val_loss)

#     # Early stopping
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(mlp.state_dict(), model_path)
#         epochs_no_improve = 0
#     else:
#         epochs_no_improve += 1

#     if epochs_no_improve >= patience:
#         print(f'Early stopping triggered at epoch {epoch}. No improvement for {patience} epochs.')
#         break
# print(f"Training complete in {(time.time() - start_time)/60:.2f} minutes")



#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
# Inference with MC Dropout
start_time = time.time()
mlp = MLPParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout)
mlp.load_state_dict(torch.load(model_path, map_location=device))
mlp.to(device)
mlp.train()  # ‼️‼️run mlp in train model to enable MC dropout

scaler = torch.load(scaler_path, weights_only=False)

hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units, use_crps=True).to(device)
hbv.eval()  # HBV is deterministic, eval mode is fine

# ----- INFERENCE -----
basin_list = pd.read_csv("data/camels531.csv") # ‼️‼️check if this is correct dataset
# remove_diagnostic_sites = [2196000, 12010000, 6280300, 1144000, 3070500, 3182500, 12143600, 11148900]  # ‼️these sites have questionable flow data so remove
# basin_list = basin_list[~basin_list['name'].isin(remove_diagnostic_sites)].reset_index(drop=True)
# basin_list = basin_list[basin_list['flag']=='diagnostic_sites']
gauge_id = basin_list["name"].values
# add a leading zero if gauge_id is numeric and has length 7
gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]
file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]

for i in range(0, len(file_list), test_batch_size):
    hbv.reset_state() # Reset HBV states before each batch
    batch_files = file_list[i:i + test_batch_size]

    dfs, static_list, precip_list, temp_list, daylen_list, qobs_list, basin_ids = [], [], [], [], [], [], []
    for fpath in batch_files:
        basin_id = os.path.basename(fpath).replace(".csv", "")
        df = pd.read_csv(fpath)
        dfs.append(df)
        basin_ids.append(basin_id)

        # Static input
        static = df[static_feats_names].iloc[0].values.astype("float32")
        static = scaler.transform(static.reshape(1, -1))
        static_list.append(static)

        # Dynamic inputs
        precip_list.append(df["precip"].values.astype("float32"))
        temp_list.append(((df["tmax"] + df["tmin"]) / 2).values.astype("float32"))
        daylen_list.append(df["daylenhr"].values.astype("float32"))
        qobs_list.append(df["qobs"].values)

    # Convert to tensors (assuming all basins have same T)
    static_tensor = torch.tensor(np.vstack(static_list), dtype=torch.float32).to(device)       # [B, F]
    precip = torch.tensor(np.stack(precip_list), dtype=torch.float32).to(device)                # [B, T]
    temp   = torch.tensor(np.stack(temp_list), dtype=torch.float32).to(device)                  # [B, T]
    daylen = torch.tensor(np.stack(daylen_list), dtype=torch.float32).to(device)                # [B, T]

    with torch.no_grad():
        ensemble_qsim = []
        for _ in range(num_ensemble):
            pars = mlp(static_tensor)  # [B, P]
            pars = constrain_multi_parameters(pars, num_hbv_units)
            qsim = hbv(pars, precip, temp, daylen)  # [B, U, T]
            ensemble_qsim.append(qsim.cpu().numpy())

    ensemble_qsim = np.stack(ensemble_qsim, axis=2)  # [B, U, num_ensemble, T]

    # Save per basin
    for b in range(len(batch_files)):
        df = dfs[b]
        qobs = qobs_list[b]
        out_df = pd.DataFrame({
            "date": df["date"],
            "qobs": np.round(qobs, 3)
        })
        for j in range(num_ensemble):
            # out_df[f"qsim_{j+1}"] = np.round(ensemble_qsim[b, :, j], 3)
            for u in range(num_hbv_units):
                out_df[f"qsim_{u+1}_{j+1}"] = np.round(ensemble_qsim[b, u, j, :], 3)
        out_df.to_csv(os.path.join(output_dir, f"pred_{basin_ids[b]}.csv"), index=False)

print(f"Inference complete in {(time.time() - start_time)/60:.2f} minutes")






# #-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#-------------------------------#--------------------------------#
# # Inference with MC Dropout
# start_time = time.time()
# mlp = MLPParameterNet(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim*num_hbv_units, dropout=dropout)
# mlp.load_state_dict(torch.load(model_path, map_location=device))
# mlp.to(device)
# mlp.train()  # ‼️‼️run mlp in train model to enable MC dropout

# scaler = torch.load(scaler_path, weights_only=False)

# hbv = DifferentiableMHBV(num_hbv_units=num_hbv_units).to(device)
# hbv.eval()  # HBV is deterministic, eval mode is fine

# # ----- INFERENCE -----
# basin_list = pd.read_csv("data/camels531.csv") # ‼️‼️check if this is correct dataset
# # remove_diagnostic_sites = [2196000, 12010000, 6280300, 1144000, 3070500, 3182500, 12143600, 11148900]  # ‼️these sites have questionable flow data so remove
# # basin_list = basin_list[~basin_list['name'].isin(remove_diagnostic_sites)].reset_index(drop=True)
# # basin_list = basin_list[basin_list['flag']=='diagnostic_sites']
# gauge_id = basin_list["name"].values
# # add a leading zero if gauge_id is numeric and has length 7
# gauge_id = [str(gid).zfill(8) if str(gid).isdigit() and len(str(gid))==7 else str(gid) for gid in gauge_id]
# file_list = [os.path.join(data_dir, f"input_{gauge_id}.csv") for gauge_id in gauge_id]

# for i in range(0, len(file_list), test_batch_size):
#     hbv.reset_state() # Reset HBV states before each batch
#     batch_files = file_list[i:i + test_batch_size]

#     dfs, static_list, precip_list, temp_list, daylen_list, qobs_list, basin_ids = [], [], [], [], [], [], []
#     for fpath in batch_files:
#         basin_id = os.path.basename(fpath).replace(".csv", "")
#         df = pd.read_csv(fpath)
#         dfs.append(df)
#         basin_ids.append(basin_id)

#         # Static input
#         static = df[static_feats_names].iloc[0].values.astype("float32")
#         static = scaler.transform(static.reshape(1, -1))
#         static_list.append(static)

#         # Dynamic inputs
#         precip_list.append(df["precip"].values.astype("float32"))
#         temp_list.append(((df["tmax"] + df["tmin"]) / 2).values.astype("float32"))
#         daylen_list.append(df["daylenhr"].values.astype("float32"))
#         qobs_list.append(df["qobs"].values)

#     # Convert to tensors (assuming all basins have same T)
#     static_tensor = torch.tensor(np.vstack(static_list), dtype=torch.float32).to(device)       # [B, F]
#     precip = torch.tensor(np.stack(precip_list), dtype=torch.float32).to(device)                # [B, T]
#     temp   = torch.tensor(np.stack(temp_list), dtype=torch.float32).to(device)                  # [B, T]
#     daylen = torch.tensor(np.stack(daylen_list), dtype=torch.float32).to(device)                # [B, T]

#     with torch.no_grad():
#         ensemble_qsim = []
#         for _ in range(num_ensemble):
#             pars = mlp(static_tensor)  # [B, P]
#             pars = constrain_multi_parameters(pars, num_hbv_units)
#             qsim = hbv(pars, precip, temp, daylen)  # [B, T]
#             ensemble_qsim.append(qsim.cpu().numpy())

#     ensemble_qsim = np.stack(ensemble_qsim, axis=2)  # [B, T, num_ensemble]

#     # Save per basin
#     for b in range(len(batch_files)):
#         df = dfs[b]
#         qobs = qobs_list[b]
#         out_df = pd.DataFrame({
#             "date": df["date"],
#             "qobs": np.round(qobs, 3)
#         })
#         for j in range(num_ensemble):
#             out_df[f"qsim_{j+1}"] = np.round(ensemble_qsim[b, :, j], 3)
#         out_df.to_csv(os.path.join(output_dir, f"pred_{basin_ids[b]}.csv"), index=False)

# print(f"Inference complete in {(time.time() - start_time)/60:.2f} minutes")
