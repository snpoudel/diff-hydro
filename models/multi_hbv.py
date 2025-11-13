# /models/hbv.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP networks for parameter estimation
class MLPParameterNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.sequential(x)


# LSTM networks for parameter estimation
class LSTMParameterNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use the last time step's output
        last_out = lstm_out[:, -1, :]
        # Use mean over all time steps
        # last_out = torch.mean(lstm_out, dim=1)
        last_out = self.dropout(last_out)
        last_out = self.fc(last_out)
        return last_out

#HBV model helper functions
# Function to compute potential evapotranspiration (PET) using Hamon's method
def compute_pet(temp, daylen, coeff_pet):
    esat = 0.611 * torch.exp(17.27 * temp / (temp + 237.3))  # saturation vapor pressure [kPa]
    potevap = coeff_pet * 29.8 * daylen * (esat / (temp + 273.2))  # Hamon's equation
    return potevap

# Function to perform triangular routing
def triangular_routing(qtotal, maxbas):
    B, T = qtotal.shape
    q_routed = torch.zeros_like(qtotal)

    for b in range(B):
        mb = int(torch.clamp(maxbas[b], min=1).item())  # Enforce min length 1 and convert to int
        # Create triangle weights: [1, 2, ..., mb//2, ..., 1]
        half = torch.arange(1, mb // 2 + 2, device=qtotal.device)
        if mb % 2 == 0: # if mb is even, the middle element is not repeated
            tri = torch.cat([half, half.flip(dims=[0])[:-1]])
        else: # if mb is odd, the middle element is repeated
            tri = torch.cat([half, half.flip(dims=[0])[1:]])
        weights = tri / tri.sum()
        kernel = weights.view(1, 1, -1)
        padded = F.pad(qtotal[b:b+1].unsqueeze(1), (kernel.size(-1) - 1, 0), mode='constant', value=0)
        routed = F.conv1d(padded, kernel).squeeze(1).squeeze(0)

        q_routed[b, :routed.shape[0]] = routed[:T]
    return q_routed


def gamma_unit_hydrograph(t, alpha, theta):
    eps = 1e-6
    return (t + eps).pow(alpha - 1) * torch.exp(-(t + eps) / theta) / (
        theta.pow(alpha) * torch.lgamma(alpha).exp()
    ) # pdf of gamma distribution


# Gamma routing that supports multiple HBV units using grouped conv
def apply_multi_gamma_routing(qin, alpha, theta):
    """
        qin:   [B, U, T] inflow per unit
        alpha: [B, U, 1] gamma shape parameter
        theta: [B, U, 1] gamma scale parameter
    Returns:
        qout: [B, U, T] routed flow per unit
    """
    B, U, T = qin.shape
    device = qin.device
    t = torch.arange(T, device=device).float()  # [T]

    # Compute UH per (B,U)
    uh = gamma_unit_hydrograph(t, alpha, theta)  # [B, U, T]
    uh = uh / uh.sum(dim=-1, keepdim=True)       # normalize
    uh = uh.flip(-1)                             # causal flip

    # Prepare for grouped convolution
    qin = qin.view(1, B*U, T)        # [1, B*U, T] (channels = B*U)
    kernel = uh.view(B*U, 1, T)      # [B*U, 1, T]
    padding = T - 1

    # Apply grouped convolution: one per (B,U)
    qout = F.conv1d(qin, kernel, padding=padding, groups=B*U)  # [1, B*U, T+T-1]
    qout = qout[:, :, :T]                                      # trim
    qout = qout.view(B, U, T)                                  # reshape back
    return qout


# Differentiable HBV model with multiple units
class DifferentiableMHBV(nn.Module):
    def __init__(self, routing=True, use_crps=False, num_hbv_units=5):
        super().__init__()
        self.routing = routing
        self.num_hbv_units = num_hbv_units
        self.use_crps = use_crps

        # Internal states (initialized later)
        self.state_upres = None
        self.state_lowres = None
        self.state_snow = None
        self.state_sliq = None
        self.state_sma = None

    def set_state(self, states):
        """Set HBV internal states from a warm-up run"""
        self.state_upres = states['state_upres']
        self.state_lowres = states['state_lowres']
        self.state_snow = states['state_snow']
        self.state_sliq = states['state_sliq']
        self.state_sma = states['state_sma']
    
    def reset_state(self):
        """Reset HBV internal states to zero"""
        self.state_upres = None
        self.state_lowres = None
        self.state_snow = None
        self.state_sliq = None
        self.state_sma = None

    def forward(self, pars_all, precip, temp, daylen, return_states=False):
        """
        Args:
            pars_all: [B, U, 20]
            precip, temp, daylen: [B, T]
            return_states: whether to return internal HBV states
        Returns:
            qavg: [B, T] or dict of states if return_states=True
        """
        B, U, P = pars_all.shape
        _, T = precip.shape
        assert U == self.num_hbv_units, "MLP output doesn't match num_hbv_units"

        # Unpack parameters
        (fc, beta, pwp, l, ks, ki, kb, kperc, coeff_pet,
         ddf, scf, ts, tm, tti, whc, crf,
         d_shape, d_scale, b_shape, b_scale) = torch.split(pars_all, 1, dim=-1)

        # Expand forcings to [B, U, T]
        precip = precip.unsqueeze(1).expand(-1, U, -1)
        temp   = temp.unsqueeze(1).expand(-1, U, -1)
        daylen = daylen.unsqueeze(1).expand(-1, U, -1)

        # Initialize states if first run
        if self.state_upres is None:
            self.state_upres = torch.zeros(B, U, 1, device=precip.device)
            self.state_lowres = torch.zeros(B, U, 1, device=precip.device)
            self.state_snow = torch.zeros(B, U, 1, device=precip.device)
            self.state_sliq = torch.zeros(B, U, 1, device=precip.device)
            self.state_sma = torch.zeros(B, U, 1, device=precip.device)

        # Outputs
        inflow_direct = torch.zeros(B, U, T, device=precip.device)
        inflow_base = torch.zeros(B, U, T, device=precip.device)

        # Compute PET
        potevap = compute_pet(temp, daylen, coeff_pet)

        for t in range(T):

            ## clamp states to a non-negative realistic range in each timestep to stabilize gradients in backprop
            # Also relu activate states throughout the model to avoid negative states
            self.state_upres = torch.clamp(self.state_upres, min=0.0, max=2000.0)
            self.state_lowres = torch.clamp(self.state_lowres, min=0.0, max=2000.0)
            self.state_snow = torch.clamp(self.state_snow, min=0.0, max=2000.0)
            self.state_sliq = torch.clamp(self.state_sliq, min=0.0, max=2000.0)
            self.state_sma = torch.clamp(self.state_sma, min=0.0, max=2000.0)

            # Current timestep inputs
            ct = temp[:, :, t:t+1]
            cp = precip[:, :, t:t+1]

            # --- Snow routine ---
            snowfrac = torch.clamp(-1 / (tti + 1e-3) * (ct - ts) + 1, 0.0, 1.0)
            snow = cp * snowfrac
            rain = cp * (1 - snowfrac)

            melt = torch.where(ct > tm, ddf * (ct - tm), torch.zeros_like(ct))
            melt = torch.min(melt, self.state_snow)

            self.state_snow = torch.relu(self.state_snow - melt)
            self.state_sliq = torch.relu(self.state_sliq + melt + rain)

            liqmax = torch.relu(self.state_snow * whc)
            pr_eff = torch.relu(self.state_sliq - liqmax)
            self.state_sliq = torch.minimum(self.state_sliq, liqmax)

            refreeze = torch.where(ct < tm, (tm - ct) * ddf * crf, torch.zeros_like(ct))
            refreeze = torch.minimum(refreeze, self.state_sliq)

            self.state_snow = torch.relu(self.state_snow + refreeze + snow * scf)
            self.state_sliq = torch.relu(self.state_sliq - refreeze)

            # --- Soil moisture ---
            eff_ratio_base = torch.clamp((self.state_sma / fc), min=1e-4, max=1.0)
            effratio = torch.pow(eff_ratio_base, beta)

            remainwater = pr_eff * (1 - effratio)
            added = torch.minimum(remainwater + self.state_sma, fc) - self.state_sma
            peff = pr_eff - remainwater
            self.state_sma = torch.relu(self.state_sma + added)

            # --- ET ---
            potevap_factor = torch.clamp(self.state_sma / (pwp * fc + 1e-3), min=0.0, max=1.0)
            pet = torch.where(
                self.state_sma > pwp * fc,
                potevap[:, :, t:t+1],
                potevap[:, :, t:t+1] * potevap_factor
            )
            et = torch.minimum(pet, self.state_sma)
            self.state_sma = torch.relu(self.state_sma - et)

            # --- Response routine ---
            self.state_upres = torch.relu(self.state_upres + peff)
            qs = torch.relu(self.state_upres - l) * ks
            qi = torch.minimum(l, self.state_upres) * ki
            qperc = torch.relu(self.state_upres - qs - qi) * kperc
            self.state_upres = self.state_upres - (qs + qi + qperc)
            self.state_upres = torch.relu(self.state_upres)
            qq = qs + qi # quickflow to direct runoff, sum of surface and interflow

            self.state_lowres = torch.relu(self.state_lowres + qperc)
            qb = self.state_lowres * kb # baseflow to groundwater runoff
            self.state_lowres = torch.relu(self.state_lowres - qb)

            inflow_direct[:, :, t] = qq.squeeze(-1)
            inflow_base[:, :, t] = qb.squeeze(-1)


        # --- Gamma routing ---
        qdirect = apply_multi_gamma_routing(inflow_direct, d_shape, d_scale)
        qbase   = apply_multi_gamma_routing(inflow_base, b_shape, b_scale)
        qtotal  = qdirect + qbase # [B, U, T] - Batch, HBV Units, Time
        qavg = qtotal.mean(dim=1)

        if return_states: # check this first
            return {
                'state_upres': self.state_upres,
                'state_lowres': self.state_lowres,
                'state_snow': self.state_snow,
                'state_sliq': self.state_sliq,
                'state_sma': self.state_sma
            }
        elif self.use_crps:
            return qtotal  # [B, U, T] return per-unit flows for CRPS loss
        else:
            return qavg    # [B, T] return average flow across units
        
    def run_spinup(self, pars_all, precip, temp, daylen):
        """Run HBV spinup with no gradient; returns final states"""
        with torch.no_grad():
            states = self.forward(pars_all, precip, temp, daylen, return_states=True)
        # detach to avoid accidental gradient tracking
        states = {k: v.detach() for k, v in states.items()}
        return states


# constrain MLP parameters for multiple HBV units
def constrain_multi_parameters(raw_pars, num_hbv_units):
    # raw_pars: [B, 20*U]
    bounds = torch.tensor([
        [1.0, 1000.0],   # fc
        [0.5, 5.0],    # beta
        [0.01, 0.99],  # pwp
        [1.0, 999.0],   # l
        [0.01, 0.99],  # ks
        [0.01, 0.99],  # ki
        [0.001, 0.99],  # kb
        [0.0001, 0.99],  # kperc
        [0.5, 2.0],    # coeff_pet
        [0.05, 10.0],  # ddf
        [0.5, 2.0],    # scf
        [-4.0, 4.0],   # ts
        [-4.0, 4.0],   # tm
        [0.1, 4.0],    # tti
        [0.05, 0.2],   # whc
        [0.1, 1.0],    # crf
        [1.0, 4.0],    # d_shape
        [0.5, 4.0],    # d_scale
        [1.0, 6.0],    # b_shape
        [1.0, 7.0],    # b_scale
    ], device=raw_pars.device)

    # Repeat bounds for each HBV unit → [20*U, 2]
    bounds = bounds.repeat(num_hbv_units, 1)

    mins = bounds[:, 0]
    maxs = bounds[:, 1]

    # Normalize raw parameters to [0, 1]
    normalized = torch.sigmoid(raw_pars)
    constrained = mins + (maxs - mins) * normalized

    # Reshape to [B, U, I]
    B = raw_pars.shape[0]
    U = num_hbv_units
    P = raw_pars.shape[1] // U # double divide makes this a int
    constrained = constrained.view(B, num_hbv_units, P) # batch, num_units, num_params
    return constrained

