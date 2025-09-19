from .ode_jump import ODEJump
from .ode_jump_encoder import TimeHybridEncoding
import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(
        self,
        hidden_dim,
    ):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.lstm = nn.LSTMCell(hidden_dim,hidden_dim)
    def forward(self, x):
        """
        x  : (B, T, C)
        ts : (B, T)   segundos unix (normalizados ou não)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []

        for i in range(T):
            h,c = self.lstm(x[:, i], (h,c))                                 # jump
            states.append(h)

        H = torch.stack(states, dim=1)   # (B, T, hidden_dim)

        return H

class TS_LSTM(ODEJump):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        static_dim: int = 0,
        denoised: bool = False,
        lam: list[float,float] = [0.9, 0.1],
        cost_columns: list = None
        
    ):
        self.lam = lam
        super().__init__(in_channels, hidden_dim, static_dim, denoised, lam, cost_columns)
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )
        self.lstm = LSTM(hidden_dim)
        self.time_encoding = TimeHybridEncoding(hidden_dim)
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(self.model_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_x_hat: bool=False,
        mask = None,
        x_denoised: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        # Embedding de entrada
        if not already_latent:
            if x_denoised is not None:
                # aplique gate (treino=True quando model.training)
                x_fused, _ = self.denoise_gate(x, x_denoised, mask, train_mode=self.training)
                h_in = torch.cat([x_fused, mask], dim=-1)
            else:
                h_in = torch.cat([x, mask], dim=-1)

            h = self.encoder(h_in)  # (B,T,hidden_dim)
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        #tm_e = self.time_encoding(timestamps.to(h.dtype)).to(h.dtype)  # tempo contínuo
        #h = h + tm_e
        h = self.lstm(h)
        state = h
        return state,self.decoder(state) if return_x_hat else None    
        