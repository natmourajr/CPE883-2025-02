import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchdiffeq import odeint 
import sys
import os
import numpy as np
import math
# --- cole na classe TSDiffusion -------------------------------------------

t0 = 1735689600.0    
max_drop = 0.9
TS_SPAN = 60 * 60 * 24 * 365
lam = [0.25, 0.25, 0.4, 0.1] #Diferente do artigo para convergir mais rápido.
sys.path.append(f'{os.environ.get("path3W","../../../")}'+'3W')
if os.environ.get('path3WLoader'): sys.path.append(os.environ.get('path3WLoader'))
from loader import Loader3W
from sklearn.model_selection import TimeSeriesSplit

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Beta schedule baseado em Nichol & Dhariwal (2021):
    β_t = 1 − ᾱ_t / ᾱ_{t−1}, onde
    ᾱ_t = cos²( (t/T + s) / (1 + s) * π/2 )
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    # ᾱ desde t=0 até t=T
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    # β_t = 1 − ᾱ_t / ᾱ_{t−1}
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(min=1e-5,max=0.999)  # evita valores muito altos

class Time2Vec(nn.Module):
    """Embedding cíclico rápido (hora do dia + dia da semana)."""
    SECS_IN_DAY = 86_400.0
    SECS_INV    = 1.0 / SECS_IN_DAY   # 1/86400
    DAYS_INV    = 1.0 / 7.0           # 1/7

    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(2))
        self.b  = nn.Parameter(torch.randn(2))

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        """
        ts: segundos Unix  (float32/64 ou int64)  shape (B,T)
        devolve: (B,T,4)
        """
        # 1) converte p/ float32 uma única vez
        ts_f = ts.to(dtype=torch.float32)

        # 2) hora do dia  (0‑1)
        secs_norm = torch.remainder(ts_f, self.SECS_IN_DAY) * self.SECS_INV

        # 3) dia da semana  (0‑1)
        #    floor(ts/86400) % 7  →  remainder( … , 7 )
        dow_norm  = torch.remainder(ts_f.mul_(self.DAYS_INV), 7.0) * self.DAYS_INV
        #           ^ in‑place multiplica por 1/7 — evita uma divisão

        # 4) concatena sem stack (menos alocação)
        pos = torch.stack((secs_norm, dow_norm), dim=-1)    # (B,T,2)

        v0 = self.w0 * pos + self.b0                        # (B,T,2)
        vp = torch.sin(pos * self.w + self.b)               # (B,T,2)

        return torch.cat((v0, vp), dim=-1)                  # (B,T,4)                  # (B,T,4)

class ODEFunc(nn.Module):
    """f_θ usado no trecho contínuo  dh/dt = f_θ(h)."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, t, h):          # t é obrigatório p/ torchdiffeq
        return self.net(h)
        
        
class JumpODEEncoder(nn.Module):
    """
    Self‑Attentive Jump‑ODE simplificado:
    - GRUCell executa o *jump* g_ψ na chegada de cada evento (x_i, t_i)
    - ODEFunc integra h(t) entre eventos.
    - Self‑attention usa máscara para faltantes (opcional).
    """
    def __init__(self, in_dim, hidden_dim, attn_heads=4, num_layers=2, dropout=0.1, ff_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_heads = attn_heads
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            nhead=attn_heads,
            d_model=hidden_dim,
            norm_first=True,  # normalização antes da atenção
            dropout=dropout,  # dropout opcional
            dim_feedforward= self.hidden_dim * 4 or ff_dim,  # feedforward dimension
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    @staticmethod
    def _causal_mask(L, device):
        # Máscara triangular superior (impede olhar para o futuro)
        return torch.triu(torch.full((L, L), float('-inf'), device=device), diagonal=1)    
        
    def forward(self, x, ts):
        """
        x  : (B, T, C)
        ts : (B, T)   segundos unix (normalizados ou não)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []

        for i in range(T):
            if i > 0:
                dt_i = (ts[:, i] - ts[:, i-1]).float().unsqueeze(-1)  # (B,1)
                h = h + dt_i * self.odefunc(None, h)                 # Euler
            h = self.gru(x[:, i], h)                                 # jump
            states.append(h)

        H = torch.stack(states, dim=1)   # (B, T, hidden_dim)

        # máscara causal (L,L) para o Transformer (batch_first=True)
        causal = self._causal_mask(T, x.device)  # (T,T) com -inf acima da diagonal
        H = H + self.transformer(H, mask=causal) # (B, T, hidden_dim)
        return H

class DiffTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (estilo Vaswani et al.) + projeção linear.
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        half_dim = model_dim // 2
        # Registrar as frequências como buffer (não são parâmetros treináveis)
        freqs = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) 
            * math.log(10000.0) / half_dim
        )
        self.register_buffer('freqs', freqs)  # shape: (half_dim,)
        # Projeção final: mantém dimensão
        self.lin = nn.Linear(model_dim, model_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor de shape (batch,) ou (batch, 1) com timesteps inteiros.
        Returns:
            emb: (batch, model_dim) embedding do timestep.
        """
        # garante shape (batch,)
        t = t.view(-1)
        # multiplica t pelas frequências: result -> (batch, half_dim)
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        # concatena seno e cosseno -> (batch, model_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        # projeta de volta ao espaço de dimensão model_dim
        return self.lin(emb)
class TSDiffusion(nn.Module):
    """
    TS-Diffusion com forward, sample e impute alinhados ao train_model.
    """
    default_features = ['ABER-CKP','P-ANULAR','P-PDG','T-TPT','T-MON-CKP','T-PDG','T-TPT']
    def __init__(
        self,
        in_channels: int,
        status_dim: int,
        hidden_dim: int = 256,
        num_steps: int = 1000,
        n_heads: int = 4,
        n_layers: int = 4,
        static_dim: int = 0,
        
    ):
        super().__init__()
        self.val_loss = float('inf')
        self.model_dim = hidden_dim
        self.status_dim = status_dim
        self.num_steps = num_steps
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels),
        )
        # Projeções
        self.pos_enc = Time2Vec()
        self.pos_proj = nn.Linear(4, hidden_dim)
        self.t_embed = DiffTimeEmbedding(hidden_dim)
        self.static_dim = static_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, hidden_dim),
                nn.ReLU()
            )
        # (a) λ(t)  — intensity do ponto de observação
        self.lambda_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)       # escalar
        )
        # (b) μ_x(t)  — média gaussiana para L1
        '''self.mean_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.model_dim, self.in_channels)
        )'''
        # (c) μ_Tmax  — previsão do horizonte da série
        self.tmax_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, status_dim)
        )
        self.encoder_ode = JumpODEEncoder(hidden_dim, hidden_dim, attn_heads=n_heads, num_layers=n_layers)
        # (d) m_b  — probabilidade de observação (Bernoulli) para L4
        self.miss_head = nn.Linear(self.model_dim, 1)
        # Schedule de difusão
        betas = cosine_beta_schedule(num_steps)
        alphas = 1 - betas
        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_bar', torch.cumprod(alphas, dim=0))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        already_latent: bool=False,
        return_pred_state: bool=False,
        return_x_hat: bool=False,
        mask = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        noise = None
        # Embedding de entrada
        if not already_latent:
            x = self.encoder(torch.cat([x, mask], dim=-1))
            noise = torch.randn_like(x)
            ab = self.alpha_bar[t].view(-1, 1, 1)
            x = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
        h = x
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        # Positional encoding via Time2Vec
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        te = self.t_embed(t).unsqueeze(1)          # (b,1,model_dim)
        h = h + te
        h = self.encoder_ode(h, timestamps)   # (B,T,model_dim)

        # Transformer
        #h = h.permute(1, 0, 2)  # (seq_len, b, model_dim)
        #h = self.transformer(h)
        #h = h.permute(1, 0, 2)  # (b, seq_len, model_dim)
        # Previsão de ruído
        state = h
        return state, noise, self.decoder(state) if return_x_hat else None, self.tmax_head(state) if return_pred_state else None

    @torch.no_grad()
    def impute(
        self, x_obs, mask, timestamps, static_feats=None,
        sampling_steps=None, device=None
    ):
        self.eval()
        device = device or x_obs.device
        steps  = min(sampling_steps or self.num_steps, self.num_steps)

        # ----- estado inicial -----
        #z_obs = self.encoder(x_obs) if hasattr(self, "encoder") else x_obs
        #z     = z_obs * mask_latent + torch.randn_like(z_obs) * (1 - mask_latent)
        
        #z = self.encoder(x_obs)
        z = self.encoder(torch.cat([x_obs, mask], dim=-1)) 
        for i in reversed(range(steps)):

            a, ab = self.alpha[i], self.alpha_bar[i]
            t   = torch.full((z.size(0),), i, device=device, dtype=torch.long)
            state, _, _, _ = self.forward(
                z, t,
                timestamps=timestamps, static_feats=static_feats,
                already_latent=True 
            )
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab))*state)
            if i > 0:
                z = z + torch.sqrt(self.beta[i]) * torch.randn_like(z)

        x_hat = self.decoder(state) * (1 - mask) + x_obs * mask
        return x_hat

            #gamma = 0.1                                   # 0<γ≤1
            #x = x - gamma * self.decoder(eps)
            #x = x * (1 - mask) + mask * x_obs
            #z = self.encoder(x)
            #z = z-eps

            #z = self.encoder(x_hat)  # re-encode para o espaço latente
            #if i > 0:
            #    z = z + torch.sqrt(self.beta[i]) * torch.randn_like(z)
            #z = F.layer_norm(z, (self.latent_dim,)) 
            # reinsere valores observados
            #z = z * (1 - mask_latent) + z_obs * mask_latent
            #z = F.layer_norm(z, (self.latent_dim,)) 
        # ----- decodifica UMA única vez depois do loop -----

    # ------------------------------------------------------------------
    def _inverse_scale(self, z: torch.Tensor, feature_cols: list) -> np.ndarray:
        """
        Converte tensor (T,C) em z‑score para escala original usando stats.pkl.
        Retorna ndarray float64 (T,C).
        """
        if not hasattr(self, "_stats_cache"):
            loader = Loader3W(); loader.load_stats("stats.pkl")
            self._stats_cache = loader.stats        # memoize

        mu = torch.tensor(
            [self._stats_cache["mean"][c] for c in feature_cols],
            dtype=z.dtype, device=z.device
        )
        sd = torch.tensor(
            [self._stats_cache["std"][c]  for c in feature_cols],
            dtype=z.dtype, device=z.device
        ).clamp(min=1e-8)                          # evita div/0

        return (z * sd + mu).cpu().numpy()         # (T,C)
    # ------------------------------------------------------------------

    def test3W(
            self, 
            window_size: int = 600, 
            feature_cols: list = default_features + [f'state-{s}' for s in range(10)], 
            static_features_cols: list = [f'{f}_relative_max' for f in default_features], 
            predict_state_cols: list = [f'state-pred-{s}' for s in range(10)],
            batch_size: int = 32,
            test_datasets: int = 2,
            status_pred_window: int = 600
            ):
        loader = Loader3W()
        loader.load_stats('./stats.pkl')
        test = pd.DataFrame()
        datasets = loader.preprocess(include_status_pred=True,status_pred_window=status_pred_window)
        for num_dataset, dataset in enumerate(datasets):
            if num_dataset >= len(loader.stats['ids']) - test_datasets:
                test = pd.concat([test, dataset], ignore_index=True)

        test_loss = self.test_model(
            df_test=test,
            feature_cols=feature_cols,
            predict_state_cols=predict_state_cols,
            static_features_cols=static_features_cols,
            timestamp_col='index',
            window_size=window_size,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        print(f'Test Loss: {test_loss[0]:.6f}')
        print(f'Test L1: {test_loss[1]:.6f}, Test L2: {test_loss[2]:.6f}, Test L3: {test_loss[3]:.6f}, Test L4: {test_loss[4]:.6f}')

    def train3W(
            self, 
            window_size: int = 600, 
            feature_cols: list = default_features + [f'state-{s}' for s in range(10)], 
            static_features_cols: list = [f'{f}_relative_max' for f in default_features], 
            predict_state_cols: list = [f'state-pred-{s}' for s in range(10)],
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-3,
            test_datasets: int = 2,
            validate: bool = True,
            early_stopping: bool = True,
            patience: int = 5,
            status_pred_window: int = 600
            ):
        lower_loss = float('inf')
        test_patience = patience
        loader = Loader3W()
        loader.load_stats('./stats.pkl')
        for i in range(1, epochs+1):
            test = pd.DataFrame()
            datasets = loader.preprocess(include_status_pred=True,status_pred_window=status_pred_window)
            for num_dataset, dataset in enumerate(datasets):
                if num_dataset < len(loader.stats['ids']) - test_datasets:
                    print(f'Starting epoch {i}/{epochs} - dataset {num_dataset+1}/{len(loader.stats["ids"])} - Partial Validation Loss: {self.val_loss:.6f}' )
                    if validate:
                        tscv = TimeSeriesSplit(n_splits=5)
                        for train_idx, val_idx in tscv.split(dataset):
                            df_train = dataset.iloc[train_idx]
                            df_val = dataset.iloc[val_idx]
                            self.train_model(
                                df_train=df_train,
                                df_val=df_val,
                                feature_cols=feature_cols,
                                static_features_cols=static_features_cols,
                                predict_state_cols=predict_state_cols,
                                timestamp_col='index',
                                epochs=1,
                                batch_size=batch_size,
                                lr=lr,
                                window_size=window_size,
                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                verbose=False
                            )
                    else:
                        self.train_model(
                            df_train=dataset,
                            df_val=None,
                            feature_cols=feature_cols,
                            static_features_cols=static_features_cols,
                            predict_state_cols=predict_state_cols,
                            timestamp_col='index',
                            epochs=1,
                            batch_size=batch_size,
                            lr=lr,
                            window_size=window_size,
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                            verbose=False
                        )                    
                else:
                    test = pd.concat([test, dataset], ignore_index=True)

            test_loss = self.test_model(
                df_test=test,
                feature_cols=feature_cols,
                predict_state_cols=predict_state_cols,
                static_features_cols=static_features_cols,
                timestamp_col='index',
                window_size=window_size,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            if early_stopping and test_loss[0] < lower_loss:
                self.save(f'ts_diffusion.pt')
                lower_loss = test_loss[0]
                test_patience = patience
            else:
                test_patience -= 1
                if test_patience <= 0:
                    print(f'Early stopping at epoch {i}/{epochs} - Test Loss: {test_loss[0]:.6f}')
                    print(f'Test L1: {test_loss[1]:.6f}, Test L2: {test_loss[2]:.6f}, Test L3: {test_loss[3]:.6f}, Test L4: {test_loss[4]:.6f}')
                    break
            print(f'Epoch {i}/{epochs} completed - Test Loss: {test_loss[0]:.6f}')
            print(f'Test L1: {test_loss[1]:.6f}, Test L2: {test_loss[2]:.6f}, Test L3: {test_loss[3]:.6f}, Test L4: {test_loss[4]:.6f}')
    def _compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        tmax: torch.Tensor,
        state: torch.Tensor,
        ts_batch: torch.Tensor,
        mask: torch.Tensor,
        mask_train: torch.Tensor,
        noise: torch.Tensor,
        state_pred: torch.Tensor
    ):
        """
        Computa a perda de difusão para um lote de dados.
        """
        # ---------- L1: log-likelihood Gaussiano ponderado por λ ----------
        # λ(t) em (B,T,1) -> espreme p/ (B,T) e usa .unsqueeze(-1) na multiplicação
        lam_t = F.softplus(self.lambda_head(state)).clamp(1e-3, 50.0)  # (B,T,1)
        lam2  = lam_t.squeeze(-1)                                      # (B,T)
        mask_err = mask * (1 - mask_train)
        # erro ao longo dos C canais observados
        sqerr = ((x - x_hat)**2 * mask_err).sum(dim=-1, keepdim=True)      # (B,T,1)

        # -½ λ ||x-μ||^2  +  ½ log λ
        log_px = -0.5 * torch.log(sqerr * lam2.unsqueeze(-1) + 1e-8)                   # (B,T,1)
        # normaliza por quantidade de observações para não depender de C/T
        #obs_per_t = mask_err.sum(dim=-1).clamp(min=1.0)                    # (B,T)
        log_px = log_px.squeeze(-1) + 0.5 * torch.log(lam2 + 1e-8)  # (B,T)   # (B,T)
        l1_seq = log_px.sum(dim=-1)                      # (B,)
        L1 = -l1_seq.mean()
        
        #L1 = F.mse_loss(x*mask_err,x_hat*mask_err)
        # Perda do ruído
        L2 = F.mse_loss(state, noise) # (B,T)
                          # (B,T,S)
        offset_state_pred = state_pred - ts_batch.unsqueeze(-1)
        offset_tmax = tmax - ts_batch.unsqueeze(-1) 
        L3 = F.mse_loss(offset_tmax,offset_state_pred)
        # ----- L4 (máscara) -----
        # máscara binária: 1 se ao menos um canal está presente no timestep
                # (B, T, 1)
        m_t = mask_train.any(dim=2, keepdim=True).float()              # (B,T,1)
        mb_pred = torch.sigmoid(self.miss_head(state)).clamp(1e-4, 1-1e-4)  # (B, T, 1)
        L4 = F.binary_cross_entropy(mb_pred, m_t)
        #if L2.item()>0.3:
        #    loss = L2
        #else:
        loss = lam[0]*L1 + lam[1]*L2 + lam[2]*L3 + lam[3]*L4
        return loss, L1.item() * x.size(0), L2.item() * x.size(0), L3.item() * x.size(0), L4.item() * x.size(0)
    def test_model(
        self,
        df_test: pd.DataFrame,
        feature_cols: list,
        predict_state_cols: list,
        static_features_cols: list,
        timestamp_col: str,
        window_size: int = None,
        device: torch.device = None
    ):
        batch_size = 200
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_ds = self._make_dataset(df_test, timestamp_col, window_size, feature_cols, static_features_cols, predict_state_cols)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        self.eval()
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        total_l3 = 0.0
        total_l4 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 5:
                    x, ts_batch, m, p, s = batch
                else:                               # caso não haja static
                    x, ts_batch, m, p = batch;  s = None
                x, ts_batch, m, p = x.to(device), ts_batch.to(device), m.to(device), p.to(device)
                if s is not None: s = s.to(device)
                t = torch.randint(0, self.num_steps, (x.size(0),), device=device)

                # 2) probabilidade de *extra-missing* cresce com t
                p_drop_t = (t.float() / (self.num_steps - 1)) * max_drop   # (B,)
                p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
                rand_mask = (torch.rand_like(m) > p_drop_t).float()
                m_test   = m * rand_mask
                x_masked = x * m_test
                noise = torch.randn_like(x)
                #x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                state, noise, x_hat, tmax = self.forward(x_masked, t, timestamps=ts_batch, static_feats=s, return_x_hat=True, 
                                                         return_pred_state=True, mask=m_test)
                loss, L1, L2, L3, L4 = self._compute_loss(
                    x, x_hat, tmax, state, ts_batch, m, m_test, noise, p
                )
                total_loss += loss.item() * x.size(0)
                total_l1 += L1
                total_l2 += L2
                total_l3 += L3
                total_l4 += L4
                #print(f'Test Loss: {loss.item():.6f} | L1: {L1:.6f} | L2: {L2:.6f} | L3: {L3:.6f} | L4: {L4:.6f}')
                total_loss += (lam[0] * L1 + lam[1] * L2 + lam[2] * L3 + lam[3] * L4)
        avg_loss = total_loss / len(test_ds)
        avg_l1 = total_l1 / len(test_ds)
        avg_l2 = total_l2 / len(test_ds)
        avg_l3 = total_l3 / len(test_ds)
        avg_l4 = total_l4 / len(test_ds)
        return avg_loss, avg_l1, avg_l2, avg_l3, avg_l4
    
    @staticmethod
    def _make_dataset(df, timestamp_col, window_size, feature_cols, static_features_cols, predict_state_cols):
        if timestamp_col != 'index':
            df = df.sort_values(timestamp_col).reset_index(drop=True)
# ---------------- NORMALIZAÇÃO DO TEMPO ----------------
        if timestamp_col != "index":
            ts_raw = pd.to_datetime(df[timestamp_col]).astype("int64") / 1e9
        else:
            ts_raw = pd.to_datetime(df.index).astype("int64") / 1e9

        ts_np = ts_raw.to_numpy(dtype=np.float32)
        ts_rel = (ts_np - t0) / TS_SPAN                 # começa em 0

        # garante numérico; valores inválidos viram NaN
        state_pred = df[predict_state_cols].values

        # normaliza para [0,1] usando o mesmo t0/TS_SPAN do resto do código
        state_pred = (state_pred - t0) / TS_SPAN

        # para o tensor:
        state_pred = np.nan_to_num(state_pred, nan=0.0)
        data_state_pred = torch.tensor(state_pred, dtype=torch.float32)
        
        #span = ts_rel[-1] if ts_rel[-1] > 0 else 1  # evita div/0
        #ts_rel = ts_rel / span                      # agora 0‑1

        times = torch.from_numpy(ts_rel)            # (L,)           # (L,)
        values_np = df[feature_cols].values   
        mask_np   = ~pd.isna(values_np) 
        values_np = np.nan_to_num(values_np, nan=0.0)
        data  = torch.tensor(values_np, dtype=torch.float32)
        mask  = torch.tensor(mask_np,  dtype=torch.float32)  # (L,C)
        #times = torch.tensor(ts.values, dtype=torch.float32)
        static = torch.tensor(df[static_features_cols].values, dtype=torch.float32) if static_features_cols else None
        if window_size is None or window_size >= len(df):
            seqs = data.unsqueeze(0)
            ts_seqs = times.unsqueeze(0)
            stat_seqs = static[0].unsqueeze(0) if static is not None else None
            mask_seqs = mask.unsqueeze(0)   # (1,L,C)
            state_pred_seqs = data_state_pred.unsqueeze(0)
        else:
            n_ws = len(df) - window_size + 1
            seqs = torch.stack([data[i:i+window_size] for i in range(n_ws)])
            ts_seqs = torch.stack([times[i:i+window_size] for i in range(n_ws)])
            mask_seqs = torch.stack([mask[i:i+window_size] for i in range(n_ws)])
            stat_seqs = static[0].unsqueeze(0).repeat(n_ws, 1)  if static is not None else None
            state_pred_seqs = torch.stack([data_state_pred[i:i+window_size] for i in range(n_ws)])
        if stat_seqs is None:
            return TensorDataset(seqs, ts_seqs, mask_seqs,state_pred_seqs)                   # 3 itens
        return TensorDataset(seqs, ts_seqs, mask_seqs, state_pred_seqs, stat_seqs)   
    # --------------------------------------------------------------------------
    # NOVO MÉTODO: sample_continue --------------------------------------------
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def sample_continue(
        self,
        x_prefix: torch.Tensor,      # (B, L0, C)  – dados já conhecidos (z‑score!)
        ts_prefix: torch.Tensor,     # (B, L0)     – segundos unix norm. 0‑1
        n_future: int,               # passos a gerar
        delta_t: float = 1.0,        # espaçamento (mesma unidade usada no treino)
        static_feats: torch.Tensor = None,
        sampling_steps: int = None,
    ):
        """
        Continua a série acrescentando `n_future` pontos após o prefixo.

        Retorna:
            times_full  – (B, L0+n_future)
            values_full – (B, L0+n_future, C)  (z‑score)
        """
        self.eval()
        device = x_prefix.device
        B, L0, C = x_prefix.shape
        L = L0 + n_future

        # ----- grade temporal completa ---------------------------------------
        last_t = ts_prefix[:, -1:]              # (B,1)
        fut_grid = torch.arange(
            1, n_future + 1, device=device, dtype=torch.float32
        ).unsqueeze(0) * delta_t + last_t       # (B, n_future)
        ts_full = torch.cat([ts_prefix, fut_grid], dim=1)  # (B, L)

        # normaliza 0‑1 exatamente como _make_dataset()
        span = ts_full[:, -1:] - ts_full[:, 0:1]
        ts_full_n = (ts_full - ts_full[:, 0:1]) / span.clamp(min=1.0)

        # ----- tensor de dados + máscara --------------------------------------
        x_full   = torch.zeros(B, L, C, device=device, dtype=x_prefix.dtype)
        mask_full = torch.zeros(B, L, C, device=device, dtype=x_prefix.dtype)

        x_full[:, :L0] = x_prefix
        mask_full[:, :L0] = 1.0                 # prefixo observado

        # static feats opcional
        if static_feats is not None:
            static_feats = static_feats.to(device)
            if static_feats.dim() == 1:
                static_feats = static_feats.unsqueeze(0)  # (1,D)

        # ----- chama imputação (gera valores onde mask==0) --------------------
        imputed = self.impute(
            x_obs=x_full,
            mask=mask_full,
            timestamps=ts_full_n,
            static_feats=static_feats,
            sampling_steps=sampling_steps,
            device=device
        )
        return ts_full_n.cpu().numpy(), imputed.cpu().numpy()    

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        feature_cols: list,
        static_features_cols: list,
        predict_state_cols: list,
        timestamp_col: str,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        window_size: int = None,
        device: torch.device = None,
        verbose: bool = True,

    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_ds = self._make_dataset(df_train, timestamp_col, window_size, feature_cols, static_features_cols,predict_state_cols)
        val_ds = self._make_dataset(df_val, timestamp_col, window_size, feature_cols, static_features_cols, predict_state_cols)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if df_val is not None else None
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.999),
                                      weight_decay=1e-2)
        self.to(device).train()
        for epoch in range(1, epochs + 1):
            total_train = 0.0
            self.train()
            for batch in train_loader:
                if len(batch) == 5:
                    x, ts_batch, m, p, s = batch
                else:                               # caso não haja static
                    x, ts_batch, m, p = batch;  s = None
                x, ts_batch, m, p = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True), p.to(device, non_blocking = True)
                if s is not None: s = s.to(device, non_blocking = True)
                t = torch.randint(0, self.num_steps, (x.size(0),), device=device)

                # 2) probabilidade de *extra-missing* cresce com t
                p_drop_t = (t.float() / (self.num_steps - 1)) * max_drop   # (B,)
                p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
                rand_mask = (torch.rand_like(m) > p_drop_t).float()
                m_train   = m * rand_mask
                x_masked = x * m_train
                #x_t = torch.sqrt(ab) * x_masked + torch.sqrt(1 - ab) * noise
                #m_latent = self._make_mask(m_train, self.latent_dim)  # (B,T,latent_dim)
                #m_model = self._make_mask(m_train, self.model_dim)    # (B,T,model_dim)
                state, noise, x_hat, tmax  = self.forward(x_masked, t, timestamps=ts_batch, static_feats=s, 
                                                          return_pred_state=True, return_x_hat=True, mask=m_train)
                # ---------- cabeças ----------
                loss,_,_,_,_= self._compute_loss(
                    x, x_hat, tmax, state, ts_batch, m, m_train,noise, p
                )
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_train += loss.item() * x.size(0) * batch_size
            if df_val is not None:
                total_val = 0.0
                total_l1 = 0.0
                total_l2 = 0.0
                total_l3 = 0.0
                total_l4 = 0.0
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 5:
                            x, ts_batch, m, p, s = batch
                        else:                               # caso não haja static
                            x, ts_batch, m = batch;  s = None
                        x, ts_batch, m, p = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True),p.to(device, non_blocking = True)
                        if s is not None: s = s.to(device, non_blocking = True)
                        t = torch.randint(0, self.num_steps, (x.size(0),), device=device)

                        # 2) probabilidade de *extra-missing* cresce com t
                        p_drop_t = (t.float() / (self.num_steps - 1)) * max_drop   # (B,)
                        p_drop_t = p_drop_t.view(-1, 1, 1)                         # broadcast
                        rand_mask = (torch.rand_like(m) > p_drop_t).float()
                        m_val   = m * rand_mask
                        x_masked = x * m_val
                        state , noise, x_hat, tmax = self.forward(x_masked, t, timestamps=ts_batch, static_feats=s, mask=m_val, return_x_hat=True, return_pred_state=True)
                        loss, L1, L2, L3, L4 = self._compute_loss(
                            x, x_hat, tmax,state, ts_batch, m, mask_train=m_val, noise=noise, state_pred=p
                        )
                        total_val += (lam[0]*L1 + lam[1]*L2 + lam[2]*L3 + lam[3]*L4)
                        total_l1 += L1
                        total_l2 += L2
                        total_l3 += L3
                        total_l4 += L4
                if verbose:
                    print(f"Epoch {epoch}/{epochs} — Train Loss: {total_train/len(train_ds):.6f} — Val Loss: {total_val/len(val_ds):.6f}")
                else: 
                    self.loss = total_train / len(train_ds)
                    self.val_loss = total_val / len(val_ds)
        print(f"Ep {epoch}: L1={total_l1/len(val_ds):.2f} L2={total_l2/len(val_ds):.2f} "
    f"L3={total_l3/len(val_ds):.2f} L4={total_l4/len(val_ds):.2f}")

    def _make_mask(self, m, dim):
        if dim % self.in_channels == 0:
            r = dim // self.in_channels
            return m.repeat_interleave(r, dim=2).float()
        else:
            return m.any(dim=-1, keepdim=True).expand(-1, -1, dim).float()

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model


    # ------------------------------------------------------------------
    # utilitário: escolhe janela com dados suficientes -----------------
    @staticmethod
    def _pick_window(df: pd.DataFrame,
                    feature_cols: list,
                    seq_len: int):
        """
        Devolve (df_win, start_idx).
        Garante que cada feature tenha ≥ min_valid*seq_len valores não‑nulos.
        Se não existir tal janela, devolve a primeira (com NaNs mesmos).
        """

        for start in range(0, len(df) - seq_len + 1):
            win = df.iloc[start:start + seq_len][feature_cols]
            return df.iloc[start:start + seq_len], start
    # ------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # FORECAST RECURSIVO  ---------------------------------------------------
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def forecast_recursive(
        self,
        x_hist:      torch.Tensor,   # (B, L0, C)   – série original em z‑score
        t_hist:      torch.Tensor,   # (B, L0)      – segundos normalizados 0‑1
        window_size: int,            # comprimento da janela deslizante
        n_future:    int,            # passos a prever
        delta_t:     float = 1.0,    # espaçamento real em segundos
        static_feats: torch.Tensor = None,
        sampling_steps: int = None,
    ):
        """
        Gera sequência recursivamente, 1 passo por vez, usando janela deslizante.
        Retorna (times_full, values_full) – ambos com shape (B, L0+n_future, C)
        """
        self.eval()
        device = x_hist.device
        B, L0, C = x_hist.shape
        assert L0 >= window_size, "histórico deve ter ≥ window_size"

        # listas para acumular
        times_list  = [t_hist.clone()]            # cada item shape (B, Li)
        values_list = [x_hist.clone()]

        # prepara static feats
        if static_feats is not None:
            static_feats = static_feats.to(device)
            if static_feats.dim() == 1:
                static_feats = static_feats.unsqueeze(0)

        # tempo real do último ponto (para avançar)
        last_t_real = t_hist[:, -1:] * (t_hist[:, -1:] > 0).float()  # já normalizado 0‑1

        for step in range(1, n_future + 1):
            # --- 1) constrói nova grade temporal --------------------------------
            #   último real + Δt (aqui Δt já normalizado pela janela original)
            t_next_real = last_t_real + (delta_t / (delta_t * (L0 - 1)))  # normalizado
            #   série de índices para a janela atual
            t_win = torch.cat([times_list[-1][..., -window_size+1:], t_next_real], dim=1)

            # --- 2) dados da janela ---------------------------------------------
            x_win  = values_list[-1][..., -window_size+1:, :]              # (B, W-1, C)
            x_pad  = torch.zeros(B, 1, C, device=device, dtype=x_hist.dtype)
            x_win  = torch.cat([x_win, x_pad], dim=1)                      # (B, W, C)

            mask_win = torch.ones_like(x_win)
            mask_win[:, -1] = 0.0                                          # último ausente

            # --- 3) imputação de 1 passo ----------------------------------------
            x_imp = self.impute(
                x_obs=x_win,
                mask=mask_win,
                timestamps=t_win,
                static_feats=static_feats,
                sampling_steps=sampling_steps,
                device=device
            )
            x_next = x_imp[:, -1:]                                          # (B,1,C)

            # --- 4) anexa resultado e atualiza referências ----------------------
            times_list.append(t_next_real)
            values_list.append(x_next)
            last_t_real = t_next_real

        # concatena tudo
        times_full  = torch.cat(times_list,  dim=1)    # (B, L0+n_future)
        values_full = torch.cat(values_list, dim=1)    # (B, L0+n_future, C)
        return times_full.cpu().numpy(), values_full.cpu().numpy()


# --------------------------------------------------------------------------
# TEST_SAMPLER  v2  --------------------------------------------------------
# --------------------------------------------------------------------------
    @torch.no_grad()
    def test_sampler(
        self,
        dataset_idx: int           = 0,
        feature_cols: list         = None,
        static_features_cols: list = None,
        window_size: int           = 48,     # tamanho da janela usada no passo‑a‑passo
        prefix_len: int            = 48,     # histórico inicial (≥ window_size)
        future_len: int            = 24,     # passos a prever recursivamente
    ):
        """
        Retorna {feature: (real_t, real_val, gen_t, gen_val)} em **z‑score**.
        O futuro é gerado recursivamente, 1 passo por vez, usando janela deslizante.
        """
        loader = Loader3W(); loader.load_stats("stats.pkl")
        df_all = loader.preprocess().send(None).sort_index()
        # -------- 1. lista de features coerente ------------------------------
        if feature_cols is None:
            state_cols = sorted([c for c in df_all.columns if c.startswith("state-")])
            feature_cols = list(self.default_features) + state_cols
        if len(feature_cols) != self.in_channels:
            raise ValueError(f"Modelo espera {self.in_channels} colunas, "
                            f"mas feature_cols tem {len(feature_cols)}.")

        if static_features_cols is None:
            static_features_cols = [f"{f}_relative_max" for f in self.default_features]

        # -------- 2. carrega dataset e escolhe janela ------------------------
        
        df_win, _ = self._pick_window(df_all, feature_cols, prefix_len)
        

        ts_raw = pd.to_datetime(df_all.index).astype("int64") / 1e9

        ts_np = ts_raw.to_numpy(dtype=np.float32)
        ts_rel = (ts_np - t0) / TS_SPAN                 # começa em 0
        # -------- 3. eixo temporal robusto (segundos) ------------------------


        # delta_t real (última diferença não‑nula)
        diffs = np.diff(ts_rel)
        delta_real = float(diffs[diffs > 0].min()) if (diffs > 0).any() else 1.0

        # normaliza 0‑1 como no treino
        t_norm = ts_rel

        # -------- 4. dados e máscara ----------------------------------------
        x_np = df_win[feature_cols].to_numpy(dtype=np.float32)
        x_np[np.isnan(x_np)] = 0.0
        x_t  = torch.tensor(x_np[None], device=self.beta.device)  # (1,L0,C)
        t_t  = torch.tensor(t_norm[None], device=self.beta.device)

        # static feats
        stat = None
        if static_features_cols and self.static_dim > 0:
            stat_vals = (
                pd.to_numeric(df_win.iloc[0][static_features_cols], errors="coerce")
                .fillna(0.).to_numpy(dtype=np.float32)
            )
            stat = torch.tensor(stat_vals).unsqueeze(0).to(self.beta.device)

        # -------- 5. geração recursiva --------------------------------------
        gen_t, gen_vals = self.forecast_recursive(
            x_hist=x_t,
            t_hist=t_t,
            window_size=window_size,
            n_future=future_len,
            delta_t=delta_real,
            static_feats=stat,
        )
        gen_t, gen_vals = gen_t[0], gen_vals[0]                # remove batch

        # -------- 6. saída ---------------------------------------------------
        out = {}
        for j, feat in enumerate(feature_cols):
            out[feat] = (
                t_norm,                       # tempo real normalizado do prefixo
                df_win[feat].values,          # valores reais do prefixo
                gen_t,                        # eixo completo (prefixo+futuro)
                gen_vals[:, j],               # valores gerados (z‑score)
            )
        return out

    @torch.no_grad()
    def evaluate_datasets(
            self,
            test_datasets: int = 2,
            window_size:   int = 600,
            batch_size:    int = 256,
            missing_frac:  float = 0.2,
            sampling_steps:int = 40,
            device:        torch.device = None,
            seed:          int = 42,
    ):
        """
        Avalia os *N* últimos datasets do Loader3W.
        Retorna: metrics[dataset_id][feature] = {"mse":…, "mae":…, "rmse":…}
        """
        device = device or next(self.parameters()).device
        self.eval()

        loader = Loader3W(); loader.load_stats("stats.pkl")
        all_ds = pd.DataFrame()
        for i,ds in enumerate(loader.preprocess()):
            if i >= len(loader.stats['ids'])-2:
                all_ds = pd.concat([all_ds,ds],ignore_index=True)
        base = list(self.default_features)
        state = sorted([c for c in all_ds.columns if c.startswith("state-")])
        feature_cols = (base + state)[: self.in_channels]
        if len(feature_cols) != self.in_channels:
            raise ValueError("in_channels incompatível.")

        static_cols = [c for c in all_ds.columns if c.endswith("_relative_max")]
        state_pred_cols = [c for c in all_ds.columns if c.startswith('state-pred')]
        all_ids = list(loader.stats['ids'])
        # pega só os *N* últimos
        eval_pairs = list(zip(all_ids[-test_datasets:], all_ds))

        rng = np.random.default_rng(seed)
        df = all_ds
        # ---------- dataset completo em tensores ----------
        t_dataset = self._make_dataset(
            df, timestamp_col='index',
            window_size=window_size,
            feature_cols=feature_cols,
            static_features_cols=static_cols,
            predict_state_cols=state_pred_cols
        )
        loader_ds = DataLoader(t_dataset, batch_size=batch_size)
        feat_names = feature_cols

        # acumuladores por feature
        mse_noise = np.zeros(len(feat_names))
        rmse_imp  = np.zeros(len(feat_names))
        mae_imp   = np.zeros(len(feat_names))
        n_points  = np.zeros(len(feat_names))

        for batch in loader_ds:
            # ------------------- unpack -------------------
            if len(batch) == 5:
                x, ts_batch, m, p, s = batch
            else:
                x, ts_batch, m, p = batch; s = None
            x, ts_batch, m, p = [t.to(device) for t in (x, ts_batch, m, p)]
            s = s.to(device) if s is not None else None

            # ------------------- Difusão (ruído) ----------
            t_rand = torch.randint(0, self.num_steps, (x.size(0),),
                                device=device)
            z = self.encoder(torch.cat([x, m], dim=-1))  # (B, T, C)
            noise = torch.randn_like(z)  # (B, T, C)
            ab = self.alpha_bar[t_rand].view(-1, 1, 1)  # (B, T, C)
            z = torch.sqrt(ab) * z + torch.sqrt(1 - ab) * noise

            # ------------------- Difusão (saída) ----------
            #mse_noise += F.mse_loss(eps, noise,
                                    #reduction='none').sum((0,1)).cpu().numpy()

            # ------------------- Imputação ----------------
            #   gera máscara faltante sintética p/ avaliar imputação
            miss_mask = m.clone()
            # sorteia valores a serem “apagados”
            mask_flat = miss_mask.reshape(-1, miss_mask.size(-1)).cpu().numpy()
            idx_del   = rng.choice(mask_flat.shape[0],
                                int(mask_flat.shape[0]*missing_frac),
                                replace=False)
            mask_flat[idx_del] = 0
            miss_mask = torch.tensor(mask_flat.reshape(miss_mask.shape),
                                    device=device,
                                    dtype=m.dtype)

            x_imp = self.impute(
                x_obs=x, mask=miss_mask,
                timestamps=ts_batch,
                static_feats=s,
                sampling_steps=sampling_steps,
                device=device
            )

            diff = (x_imp - x).abs() * (1 - miss_mask)  # só onde NaN
            mae_imp += diff.sum((0,1)).cpu().numpy()
            rmse_imp += (diff**2).sum((0,1)).cpu().numpy()
            n_points += (1 - miss_mask).sum((0,1)).cpu().numpy()

        # normaliza
        mae_imp  /= n_points.clip(min=1)
        rmse_imp = np.sqrt(rmse_imp / n_points.clip(min=1))
        #mse_noise /= len(t_dataset) * t_dataset.tensors[0].shape[1]

        # salva
        metrics_out = {
            feat: {
                #"mse": float(mse_noise[i]),
                "mae": float(mae_imp[i]),
                "rmse": float(rmse_imp[i])}
            for i, feat in enumerate(feat_names)
            }

        return metrics_out

    @staticmethod
    def plot_metrics_matplotlib(metrics_dict):
        """
        Gera 3 gráficos para um único dataset:
        1. Barras de RMSE por feature
        2. Barras de MAE por feature
        3. Linha de MSE por feature
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # Organiza os dados em DataFrame
        records = []
        for feature, vals in metrics_dict.items():
            records.append({
                "feature": feature,
                #"mse":   vals["mse"],
                "mae":   vals["mae"],
                "rmse":  vals["rmse"]
            })
        df = pd.DataFrame(records)

        # 1) Barras de RMSE
        plt.figure(figsize=(12, 4))
        sns.barplot(data=df, x="feature", y="rmse")
        plt.xticks(rotation=45)
        plt.ylabel("RMSE")
        plt.title("RMSE por Feature")
        plt.tight_layout()
        plt.show()

        # 2) Barras de MAE
        plt.figure(figsize=(12, 4))
        sns.barplot(data=df, x="feature", y="mae")
        plt.xticks(rotation=45)
        plt.ylabel("MAE")
        plt.title("MAE por Feature")
        plt.tight_layout()
        plt.show()

        # 3) Linha de MSE
        '''plt.figure(figsize=(12, 4))
        plt.plot(df["feature"], df["mse"], marker="o")
        plt.xticks(rotation=45)
        plt.ylabel("MSE")
        plt.title("MSE por Feature")
        plt.grid(True)
        plt.tight_layout()
        plt.show()'''

    # ---------------------------------------------------------------------
    # 3)  TESTE DA IMPUTAÇÃO  --------------------------------------------
    # ---------------------------------------------------------------------
# ------------------------------------------------------------------
# TESTE DA IMPUTAÇÃO — versão em batches GPU-friendly  --------------
# ------------------------------------------------------------------
    @torch.no_grad()
    def test_impute(
        self,
        dataset_idx: int   = 0,
        missing_frac: float = 0.2,
        random_state: int  = 42,
        chunk_len:    int  = 4_096,
        sampling_steps: int = 40,
        range_: tuple = ()
    ):
        loader = Loader3W(); loader.load_stats("stats.pkl")
        df = loader.preprocess().send(None).sort_index()
        if range_:
            df = df[df.index >= range_[0]]
            df = df[df.index <= range_[1]]
        # -------- colunas coerentes --------
        base = list(self.default_features)
        state = sorted([c for c in df.columns if c.startswith("state-")])
        feature_cols = (base + state)[: self.in_channels]
        if len(feature_cols) != self.in_channels:
            raise ValueError("in_channels incompatível.")

        static_cols = [c for c in df.columns if c.endswith("_relative_max")]

        # -------- eixo temporal absoluto (segundos) --------
        t_sec = (pd.to_datetime(df.index).astype("int64") / 1e9)\
                .to_numpy(dtype=np.float32)               # shape (T,)
        T, C = len(t_sec), self.in_channels

        # -------- dados + máscara --------
        data_np = df[feature_cols].to_numpy(dtype=np.float32)
        rng = np.random.default_rng(random_state)

        miss_np = data_np.copy()
        idx_flat = rng.choice(miss_np.size,
                            int(miss_np.size * missing_frac),
                            replace=False)
        miss_np.reshape(-1)[idx_flat] = np.nan
        mask_miss = (~np.isnan(miss_np)).astype(np.float32)

        # -------- static feats --------
        if self.static_dim > 0 and static_cols:
            stat_vals = pd.to_numeric(df.iloc[0][static_cols], errors="coerce")\
                        .fillna(0.0).astype(np.float32).values
            stat_vals = np.pad(stat_vals,
                            (0, max(0, self.static_dim - len(stat_vals))),
                            constant_values=0.0)[: self.static_dim]
            stat = torch.tensor(stat_vals, device=self.beta.device)\
                    .unsqueeze(0)     # (1,D)
        else:
            stat = None

        # -------- acumulador de saída --------
        imp_full = np.empty_like(miss_np)

        # -------- processamento em chunks --------
        for start in range(0, T, chunk_len):
            end = min(start + chunk_len, T)
            slc = slice(start, end)

            # --- normaliza tempo local 0-1 ----
            t_chunk = t_sec[slc]
            span = t_chunk[-1] - t_chunk[0]
            span = span if span > 0 else 1.0
            t_norm = (t_chunk - t_chunk[0]) / span          # (L_chunk,)

            # --- tensores na GPU ----
            x_obs  = torch.tensor(
                        np.nan_to_num(miss_np[slc], nan=0.0)[None],
                        dtype=torch.float32, device=self.beta.device)
            m_t    = torch.tensor(mask_miss[slc][None],
                                dtype=torch.float32, device=self.beta.device)
            ts_t   = torch.tensor(t_norm[None],
                                dtype=torch.float32, device=self.beta.device)

            # --- imputação ----
            imp_chunk = self.impute(x_obs, m_t,
                                    timestamps=ts_t,
                                    static_feats=stat, sampling_steps=sampling_steps)[0]   # (L,C)
            imp_full[slc] = imp_chunk.cpu().numpy()

        # -------- organiza saída --------
        out = {feat: (t_sec,                 # eixo absoluto em segundos
                    df[feat].values,       # original
                    miss_np[:, j],         # com NaNs
                    imp_full[:, j])        # imputado (z-score)
            for j, feat in enumerate(feature_cols)}
        return out



if __name__ == '__main__':
    ld = Loader3W()
    ld.load_stats()
    ts_diffusion = TSDiffusion(
        in_channels=17,
        static_dim=7,
        hidden_dim=256,
        num_steps=1000,
        status_dim=10,
        )   
    '''ts_diffusion = ts_diffusion.load('state_ep5.pt',in_channels=17,
        latent_dim=170,
        model_dim=170*2,
        static_dim=7,
        hidden_dim=1024,
        num_steps=1000)'''
    ts_diffusion=ts_diffusion.to(torch.device('cuda'))
    ts_diffusion.train3W(
        60,
        batch_size=5000,
    )
    # 1) Extrai métricas nos **2 últimos** datasets
    '''metrics = ts_diffusion.evaluate_datasets(
        test_datasets=2,
        window_size=15,
        batch_size=256,
        missing_frac=0.3,     # igual ao seu test_impute
        sampling_steps=50
    )

    # 2) Plota
    ts_diffusion.plot_metrics_matplotlib(metrics)'''
    #print(ts_diffusion.test_sampler(prefix_len=15,future_len=5, window_size=15))
            
