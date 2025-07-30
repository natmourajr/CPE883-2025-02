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

sys.path.append(f'{os.environ.get("path3W","../../../")}'+'3W')
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
    def __init__(self, in_dim, hidden_dim, attn_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(in_dim, hidden_dim)
        self.odefunc = ODEFunc(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, attn_heads, batch_first=True)
        
    def forward(self, x, ts, mask=None):
        """
        x  : (B, T, C)       – valores (já no espaço latente se houver encoder linear)
        ts : (B, T)          – segundos unix
        mask: (B, T, 1) bool – True se valor presente (p/ atenção); None → tudo presente
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        states = []
        
        for i in range(T):
            if i > 0:                         # integra de t_{i-1} → t_i

                dt_i = (ts[:, i] - ts[:, i-1]).float().unsqueeze(-1)  # (B,1)
                h = h + dt_i * self.odefunc(None, h)   
            # jump ‑ atualiza estado com valor observado
            h = self.gru(x[:, i], h)
            states.append(h)
        
        H = torch.stack(states, dim=1)        # (B, T, hidden_dim)
        if mask is not None:
            # (B,T,C) → (B,T)   True se ao menos um canal está presente
            m_time = mask.to(torch.bool).any(dim=2)            # (B,T)
            key_pad = ~m_time                                  # True = IGNORAR

            H, _ = self.attn(H, H, H, key_padding_mask=key_pad)  # <- 2‑D 👍
        else:
            H, _ = self.attn(H, H, H)
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
        latent_dim: int = None,
        model_dim: int = 64,
        hidden_dim: int = 128,
        num_steps: int = 1000,
        n_heads: int = 4,
        n_layers: int = 4,
        pos_dim: int = 16,
        static_dim: int = 0
    ):
        super().__init__()
        self.val_loss = float('inf')
        self.model_dim = model_dim
        self.num_steps = num_steps
        self.latent_dim = latent_dim or in_channels
        self.in_channels = in_channels
        if latent_dim is not None:
            self.encoder = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_channels),
            )
        # Projeções
        self.input_proj = nn.Linear(self.latent_dim, model_dim)
        self.pos_enc = Time2Vec()
        self.pos_proj = nn.Linear(4, model_dim)
        self.t_embed = DiffTimeEmbedding(model_dim)
        self.static_dim = static_dim
        head_dim = model_dim
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, model_dim),
                nn.ReLU()
            )
        # (a) λ(t)  — intensity do ponto de observação
        self.lambda_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, 1)       # escalar
        )
        # (b) μ_x(t)  — média gaussiana para L1
        self.mean_head = nn.Linear(head_dim, self.latent_dim)
        # (c) μ_Tmax  — previsão do horizonte da série
        self.tmax_head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Linear(head_dim // 2, 1),
            nn.Sigmoid()        # range 0‑1
        )
        self.output_proj = nn.Linear(model_dim, self.latent_dim)
        self.encoder_ode = JumpODEEncoder(model_dim, model_dim, attn_heads=n_heads)
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
        return_state: bool=False,
        mask = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_channels) - dados ruidosos.
            t: (batch,) - passos de difusão.
            timestamps: (batch, seq_len) - colunas de tempo.
            static_feats: (batch, static_dim).
        """
        b, seq_len, _ = x.shape
        device = x.device
        # Embedding de entrada
        if hasattr(self, 'encoder') and not already_latent:
            x = self.encoder(x)
        h = self.input_proj(x)
        # Positional encoding via Time2Vec
        if timestamps is None:
            raise ValueError("timestamps são obrigatórios para Jump‑ODE Encoder")
        h = self.encoder_ode(h, timestamps, mask=mask)   # (B,T,model_dim)
        te = self.t_embed(t).unsqueeze(1)          # (b,1,model_dim)
        h = h + te
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        # Transformer
        #h = h.permute(1, 0, 2)  # (seq_len, b, model_dim)
        #h = self.transformer(h)
        #h = h.permute(1, 0, 2)  # (b, seq_len, model_dim)
        # Previsão de ruído
        state = h
        eps_pred = self.output_proj(h)
        if hasattr(self,'decoder') and not already_latent:
            eps_pred = self.decoder(eps_pred)
        if return_state:
            return eps_pred, state, x
        return eps_pred

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        Tmax_scale: float = 1.1,        # δ para T̂_max
        lam_pad: float = 1.2,           # cota adaptativa de λ_max
        sampling_steps: int = None,
        device: torch.device = None
    ):
        device = device or next(self.parameters()).device

        # 1) Reverse‑diffusion para obter z₀
        steps = min(sampling_steps or self.num_steps, self.num_steps)
        z = torch.randn(batch_size, 1, self.latent_dim, device=device)
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            eps = self.forward(z, t, timestamps=None,
                               static_feats=None,
                               already_latent=True, return_state=False)
            a, ab = self.alpha[i], self.alpha_bar[i]
            noise = torch.randn_like(z) if i > 0 else torch.zeros_like(z)
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab))*eps) \
                + torch.sqrt(self.beta[i]) * noise

        # 2) Projeta z₀ para o espaço de estado contínuo (model_dim)
        h = self.input_proj(z.squeeze(1))   # shape (B, model_dim)

        # 3) Thinning
        samples = []
        for b in range(batch_size):
            t_cur = 0.0
            hb = h[b]  # estado inicial deste exemplo, shape (model_dim,)
            Tmax_pred = self.tmax_head(hb.unsqueeze(0)).item() * Tmax_scale

            # estima λ_max inicial
            lam0 = F.softplus(self.lambda_head(hb.unsqueeze(0))).item()
            lam_max = lam0 * lam_pad + 1e-4

            times, values = [], []
            while t_cur < Tmax_pred:
                w = torch.empty(1, device=device).exponential_(lam_max)
                dt = w.item()
                t_cur += dt
                if t_cur >= Tmax_pred:
                    break

                # --------- dinâmica contínua ---------
                hb0 = hb.unsqueeze(0)                                    # (1, model_dim)
                k1  = self.encoder_ode.odefunc(None, hb0).squeeze(0)     # f(h)
                hb  = hb + w * k1                                        # Euler

                # --------- cabeças λ(t) e μ_x(t) ---------
                lam_t = F.softplus(self.lambda_head(hb.unsqueeze(0))).item()
                mu_x  = self.mean_head(hb.unsqueeze(0)).squeeze(0)

                # 3.4 Thinning accept/reject
                if torch.rand(1, device=device).item() < lam_t / lam_max:
                    xi = mu_x + torch.randn_like(mu_x)
                    times.append(t_cur)
                    values.append(xi)

                # 3.5 Ajuste adaptativo de λ_max se necessário
                if lam_t > lam_max:
                    lam_max = lam_t * lam_pad

            # 4) Decodifica valores de volta ao espaço original
            if hasattr(self, 'decoder'):
                if values:
                    vals = torch.stack(values)
                else:
                    vals = torch.empty(0, self.latent_dim, device=device)
                vals = self.decoder(vals)
            else:
                vals = torch.stack(values) if values else torch.empty(0, self.latent_dim, device=device)

            samples.append((torch.tensor(times, device=device), vals))

        return samples


    @torch.no_grad()
    def impute(
        self,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        sampling_steps: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        device = device or x_obs.device
        steps = min(sampling_steps or self.num_steps, self.num_steps)

        # 1) Ajusta máscara para latente
        if hasattr(self, 'encoder'):
            if self.latent_dim == self.in_channels:
                mask_latent = mask  # (B,T,C)
            else:
                mask_latent = (
                    mask.any(dim=-1, keepdim=True)
                        .float()
                        .repeat(-1, -1, self.latent_dim)
                )
        else:
            mask_latent = mask

        # 2) Codifica observados + adiciona ruído inicial
        z_obs = self.encoder(x_obs) if hasattr(self, 'encoder') else x_obs
        noise = torch.randn_like(z_obs)
        z = z_obs * mask_latent + noise * (1 - mask_latent)

        # 3) Reverse‑diffusion condicionado
        B = z.size(0)
        for i in reversed(range(steps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            eps = self.forward(
                z, t,
                timestamps=timestamps,
                static_feats=static_feats,
                already_latent=True,
                return_state=False,
                mask=mask  # propaga máscara na atenção
            )
            a, ab = self.alpha[i], self.alpha_bar[i]
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab)) * eps)
            if i > 0:
                z = z + torch.sqrt(self.beta[i]) * torch.randn_like(z)

            # 3.1 Reinsere as observações originais
            z = z * (1 - mask_latent) + z_obs * mask_latent

        # 4) Decodifica para o espaço original
        return self.decoder(z) if hasattr(self, 'decoder') else z
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
    def train3W(
            self, 
            window_size: int = 600, 
            feature_cols: list = default_features + [f'state-{s}' for s in range(10)], 
            static_features_cols: list = [f'{f}_relative_max' for f in default_features], 
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-3,
            test_datasets: int = 2,
            validate: bool = True
            ):
        loader = Loader3W()
        loader.load_stats('stats.pkl')
        for i in range(1, epochs+1):
            test = pd.DataFrame()
            datasets = loader.preprocess()
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
                static_features_cols=static_features_cols,
                timestamp_col='index',
                window_size=window_size,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            print(f'Epoch {i}/{epochs} completed - Test Loss: {test_loss:.6f}')

    def test_model(
        self,
        df_test: pd.DataFrame,
        feature_cols: list,
        static_features_cols: list,
        timestamp_col: str,
        window_size: int = None,
        device: torch.device = None
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_ds = self._make_dataset(df_test, timestamp_col, window_size, feature_cols, static_features_cols)
        test_loader = DataLoader(test_ds, batch_size=200)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 4:
                    x, ts_batch, m, s = batch
                else:                               # caso não haja static
                    x, ts_batch, m = batch;  s = None
                x, ts_batch, m = x.to(device), ts_batch.to(device), m.to(device)
                if s is not None: s = s.to(device)
                t = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                ab = self.alpha_bar[t].view(-1, 1, 1)
                x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                eps_pred = self.forward(x_t, t, timestamps=ts_batch, static_feats=s)
                loss = F.mse_loss(eps_pred, noise)
                total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(test_ds)
        return avg_loss    
    
    @staticmethod
    def _make_dataset(df, timestamp_col, window_size, feature_cols, static_features_cols):
        if timestamp_col != 'index':
            df = df.sort_values(timestamp_col).reset_index(drop=True)
# ---------------- NORMALIZAÇÃO DO TEMPO ----------------
        if timestamp_col != "index":
            ts_raw = pd.to_datetime(df[timestamp_col]).astype("int64") / 1e9
        else:
            ts_raw = pd.to_datetime(df.index).astype("int64") / 1e9

        ts_np = ts_raw.to_numpy(dtype=np.float32)
        ts_rel = ts_np - ts_np[0]                   # começa em 0
        span = ts_rel[-1] if ts_rel[-1] > 0 else 1  # evita div/0
        ts_rel = ts_rel / span                      # agora 0‑1

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
        else:
            n_ws = len(df) - window_size + 1
            seqs = torch.stack([data[i:i+window_size] for i in range(n_ws)])
            ts_seqs = torch.stack([times[i:i+window_size] for i in range(n_ws)])
            mask_seqs = torch.stack([mask[i:i+window_size] for i in range(n_ws)])
            stat_seqs = static[0].unsqueeze(0).repeat(n_ws, 1)  if static is not None else None
        if stat_seqs is None:
            return TensorDataset(seqs, ts_seqs, mask_seqs)                   # 3 itens
        return TensorDataset(seqs, ts_seqs, mask_seqs, stat_seqs)   
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
        timestamp_col: str,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        window_size: int = None,
        device: torch.device = None,
        verbose: bool = True
    ):
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        lam = [0.001, 0.4, 0.05, 0.05]
        train_ds = self._make_dataset(df_train, timestamp_col, window_size, feature_cols, static_features_cols)
        val_ds = self._make_dataset(df_val, timestamp_col, window_size, feature_cols, static_features_cols)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if df_val is not None else None
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=(0.9,0.999),
                                      weight_decay=1e-2)
        self.to(device).train()
        for epoch in range(1, epochs + 1):
            total_train = 0.0
            self.train()
            for batch in train_loader:
                if len(batch) == 4:
                    x, ts_batch, m, s = batch
                else:                               # caso não haja static
                    x, ts_batch, m = batch;  s = None
                x, ts_batch, m = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True)
                if s is not None: s = s.to(device, non_blocking = True)
                t = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                ab = self.alpha_bar[t].view(-1, 1, 1)
                x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                eps_pred, state, x_lat  = self.forward(x_t, t, timestamps=ts_batch, static_feats=s, return_state=True, mask=m)
                # ---------- cabeças ----------
                lam_t = F.softplus(self.lambda_head(state)).clamp(1e-3, 10.0)
                mu_x  = self.mean_head(state)                          # (B,T,latent_dim)

                lam2 = lam_t.squeeze(-1).clamp_(min=1e-3, max=50)      # in‑place ✔
                log_px = -0.5 * ((x_lat - mu_x)**2).sum(-1, keepdim=True)
                                    # (B,T)
                ts  = ts_batch                                         # (B,T)  já em segundos                
                # 1. parte observada – regra do trapézio
                dt          = ts[:, 1:] - ts[:, :-1]                   # (B,T‑1)
                int_obs     = 0.5 * (lam2[:, :-1] + lam2[:, 1:]) * dt    # (B,T‑1)
                integral_obs = int_obs.sum(-1)                         # (B,)

                # 2. cauda até T̂_max (μ_Tmax predito)
                mu_Tmax = self.tmax_head(state[:, -1]).squeeze(-1)     # (B,)
                tail_dt = torch.relu(mu_Tmax - ts[:, -1])              # (B,)
                integral_tail = lam2[:, -1] * tail_dt                   # (B,)

                integral_total = integral_obs + integral_tail          # (B,)

                # ---------- Loss L1 completa ----------
                # log_px já contém -0.5*sum(...)
                log_event = (log_px.squeeze(-1) + torch.log(lam2 + 1e-8)).sum(-1)  # soma sobre T
                # se quiser reduzir por seq_len use .mean(-1)

                L1 = -(log_event - integral_total).mean()
                
                # Perda do ruído
                L2 = F.mse_loss(eps_pred, noise)
                # ---------- L3  (horizonte máximo) ----------
                tN      = ts_batch[:, -1].unsqueeze(-1)   # (B,1)
                mu_Tmax = self.tmax_head(state[:, -1])
                L3 = ((tN*1.1 - mu_Tmax).pow(2)).mean()
                # ----- L4 (máscara) -----
                # máscara binária: 1 se ao menos um canal está presente no timestep
                m_t = m.any(dim=2, keepdim=True).float()        # (B, T, 1)

                mb_pred = torch.sigmoid(self.miss_head(state)).clamp(1e-4, 1-1e-4)  # (B, T, 1)

                ce = m_t * torch.log(mb_pred + 1e-8) + \
                    (1 - m_t) * torch.log(1 - mb_pred + 1e-8)
                L4 = -ce.mean()

                loss = lam[0]*L1 + lam[1]*L2 + lam[2]*L3 + lam[3]*L4
                #loss = lam[1]*L2 + lam[3]*L4
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_train += loss.item() * x.size(0)
            if df_val is not None:
                total_val = 0.0
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 4:
                            x, ts_batch, m, s = batch
                        else:                               # caso não haja static
                            x, ts_batch, m = batch;  s = None
                        x, ts_batch, m = x.to(device, non_blocking = True), ts_batch.to(device, non_blocking = True), m.to(device, non_blocking = True)
                        if s is not None: s = s.to(device, non_blocking = True)
                        t = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                        noise = torch.randn_like(x)
                        ab = self.alpha_bar[t].view(-1, 1, 1)
                        x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                        eps_pred = self.forward(x_t, t, timestamps=ts_batch, static_feats=s, mask=m)
                        total_val += F.mse_loss(eps_pred, noise).item() * x.size(0)
                if verbose:
                    print(f"Epoch {epoch}/{epochs} — Train Loss: {total_train/len(train_ds):.6f} — Val Loss: {total_val/len(val_ds):.6f}")
                else:
                    self.loss = total_train / len(train_ds)
                    self.val_loss = total_val / len(val_ds)
        print(f"Ep {epoch}: L1={L1.item():.2f} L2={L2.item():.2f} "
    f"L3={L3.item():.2f} L4={L4.item():.2f}")

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model

    @torch.no_grad()
    def sample_regular(
        self,
        batch_size: int = 1,
        seq_len:   int = 15,
        delta_t:   float = 1.0,
        static_feats: torch.Tensor = None,
        sampling_steps: int = None,
        device: torch.device = None,
    ):
        """
        Gera (times, values) — ambos em espaço **normalizado** (z‑score).
        """
        device = device or next(self.parameters()).device
        steps  = min(sampling_steps or self.num_steps, self.num_steps)

        # --- dentro de sample_regular ----------------------------------------
        ts_grid = (
            torch.arange(seq_len, device=device, dtype=torch.float32) * delta_t
        ).unsqueeze(0).repeat(batch_size, 1)

        # normaliza: 0‑1 exactamente como em _make_dataset()
        ts_grid = ts_grid / ts_grid[:, -1:].clamp(min=1.0)

        # estado inicial (ruído)
        z = torch.randn(batch_size, seq_len, self.latent_dim, device=device)

        if static_feats is not None:
            static_feats = static_feats.to(device=device, dtype=torch.float32)

        # reverse‑diffusion
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            eps = self.forward(
                z, t, timestamps=ts_grid,
                static_feats=static_feats,
                already_latent=True
            )
            a, ab = self.alpha[i], self.alpha_bar[i]
            noise = torch.randn_like(z) if i > 0 else torch.zeros_like(z)
            z = (1/torch.sqrt(a)) * (z - ((1-a)/torch.sqrt(1-ab))*eps) \
                + torch.sqrt(self.beta[i]) * noise

        if hasattr(self, "decoder"):
            z = torch.nn.functional.layer_norm(
                z, normalized_shape=(self.latent_dim,)
            )                      # evita “explosão” inicial
            z = self.decoder(z)      # continua em z‑score

        return [(ts_grid[b].cpu().numpy(), z[b].cpu().numpy())
                for b in range(batch_size)]



    # ------------------------------------------------------------------
    # utilitário: escolhe janela com dados suficientes -----------------
    @staticmethod
    def _pick_window(df: pd.DataFrame,
                    feature_cols: list,
                    seq_len: int,
                    min_valid: float = 0.66):
        """
        Devolve (df_win, start_idx).
        Garante que cada feature tenha ≥ min_valid*seq_len valores não‑nulos.
        Se não existir tal janela, devolve a primeira (com NaNs mesmos).
        """
        thresh = int(seq_len * min_valid)

        for start in range(0, len(df) - seq_len + 1):
            win = df.iloc[start:start + seq_len][feature_cols]
            ok  = (win.notna().sum(axis=0) >= thresh).all()
            if ok:
                return df.iloc[start:start + seq_len], start

        # fallback: primeira janela
        return df.iloc[:seq_len], 0
    # ------------------------------------------------------------------

# --------------------------------------------------------------------------
# TEST_SAMPLER NOVO  -------------------------------------------------------
# --------------------------------------------------------------------------
@torch.no_grad()
def test_sampler(
    self,
    dataset_idx: int           = 0,
    feature_cols: list         = None,
    static_features_cols: list = None,
    prefix_len: int            = 48,     # pontos conhecidos
    future_len: int            = 24,     # pontos a gerar
    delta_t: float             = 60.0,   # mesmo passo usado no treino
    min_valid: float           = 0.66,   # % de valores não‑nulos em cada col.
):
    """
    Retorna {feature: (real_t, real_val, gen_t, gen_val)} em **z‑score**.
    """
    feature_cols = feature_cols or self.default_features
    static_features_cols = static_features_cols or \
        [f"{f}_relative_max" for f in self.default_features]

    # --------- carrega dataset -------------------------------------------
    loader = Loader3W(); loader.load_stats("stats.pkl")
    datasets = loader.preprocess()
    if dataset_idx >= len(datasets):
        raise IndexError("dataset_idx fora do intervalo disponível.")
    df = datasets[dataset_idx].sort_index()

    # --------- escolhe janela com dados suficientes ----------------------
    df_win, _ = self._pick_window(df, feature_cols, prefix_len, min_valid)

    # --------- prepara tensores do prefixo -------------------------------
    ts_prefix = pd.to_datetime(df_win.index)
    t_secs = (ts_prefix.astype("int64") / 1e9).to_numpy(dtype=np.float32)
    t_secs -= t_secs[0]                       # começa em zero
    xs_np = df_win[feature_cols].to_numpy(dtype=np.float32)
    mask_np = ~np.isnan(xs_np);  xs_np[np.isnan(xs_np)] = 0.0

    t_t = torch.tensor(t_secs[None], device=self.beta.device)          # (1,L0)
    x_t = torch.tensor(xs_np[None],  device=self.beta.device)          # (1,L0,C)

    # --------- static feats ---------------------------------------------
    stat = None
    if static_features_cols and self.static_dim > 0:
        stat_np = (
            pd.to_numeric(df_win.iloc[0][static_features_cols], errors="coerce")
              .fillna(0.).to_numpy(dtype=np.float32)
        )
        stat = torch.tensor(stat_np).unsqueeze(0).to(self.beta.device)  # (1,D)

    # --------- continua a série -----------------------------------------
    gen_t, gen_vals = self.sample_continue(
        x_prefix=x_t,
        ts_prefix=t_t,
        n_future=future_len,
        delta_t=delta_t,
        static_feats=stat,
    )
    gen_t, gen_vals = gen_t[0], gen_vals[0]        # remove batch dim

    # --------- organiza saída -------------------------------------------
    out = {}
    for j, feat in enumerate(feature_cols):
        out[feat] = (
            t_secs,                   # prefixo t
            df_win[feat].values,      # prefixo y (com NaNs)
            gen_t,                    # eixo completo gerado
            gen_vals[:, j],           # valores gerados
        )
    return out



    # ---------------------------------------------------------------------
    # 3)  TESTE DA IMPUTAÇÃO  --------------------------------------------
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def test_impute(
        self,
        dataset_idx: int              = 0,
        feature_cols: list            = None,
        static_features_cols: list    = None,
        missing_frac: float           = 0.2,
        random_state: int             = 42,
    ):
        """
        Retorna dicionário {feature: (times, orig, miss, imp)}
        """
        loader = Loader3W(); loader.load_stats("stats.pkl")
        df = loader.preprocess()[dataset_idx].sort_index()

        feature_cols = feature_cols or self.default_features
        static_features_cols = static_features_cols or \
                            [f"{f}_relative_max" for f in self.default_features]

        # ---------- série & timestamps ----------
        times_ts = pd.to_datetime(df.index)
        times = (
            (times_ts.astype("int64") / 1e9) - (times_ts[0].value / 1e9)
        )
        data_np = df[feature_cols].values.astype(np.float32)
        mask_np = (~np.isnan(data_np)).astype(np.float32)
        data_np[np.isnan(data_np)] = 0.0

        # ---------- injeta faltantes ----------
        rng = np.random.default_rng(random_state)
        miss_np = data_np.copy()
        T, C = miss_np.shape
        idx_flat = rng.choice(T * C, int(T * C * missing_frac), replace=False)
        miss_np.reshape(-1)[idx_flat] = np.nan
        mask_miss = (~np.isnan(miss_np)).astype(np.float32)
        miss_np[np.isnan(miss_np)] = 0.0

        # ---------- tensores ----------
        x_obs  = torch.tensor(miss_np[None], dtype=torch.float32, device=self.beta.device)
        mask_t = torch.tensor(mask_miss[None], dtype=torch.float32, device=self.beta.device)
        ts_t   = torch.tensor(times[None], dtype=torch.float32, device=self.beta.device)
        stat   = torch.tensor(
            df.iloc[0][static_features_cols].values,
            dtype=torch.float32, device=self.beta.device
        ).unsqueeze(0)

        # ---------- imputação ----------
        imputed = self.impute(x_obs, mask_t, timestamps=ts_t, static_feats=stat)
        imp_np = imputed[0].cpu().numpy()

        # ---------- organiza saída ----------
        out = {}
        for j, feat in enumerate(feature_cols):
            orig_col = df[feat].values
            miss_col = miss_np[:, j]
            imp_col  = imp_np[:, j]
            out[feat] = (times.values, orig_col, miss_col, imp_col)
        return out



            
