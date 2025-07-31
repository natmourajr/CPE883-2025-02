import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
import sys
import os

sys.path.append(f'{os.environ.get("path3W","../../../")}'+'3W')
from loader import Loader3W
from sklearn.model_selection import TimeSeriesSplit


class Time2Vec(nn.Module):
    """Senoidal positional encoding."""
    def __init__(self, dim: int):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w  = nn.Parameter(torch.randn(dim - 1))
        self.b  = nn.Parameter(torch.randn(dim - 1))

    def forward(self, timestamps: torch.Tensor, device = None) -> torch.Tensor:
        timestamps = timestamps.to(device)
        pos = (timestamps - timestamps.min(dim=1, keepdim=True)[0]) / (
            timestamps.max(dim=1, keepdim=True)[0] - timestamps.min(dim=1, keepdim=True)[0] + 1e-8
        )
        v0 = self.w0 * pos + self.b0
        vp = torch.sin(pos.unsqueeze(-1) * self.w + self.b)
        if device is not None:
            v0 = v0.to(device)
            vp = vp.to(device)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)

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
    return betas.clamp(max=0.999)  # evita valores muito altos

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
        self.pos_enc = Time2Vec(pos_dim)
        self.pos_proj = nn.Linear(pos_dim, model_dim)
        self.static_dim = static_dim
        self.t_embed = DiffTimeEmbedding(model_dim)
        if static_dim > 0:
            self.static_proj = nn.Sequential(
                nn.Linear(static_dim, model_dim),
                nn.ReLU()
            )
        self.output_proj = nn.Linear(model_dim, self.latent_dim)
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
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
        already_latent: bool=False
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
        if hasattr(self, 'encoder') and not already_latent:
            x = self.encoder(x)
        # Embedding de entrada
        h = self.input_proj(x)
        # Positional encoding via Time2Vec
        if timestamps is not None:
            pe = self.pos_enc(timestamps, device=device)  # (b, seq_len, pos_dim)
            h = h + self.pos_proj(pe)


        te = self.t_embed(t).unsqueeze(1)          # (b,1,model_dim)
        h = h + te
        # Static features
        if static_feats is not None and self.static_dim > 0:
            se = self.static_proj(static_feats).unsqueeze(1)  # (b,1,model_dim)
            h = h + se
        # Transformer
        h = h.permute(1, 0, 2)  # (seq_len, b, model_dim)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)  # (b, seq_len, model_dim)
        h = self.output_proj(h)
        # 8) decoder latente → original
        if hasattr(self, 'decoder') and not already_latent:
            h = self.decoder(h)
        return h

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        timestamps: torch.Tensor = None,
        static_feats: torch.Tensor = None,
        sampling_steps: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Gera amostras via reverse diffusion, opcionalmente condicionando em timestamps e static_feats.
        """
        device = device or next(self.parameters()).device
        steps = min(sampling_steps or self.num_steps, self.num_steps)
        x = torch.randn(batch_size, seq_len, self.latent_dim, device=device)
        for i in reversed(range(steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            eps = self.forward(x, t, timestamps=timestamps, static_feats=static_feats, already_latent=True)
            a, ab = self.alpha[i], self.alpha_bar[i]
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(a)) * (x - ((1 - a) / torch.sqrt(1 - ab)) * eps) + torch.sqrt(self.beta[i]) * noise
        # decodifica para o espaço original
        if hasattr(self, 'decoder'):
            x = self.decoder(x)
        return x

    @torch.no_grad()
    def impute(self, x_obs, mask, timestamps=None, static_feats=None,
           sampling_steps=None, device=None):
        device = device or x_obs.device
        steps = min(sampling_steps or self.num_steps, self.num_steps)

        # ------------- AJUSTE DA MÁSCARA -------------
        if hasattr(self, 'encoder'):
            if self.latent_dim == self.in_channels:
                mask_latent = mask           # shape (B,T,C)
            else:
                # mascara “verdadeira” se **qualquer** canal de entrada estiver presente
                # resultado: (B,T,1) → expandido para (B,T,latent_dim)
                mask_latent = mask.any(dim=-1, keepdim=True)\
                                .float().expand(-1, -1, self.latent_dim)
        else:
            mask_latent = mask
        # ---------------------------------------------

        # codifica somente as observações
        z_obs = self.encoder(x_obs) if hasattr(self, 'encoder') else x_obs
        noise = torch.randn_like(z_obs)
        z     = z_obs * mask_latent + noise * (1 - mask_latent)

        b = z.size(0)
        for i in reversed(range(steps)):
            t   = torch.full((b,), i, device=device, dtype=torch.long)
            eps = self.forward(z, t, timestamps=timestamps, static_feats=static_feats, already_latent=True)
            a, ab = self.alpha[i], self.alpha_bar[i]
            z     = (1 / torch.sqrt(a)) * (z - ((1 - a) / torch.sqrt(1 - ab)) * eps)
            if i > 0:
                z += torch.sqrt(self.beta[i]) * torch.randn_like(z)
            # reinsere valores observados
            z = z * (1 - mask_latent) + z_obs * mask_latent

        return self.decoder(z) if hasattr(self, 'decoder') else z

    def train3W(
            self, 
            window_size: int = 600, 
            feature_cols: list = default_features + [f'state-{s}' for s in range(10)], 
            static_features_cols: list = [f'{f}_relative_max' for f in default_features], 
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-3,
            test_datasets: int = 2,
            ):
        loader = Loader3W()
        loader.load_stats('stats.pkl')
        for i in range(1, epochs+1):
            test = pd.DataFrame()
            datasets = loader.preprocess()
            for num_dataset, dataset in enumerate(datasets):
                if num_dataset < len(loader.stats['ids']) - test_datasets:
                    print(f'Starting epoch {i}/{epochs} - dataset {num_dataset+1}/{len(loader.stats["ids"])} - Partial Validation Loss: {self.val_loss:.6f}' )
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
        test_loader = DataLoader(test_ds, batch_size=1)
        self.to(device).eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, ts_batch = batch[0].to(device), batch[1].to(device)
                s = batch[2].to(device) if len(batch) > 2 else None
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
        ts = pd.to_datetime(df[timestamp_col] if timestamp_col != 'index' else df.index).astype('int64') / 1e9
        data = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        times = torch.tensor(ts.values, dtype=torch.float32)
        static = torch.tensor(df[static_features_cols].values, dtype=torch.float32) if static_features_cols else None
        if window_size is None or window_size >= len(df):
            seqs = data.unsqueeze(0)
            ts_seqs = times.unsqueeze(0)
            stat_seqs = static[0].unsqueeze(0) if static is not None else None
        else:
            n_ws = len(df) - window_size + 1
            seqs = torch.stack([data[i:i+window_size] for i in range(n_ws)])
            ts_seqs = torch.stack([times[i:i+window_size] for i in range(n_ws)])
            stat_seqs = static[0].unsqueeze(0).repeat(n_ws, 1)  if static is not None else None
        if stat_seqs is None:
            return TensorDataset(seqs, ts_seqs)
        return TensorDataset(seqs, ts_seqs, stat_seqs)
    
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


        train_ds = self._make_dataset(df_train, timestamp_col, window_size, feature_cols, static_features_cols)
        val_ds = self._make_dataset(df_val, timestamp_col, window_size, feature_cols, static_features_cols)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device).train()
        for epoch in range(1, epochs + 1):
            total_train = 0.0
            self.train()
            for batch in train_loader:
                x, ts_batch = batch[0].to(device), batch[1].to(device)
                s = batch[2].to(device) if len(batch) > 2 else None
                t = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                noise = torch.randn_like(x)
                ab = self.alpha_bar[t].view(-1, 1, 1)
                x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                eps_pred = self.forward(x_t, t, timestamps=ts_batch, static_feats=s)
                loss = F.mse_loss(eps_pred, noise)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_train += loss.item() * x.size(0)
            total_val = 0.0
            self.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x, ts_batch = batch[0].to(device), batch[1].to(device)
                    s = batch[2].to(device) if len(batch) > 2 else None
                    t = torch.randint(0, self.num_steps, (x.size(0),), device=device)
                    noise = torch.randn_like(x)
                    ab = self.alpha_bar[t].view(-1, 1, 1)
                    x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * noise
                    eps_pred = self.forward(x_t, t, timestamps=ts_batch, static_feats=s)
                    total_val += F.mse_loss(eps_pred, noise).item() * x.size(0)
            if verbose:
                print(f"Epoch {epoch}/{epochs} — Train Loss: {total_train/len(train_ds):.6f} — Val Loss: {total_val/len(val_ds):.6f}")
            else:
                self.loss = total_train / len(train_ds)
                self.val_loss = total_val / len(val_ds)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, *args, **kwargs):
        model = cls(*args, **kwargs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        return model


            
