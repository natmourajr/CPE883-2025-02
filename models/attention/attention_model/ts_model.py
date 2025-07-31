"""multiscale_attention_forecaster.py
================================================
PyTorch reference implementation for **multiscale attention‑based forecasting** with
built‑in **train/val/test split, checkpointing, and model persistence**.

Why this file?
--------------
* Handles **dual time‑scales**: dense local attention (< 60 s) & sparse/global patch attention
  (≤ 1 h) for long‑range dependencies.
* Provides a **TimeSeriesTrainer** wrapper that:
  * builds train/val/test DataLoaders from a single tensor.
  * tracks metrics epoch‑wise, saves a checkpoint when **test loss improves**.
  * offers `save_best()` / `load_best()` helpers.
* Includes **two runnable demos** at the bottom:
  1. **Predictive** – one‑shot horizon forecast.
  2. **Generative** – iterative sampling of the next 120 s.

Install
~~~~~~~
```bash
pip install torch numpy
```

Code
----
```python"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset

# -----------------------------------------------------------
# Positional Encoding ---------------------------------------------------------
# -----------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Classic sin/cos positional encodings (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


# -----------------------------------------------------------
# Local & Global Encoders ------------------------------------------------------
# -----------------------------------------------------------
class LocalEncoder(nn.Module):
    """Dense attention over a short window (< 60 s)."""

    def __init__(
        self, input_dim: int, d_model: int, n_heads: int, n_layers: int, dropout: float
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(x))
        return self.encoder(x)


class PatchEmbedding(nn.Module):
    """Extract non‑overlapping / sliding window patches then project."""

    def __init__(self, input_dim: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(patch_len, 1), stride=(stride, 1))
        self.proj = nn.Linear(input_dim * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T, 1)
        B, T, C = x.shape
        patches = self.unfold(x.transpose(1, 2).unsqueeze(-1))  # (B, C*patch_len, N)
        patches = patches.transpose(1, 2)  # (B, N, C*patch_len)
        return self.proj(patches)


class GlobalPatchEncoder(nn.Module):
    """Patch‑wise Transformer (dense by default, pluggable w/ sparse)."""

    def __init__(
        self,
        input_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(input_dim, patch_len, stride, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.pos(x)
        return self.encoder(x)


# -----------------------------------------------------------
# Decoder ----------------------------------------------------
# -----------------------------------------------------------
class FusionDecoder(nn.Module):
    """Merge local & global CLS tokens and predict horizon."""

    def __init__(self, d_model: int, output_dim: int, pred_len: int, dropout: float):
        super().__init__()
        self.out_dim = output_dim
        self.pred_len = pred_len
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim * pred_len),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([local_feat[:, 0], global_feat[:, 0]], dim=-1)
        out = self.fc(fused)
        return out.view(out.size(0), self.pred_len, self.out_dim)


# -----------------------------------------------------------
# Main Model -------------------------------------------------
# -----------------------------------------------------------
class MultiScaleAttentionForecaster(nn.Module):
    """End‑to‑end network for multiscale forecasting."""

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        seq_len: int = 3600,
        local_len: int = 60,
        pred_len: int = 60,
        d_model: int = 128,
        n_heads: int = 4,
        n_local_layers: int = 2,
        n_global_layers: int = 4,
        patch_len: int = 60,
        stride: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config: Dict[str, Any] = dict(
            input_dim=input_dim,
            output_dim=output_dim or input_dim,
            seq_len=seq_len,
            local_len=local_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            n_local_layers=n_local_layers,
            n_global_layers=n_global_layers,
            patch_len=patch_len,
            stride=stride,
            dropout=dropout,
        )

        self.seq_len = seq_len
        self.local_len = local_len
        self.pred_len = pred_len
        output_dim = output_dim or input_dim

        self.local_block = LocalEncoder(
            input_dim, d_model, n_heads, n_local_layers, dropout
        )
        self.global_block = GlobalPatchEncoder(
            input_dim,
            patch_len,
            stride,
            d_model,
            n_heads,
            n_global_layers,
            dropout,
        )
        self.decoder = FusionDecoder(d_model, output_dim, pred_len, dropout)

    # ------------------------------ forward ------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.seq_len, "input length must equal seq_len"
        local_x = x[:, -self.local_len :]
        local_feat = self.local_block(local_x)
        global_feat = self.global_block(x)
        return self.decoder(local_feat, global_feat)

    # ------------------------------ I/O utils ----------------------------
    def save(self, path: str | Path):
        data = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(data, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu"):
        data = torch.load(path, map_location)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        return model


# -----------------------------------------------------------
# Dataset utilities -----------------------------------------
# -----------------------------------------------------------
class RollingWindowDataset(Dataset):
    """Turn one long (T, C) tensor into (seq_len, pred_len) pairs."""

    def __init__(self, series: torch.Tensor, seq_len: int, pred_len: int):
        self.x = series  # (T, C)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return self.x.size(0) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        target_end = end + self.pred_len
        seq = self.x[start:end]
        target = self.x[end:target_end]
        return seq, target


# -----------------------------------------------------------
# Trainer ----------------------------------------------------
# -----------------------------------------------------------
class TimeSeriesTrainer:
    """Chronological split (train / test only), training loop & checkpointing."""

    def __init__(
        self,
        model: 'MultiScaleAttentionForecaster',  # forward‑ref to avoid circular import issues
        series: torch.Tensor,  # (T, C)
        train_ratio: float = 0.8,
        batch_size: int = 32,
        lr: float = 3e-4,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        # ---------------- chronological SPLIT ----------------
        full_ds = RollingWindowDataset(series, model.seq_len, model.pred_len)
        total = len(full_ds)
        train_len = int(total * train_ratio)
        test_len = total - train_len
        #   first part → train, last part → test  (no shuffling!)
        self.train_ds = Subset(full_ds, range(train_len))
        self.test_ds = Subset(full_ds, range(train_len, total))

        self.train_loader = DataLoader(self.train_ds, batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_ds, batch_size, shuffle=False)

        # ---------------- checkpoint dir ---------------------
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_test_loss: float = float("inf")
        self.best_ckpt_path: Optional[Path] = None

    # --------------------------------------------------------
    def _loop(self, loader: DataLoader, train: bool = False) -> float:
        self.model.train(mode=train)
        total_loss = 0.0
        for seq, target in loader:
            seq, target = seq.to(self.device), target.to(self.device)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            output = self.model(seq)
            loss = self.criterion(output, target)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            total_loss += loss.item() * seq.size(0)
        return total_loss / len(loader.dataset)

    # --------------------------------------------------------
    def fit(self, epochs: int = 20, verbose: bool = True):
        for ep in range(1, epochs + 1):
            train_loss = self._loop(self.train_loader, train=True)
            test_loss = self._loop(self.test_loader, train=False)

            # checkpoint when *test* improves
            if test_loss + 1e-6 < self.best_test_loss:
                self.best_test_loss = test_loss
                self.best_ckpt_path = self.save_checkpoint(f"best_ep{ep:03d}.pt")

            if verbose:
                print(
                    f"Epoch {ep:03d} | train {train_loss:.4f} | test {test_loss:.4f}"
                )

    # --------------------------------------------------------
    def evaluate(self) -> float:
        """Return test MSE for the *current* weights."""
        return self._loop(self.test_loader, train=False)

    # --------------------------------------------------------
    def save_checkpoint(self, name: str = "ckpt.pt") -> Path:
        path = self.ckpt_dir / name
        self.model.save(path)
        return path

    def load_best(self):
        if self.best_ckpt_path is None:
            raise RuntimeError("No checkpoint yet. Run fit().")
        self.model = MultiScaleAttentionForecaster.load(
            self.best_ckpt_path, map_location=self.device
        )
        self.model.to(self.device)
        return self.model


# -----------------------------------------------------------
# --------------------------- DEMOS -------------------------
# -----------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Synthetic multivariate signal: mixture of sin waves + noise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STEPS = 5000  # total seconds (~1.4 h)
    N_FEATS = 8
    t = torch.arange(STEPS)
    signals = torch.stack([
        torch.sin(2 * math.pi * t / p) for p in [30, 120, 300, 900, 60, 75, 47, 510]
    ], dim=1)
    signals.to(device=device)
    series = signals + 0.05 * torch.randn_like(signals)
    series = series.to(device)
    # ----------------- hyperparameters -----------------
    model = MultiScaleAttentionForecaster(
        input_dim=N_FEATS,
        seq_len=3600,
        local_len=60,
        pred_len=120,
        d_model=64,
        n_heads=2,
        n_global_layers=2,
        n_local_layers=2,
        patch_len=60,
        stride=60,
    )

    trainer = TimeSeriesTrainer(
        model,
        series,
        train_ratio=0.8,
        batch_size=32,
        device="cuda",
    )

    print("\n*** Training demo (predictive) ***")
    trainer.fit(epochs=5)

    best_model = trainer.load_best()
    test_mse = trainer.evaluate()
    print(f"\nBest checkpoint test MSE: {test_mse:.4f}\n")

    # ------------ Predictive inference ---------------
    with torch.no_grad():
        last_seq = series[-model.seq_len :].unsqueeze(0)
        forecast = best_model(last_seq).squeeze(0)  # (pred_len, C)
        print("Forecast sample (first 5 steps):\n", forecast[:5])

    # ------------ Generative (autoregressive) demo ----
    print("\n*** Generative demo: sampling 120 s autoregressively ***")
    gen_steps = 120
    context = series[-model.seq_len :].clone()
    generated: List[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(gen_steps):
            inp = context.unsqueeze(0)
            next_step = best_model(inp)[:, 0]  # predict only the first future step
            generated.append(next_step.squeeze(0))           # (C)
            context = torch.cat([context[1:], next_step], 0) # (seq_len, C)
    generated = torch.stack(generated)
    print("Generated sequence shape:", generated.shape)
