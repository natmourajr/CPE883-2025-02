"""
Time series KAN transform.

Summary: Apply the KAN transform algorithm in a inputed time series.

Considerations:
    - 


version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1]
"""
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import PatchEmbed
from typing import Union, Tuple

# Ajuste o caminho para a pasta 'kat' (onde está o katransformer.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'kat/')))
from katransformer import KATVisionTransformer

# PatchEmbed1D para série temporal
class PatchEmbed1D(nn.Module):
    def __init__(self, img_size=None, patch_size=16, embed_dim=128, **kwargs):
        super().__init__()
        if isinstance(img_size, tuple):
            seq_len = img_size[0]  # caso seja tupla, pega o primeiro
        else:
            seq_len = img_size or 128
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):  # x shape: (B, seq_len, 1)
        x = x.transpose(1, 2)  # (B, 1, seq_len)
        x = self.proj(x)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

# Subclasse para adaptar o KATVisionTransformer
class KATTimeSeriesTransformer(KATVisionTransformer):
    def __init__(self, seq_len, patch_size, embed_dim=128, depth=6, num_heads=4, num_classes=1):
        super().__init__(
            img_size=seq_len,
            patch_size=patch_size,
            in_chans=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            embed_layer=PatchEmbed1D,
        )

# --- Criar dados sintéticos ---

def create_sin_data(batch_size, seq_len, forecast_len):
    t = torch.linspace(0, 8 * 3.1416, seq_len + forecast_len)
    series = torch.sin(t).unsqueeze(0).repeat(batch_size, 1)
    noise = 0.5 * torch.randn_like(series)
    data = series + noise
    x = data[:, :seq_len].unsqueeze(-1)    # (batch_size, seq_len, 1)
    y = data[:, seq_len:].unsqueeze(-1)    # (batch_size, forecast_len, 1)
    # import ipdb
    # ipdb.set_trace()
    return x, y

def prepare_data(batch_size, seq_len, forecast_len):

    # TODO: Trocar pelos valores reais
    # Série de dados com dadas e valores.
    date_range = pd.date_range(start="2023-01-01", periods=seq_len + forecast_len, freq="D")
    values = pd.Series(50 + 10 * torch.randn(seq_len + forecast_len).numpy(), index=date_range)

    # Convertendo valores para tensor (normalmente valores já estão normalizados)
    data = torch.tensor(values.values, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)  # (batch_size, total_len)

    # Separando entrada (x) e futuro (y)
    x = data[:, :seq_len].unsqueeze(-1)        # (batch_size, seq_len, 1)
    y = data[:, seq_len:].unsqueeze(-1)        # (batch_size, forecast_len, 1)

    return x, y


def compute_mse_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Compute Mean Squared Error (MSE) between true and predicted tensors.
    
    Args:
        y_true (torch.Tensor): Ground truth values.
        y_pred (torch.Tensor): Predicted values.
    
    Returns:
        float: MSE value.
    """
    return torch.mean((y_true - y_pred) ** 2).item()

# --- Treinamento simples ---

seq_len = 128
patch_size = 16
forecast_len = 64
batch_size = 16
model = KATTimeSeriesTransformer(seq_len=seq_len, patch_size=patch_size, num_classes=forecast_len)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Using a sine series
x_train, y_train = create_sin_data(256, seq_len, forecast_len)
# Using a mock series with data/values
# x_train, y_train = prepare_data(256, seq_len, forecast_len)

# Standardize the data
mean = x_train.mean()
std = x_train.std()

x_train = (x_train - mean) / std
y_train = (y_train - mean) / std


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size,
    shuffle=True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.train()
for epoch in range(10):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)  # output shape: (B, forecast_len)
        loss = criterion(output, yb.squeeze(-1))  # ajustar para sequência
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}: Loss {total_loss/len(train_loader.dataset):.6f}")



# --- Avaliação e previsão direta de sequência ---
model.eval()
with torch.no_grad():
    t, y_test = create_sin_data(1, seq_len, forecast_len)
    
    # Standardize the data
    t = (t - mean) / std

    t = t.to(device)

    pred_seq = model(t)  # (1, forecast_len)

    pred_seq = pred_seq * std + mean
    pred_seq = pred_seq.squeeze(0).cpu().numpy()
    true_seq = y_test.squeeze(0).squeeze(-1).numpy()

    print("\nSequência prevista:")
    print(pred_seq)

    print("\nSequência real:")
    print(true_seq)

    mse = compute_mse_torch(torch.tensor(true_seq), torch.tensor(pred_seq))
    print(f"MSE: {mse:.6f}")


# Plot
fontsize=18
plt.rcParams.update({
    "font.size": fontsize,              # tamanho padrão
})
plt.figure(figsize=(12, 6))
plt.plot(true_seq, label='Real', marker='o')
plt.plot(pred_seq, label='Previsto', marker='x')
plt.title("Previsão da Série Temporal (TKAT)", fontsize=fontsize)
plt.xlabel("Passo de tempo", fontsize=fontsize)
plt.ylabel("Valor", fontsize=fontsize)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
