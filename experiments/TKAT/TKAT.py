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


import torch
import torch.nn as nn
import numpy as np
from katransformer import KATransformer  # from https://github.com/Adamdad/kat

# --- 1. Simulate toy time series ---
batch_size = 16
past_seq_len = 128
forecast_len = 200
patch_size = 16

# Make synthetic training data: sine waves + noise
t = np.linspace(0, 4 * np.pi, past_seq_len + forecast_len)
series = np.sin(t)[None, :] + 0.1 * np.random.randn(batch_size, past_seq_len + forecast_len)

x = torch.tensor(series[:, :past_seq_len], dtype=torch.float32)      # Input: past
y = torch.tensor(series[:, past_seq_len:], dtype=torch.float32)      # Target: future

# --- 2. Patchify input ---
def patchify(x, patch_size):
    B, L = x.shape
    newL = L // patch_size
    x = x[:, :newL * patch_size]
    return x.view(B, newL, patch_size)

x_patches = patchify(x, patch_size)  # shape: (B, num_patches, patch_size)

# --- 3. Define modified KAT model ---
class TimeSeriesKAT(nn.Module):
    def __init__(self, patch_size, num_patches, forecast_len, embed_dim=64):
        super().__init__()
        self.kat = KATransformer(
            seq_len=patch_size,
            num_layers=4,
            num_heads=4,
            embed_dim=embed_dim,
        )
        self.head = nn.Sequential(
            nn.Flatten(),  # shape: (B, num_patches * embed_dim)
            nn.Linear(num_patches * embed_dim, forecast_len)
        )

    def forward(self, x):
        x = self.kat(x)          # (B, num_patches, embed_dim)
        x = self.head(x)         # (B, forecast_len)
        return x

model = TimeSeriesKAT(
    patch_size=patch_size,
    num_patches=past_seq_len // patch_size,
    forecast_len=forecast_len,
    embed_dim=64
).to('cuda' if torch.cuda.is_available() else 'cpu')

# --- 4. Training loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
device = model.head[1].weight.device

for epoch in range(20):
    model.train()
    pred = model(x_patches.to(device))  # shape: (B, 200)
    loss = loss_fn(pred, y.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# --- 5. Forecast on new input ---
model.eval()
with torch.no_grad():
    future = model(x_patches[:1].to(device))  # shape: (1, 200)
    print("Forecasted shape:", future.shape)