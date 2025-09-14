"""
Transformer.

Summary:

Considerations:
    - 

Improvement Sugestions:
    - 
    

version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] 

"""
import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/LSTM/LSTM_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import math
from collector import Collector


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'


# Positional Encoding clássico (sinusoidal)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin para índices pares
        pe[:, 1::2] = torch.cos(position * div_term)  # cos para ímpares
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        x = x + self.pe[:, :x.size(1)]
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_len, embed_dim):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(input_dim * patch_len, embed_dim)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T // self.patch_len, self.patch_len * D)
        return self.proj(x)  # [B, Num_Patches, Embed_Dim]

class PatchTST(nn.Module):
    def __init__(self, input_dim=1, patch_len=4, embed_dim=64, num_heads=4,
                 num_layers=2, pred_len=4):
        super().__init__()
        self.embedding = PatchEmbedding(input_dim, patch_len, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, pred_len)

    def forward(self, x):
        # x: [B, T, D]
        x = self.embedding(x)            # [B, T_patches, embed_dim]
        x = self.pos_encoder(x)          # adiciona encoding posicional
        x = self.transformer(x)          # passa pelo transformer
        x = x.mean(dim=1)                # pooling global
        return self.head(x)              # [B, pred_len]

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses
    

def Path_Transformer():
    # Parameters
    serie_size = -1
    window_size = 50    # Number of examples in each time series batch
    predict_steps = 1
    batch_size = 32
    epochs = 20

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    train_loader, test_loader  = ceemdan_collector.read_data(
        file, serie_size, window_size, predict_steps, batch_size, freq_transform=False
    )

    model = PatchTST(
        input_dim=input_size,
        patch_len=5,            # Tamanho de patch (ajustável — window_size deve ser múltiplo de patch len)
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        pred_len=predict_steps  # Quantos passos futuros você quer prever
    )

    # Train
    train_loss, val_loss = train_model(model, train_loader, test_loader, epochs=epochs)

    # Plot loss
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

    # Evaluate final model on test set
    model.eval()
    preds = []
    actuals = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(next(model.parameters()).device)
            pred = model(xb).cpu().numpy()
            preds.append(pred)
            actuals.append(yb.numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 4))
    plt.plot(actuals, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Model Predictions vs Actual on Test Set")
    plt.show()

if __name__=='__main__':

    Path_Transformer()