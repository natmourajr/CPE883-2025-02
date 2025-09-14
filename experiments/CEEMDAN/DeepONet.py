"""
DeepONet.

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
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/DeepONet/DeepONet_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import math
from collector import Collector
from DeepONet_model import DeepONet


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'


def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            # xb: [batch, window_size, 1]
            xb = xb.squeeze(-1).to(device)  # [batch, window_size]
            yb = yb.to(device)               # [batch, predict_steps]

            # Construir entrada trunk como passo(s) futuro(s) para cada batch
            # Exemplo: se predict_steps=1, trunk_input = [[1], [1], ..., [1]] shape: [batch, 1]
            batch_size = xb.size(0)
            pred_steps = yb.size(1)
            trunk_input = torch.arange(1, pred_steps+1, dtype=torch.float32).unsqueeze(0).repeat(batch_size,1).to(device)

            optimizer.zero_grad()
            preds = model(xb, trunk_input)
            loss = criterion(preds, yb.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.squeeze(-1).to(device)
                yb = yb.to(device)
                batch_size = xb.size(0)
                pred_steps = yb.size(1)
                trunk_input = torch.arange(1, pred_steps+1, dtype=torch.float32).unsqueeze(0).repeat(batch_size,1).to(device)
                preds = model(xb, trunk_input)
                loss = criterion(preds, yb.squeeze(-1))
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses


def DPNET():
        # Parâmetros (ajuste conforme seu dataset)
    window_size = 50      # quantidade de passos históricos usados na previsão
    predict_steps = 1     # horizonte de previsão
    batch_size = 32
    epochs = 20
    hidden_dim = 128      # tamanho do espaço latente DeepONet

    base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'
    file = 'final_la_haute_R0711.csv'

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    train_loader, test_loader  = ceemdan_collector.read_data(
        file, serie_size=-1, window_size=window_size, predict_steps=predict_steps,
        batch_size=batch_size, freq_transform=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONet(branch_input_dim=window_size,
                     trunk_input_dim=predict_steps,
                     hidden_dim=hidden_dim,
                     output_dim=1)

    train_loss, val_loss = train_model(model, train_loader, test_loader, epochs=epochs, device=device)

    # Plot perdas
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('DeepONet Training Loss (Energia Eólica)')
    plt.show()

    # Avaliação final
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.squeeze(-1).to(device)
            yb = yb.to(device)
            batch_size = xb.size(0)
            trunk_input = torch.arange(1, predict_steps+1, dtype=torch.float32).unsqueeze(0).repeat(batch_size,1).to(device)
            pred = model(xb, trunk_input).cpu().numpy()
            preds.append(pred)
            actuals.append(yb.squeeze(-1).cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    plt.figure(figsize=(10,4))
    plt.plot(actuals, label='True')
    plt.plot(preds, label='Predicted')
    plt.legend()
    plt.title('Previsão DeepONet - Energia Eólica')
    plt.show()


    # Calcula o erro
    mse = np.mean((preds - actuals) ** 2)
    mae = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(mse)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")


if __name__=='__main__':

    DPNET()