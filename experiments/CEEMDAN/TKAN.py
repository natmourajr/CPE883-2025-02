import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/TKAN/TKAN_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tkan import TKAN
from collector import Collector
from TKAN_model import TKANModel


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'

    
# Função de treinamento
def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    train_loss = 0

    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    return train_losses, val_losses

def TKAN():
    # Parameters
    serie_size = -1
    window_size = 50    # Number of examples in each time series batch
    predict_steps = 1
    batch_size = 32
    epochs = 4
    
    input_dim = 1
    hidden_dim = 64
    output_dim = 1

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    train_loader, test_loader  = ceemdan_collector.read_data(
        file, serie_size, window_size, predict_steps, batch_size, freq_transform=False
    )

    # Definir o modelo
    model = TKANModel(input_dim, hidden_dim, output_dim)

    # Definir o otimizador e a função de perda
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Train
    train_loss, val_loss = train_model(model, train_loader, optimizer, criterion, epochs=epochs)

    # Plot loss
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()

    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(next(model.parameters()).device)  # Move entrada para dispositivo do modelo
            pred = model(xb).cpu().numpy()                # Previsão para CPU + numpy
            preds.append(pred)

            yb = yb.cpu().numpy()
            actuals.append(yb)

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Caso necessário, ajuste shape (se for [N, 1], deixa só [N])
    if preds.ndim > 1 and preds.shape[1] == 1:
        preds = preds.squeeze(axis=1)
    if actuals.ndim > 1 and actuals.shape[1] == 1:
        actuals = actuals.squeeze(axis=1)

    # Plotando as previsões x valores reais
    plt.figure(figsize=(10, 4))
    plt.plot(actuals, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Model Predictions vs Actual on Test Set")
    plt.show()
    

if __name__=='__main__':

   TKAN()