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
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from collector import Collector
from DeepONet_model import DeepONet

from utils import train_model


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'


def DPNET(predict_steps=1, serie_size=-1, batch_size=32, window_size=50, epochs=20, test_ratio=0.2, hidden_dim = 128,
        folds=2):
    """
    hidden_dim = 128: tamanho do espaço latente DeepONet
    """

    np.random.seed(42)

   # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    
    dataset  = ceemdan_collector.read_data(
        file, serie_size, window_size, predict_steps, batch_size, freq_transform=False
    )

    split = int(len(dataset) * (1 - test_ratio))
    train_val_ds = torch.utils.data.Subset(dataset, range(split))
    test_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # -----------------------------
    # Validação cruzada temporal
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=folds)
    val_losses_all = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val_ds)):
        print(f"Fold {fold+1}")
        
        train_ds = torch.utils.data.Subset(train_val_ds, train_idx)
        val_ds = torch.utils.data.Subset(train_val_ds, val_idx)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Inicializa modelo novo
        model = DeepONet(branch_input_dim=window_size,
                    trunk_input_dim=predict_steps,
                    hidden_dim=hidden_dim,
                    output_dim=1)
        
        # Treina
        train_loss, val_loss = train_model(model, train_loader, val_loader, epochs=epochs)
        
        val_losses_all.append(val_loss[-1])
        
        # Plot loss por fold
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.title(f"Fold {fold+1} Loss")
        plt.legend()
        plt.show()

        # Estatísticas da validação cruzada
    mean_val_loss = np.mean(val_losses_all)
    std_val_loss = np.std(val_losses_all)
    print(f"Validação Cruzada: μ MSE = {mean_val_loss:.4f}, σ = {std_val_loss:.4f}")

    # -----------------------------
    # Treina modelo final em todo treino + val
    # -----------------------------
    full_train_loader = DataLoader(train_val_ds, batch_size=batch_size, shuffle=False)
    final_model = DeepONet(branch_input_dim=window_size,
                    trunk_input_dim=predict_steps,
                    hidden_dim=hidden_dim,
                    output_dim=1)
    train_model(final_model, full_train_loader, test_loader=None, epochs=epochs)  # Sem validação agora

    # -----------------------------
    # Avaliação no conjunto de teste
    # -----------------------------
    final_model.eval()
    preds = []
    actuals = []

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    plt.figure(figsize=(10,4))
    plt.plot(actuals, label='True')
    plt.plot(preds, label='Predicted')
    plt.legend()
    plt.title('Previsão DeepONet - Energia Eólica')
    plt.show()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 4))
    plt.plot(actuals, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Model Predictions vs Actual on Test Set")
    plt.show()

    print("Finished...")


if __name__=='__main__':

    DPNET()