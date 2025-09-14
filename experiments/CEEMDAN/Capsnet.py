"""
Time series Capsnet for a regression problem based on the model Y. Kim et al.

Summary:

Considerations:
    - 

Improvement Sugestions:
    - 
    

version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] Y. Kim, P. Wang, Y. Zhu and L. Mihaylova, "A Capsule Network for Traffic Speed Prediction in Complex Road Networks," 
2018 Sensor Data Fusion: Trends, Solutions, Applications (SDF), Bonn, Germany, 2018, pp. 1-6, doi: 10.1109/SDF.2018.8547068.

"""
import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/TimeCaps_Forecast/template_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import pywt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from utils import train_model
from collector import Collector
from Reg_Timescap import CapsNetRegressor

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'

predict_steps = 1


def Capsnet(predict_steps=1, batch_size=32, window_size = 50):

    np.random.seed(42)

    # Parameters
    serie_size = -1
    window_size = 50
    scales = np.arange(1, 31)  # 30 scales
    epochs = 20

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    dataset = ceemdan_collector.read_data(file, serie_size, window_size, predict_steps)

    
    test_ratio = 0.2
    split = int(len(dataset) * (1 - test_ratio))
    train_val_ds = torch.utils.data.Subset(dataset, range(split))
    test_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    batch_size = 32
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Validação cruzada temporal
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=5)
    val_losses_all = []


    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val_ds)):
        print(f"Fold {fold+1}")
        
        train_ds = torch.utils.data.Subset(train_val_ds, train_idx)
        val_ds = torch.utils.data.Subset(train_val_ds, val_idx)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Inicializa modelo novo
        input_shape = train_ds[0][0].shape
        model = CapsNetRegressor(input_shape, predict_steps=predict_steps)
        
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
    # Treina modelo final em todo treino+val
    # -----------------------------
    full_train_loader = DataLoader(train_val_ds, batch_size=batch_size, shuffle=False)
    input_shape = train_val_ds[0][0].shape
    final_model = CapsNetRegressor(input_shape, predict_steps=predict_steps)
    train_model(final_model, full_train_loader, test_loader=None, epochs=epochs)  # Sem validação agora

    # -----------------------------
    # Avaliação no conjunto de teste
    # -----------------------------
    final_model.eval()
    preds = []
    actuals = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(next(final_model.parameters()).device)
            pred = final_model(xb).cpu().numpy()
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
    Capsnet(predict_steps)