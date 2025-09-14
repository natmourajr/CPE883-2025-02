"""
Time series CNN for a regression problem.

Summary:

Considerations:
    - 

Improvement Sugestions:
    - Try a regularization: Include a dropout layer to prevent overfitting, or a L2 regularization in optimizer (validation loss is not decreasing).
    - Decrease the model complexity.
    - Normalized scalograms the data before the model inputation.
    

version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] 

"""

import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/CNN/Reg_CNN/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collector import Collector
from Reg_CNN import CNNRegressor

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'


# 5. Training loop
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


def CNN():
    np.random.seed(42)

    # Parameters
    serie_size = -1
    window_size = 50    # Number of examples in each time series batch
    predict_steps = 1
    scales = np.arange(1, 31)  # 30 scales
    batch_size = 32
    epochs = 20

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    dataset = ceemdan_collector.read_data(file, serie_size, window_size, predict_steps)

    # Time-based split: no shuffling
    split = int(0.8 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    test_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # CNN Model
    input_shape = train_ds[0][0].shape  # (1, S, T)
    model = CNNRegressor(input_shape, output_size=predict_steps)

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


if __name__ == "__main__":

    CNN()