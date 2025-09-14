import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/GRU/GRU_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from collector import Collector
from GRU_model import GRUPredictor


base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'


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


def GRU(predict_steps=1):
    # Parameters
    serie_size = -1
    window_size = 50    # Number of examples in each time series batch
    batch_size = 32
    epochs = 20

    input_size = 1
    hidden_size = 64
    num_layers = 2
    output_size = 1

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    train_loader, test_loader = ceemdan_collector.read_data(
        file, serie_size, window_size, predict_steps, batch_size, freq_transform=False
    )

    model = GRUPredictor(input_size, hidden_size, num_layers, output_size)

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


if __name__ == '__main__':

    GRU()