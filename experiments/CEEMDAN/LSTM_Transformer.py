"""
Hybrid LSTM + Transformer for short sequences (window_size=6)
"""
import os
import sys
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/models/LSTM/LSTM_model/")
sys.path.append("/home/felipe/doutorado/CPE883-2025-02/dataloaders/CEEMDAN/")
file = 'final_la_haute_R0711.csv'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from matplotlib import pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import math
from collector import Collector
from utils import train_model, NormalizedDataset

base_path = '/home/felipe/doutorado/CEEMDAN-EWT-LSTM/dataset'

# -----------------------------
# Hybrid LSTM + Transformer
# -----------------------------
class LSTMTransformerHybrid(nn.Module):
    def __init__(self, input_dim=1, lstm_hidden=32, lstm_layers=1,
                 embed_dim=16, num_heads=2, num_layers=1, pred_len=1, dropout=0.1, window_size=6):
        super().__init__()
        self.window_size = window_size
        
        # LSTM para sequência curta
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Embedding para Transformer
        self.embedding = nn.Linear(lstm_hidden, embed_dim)
        # Positional encoding como parâmetro treinável
        self.pos_encoder = nn.Parameter(torch.zeros(1, window_size, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Head de previsão
        self.head = nn.Linear(embed_dim, pred_len)
        
    def forward(self, x):
        # x: [B, T, D]
        lstm_out, _ = self.lstm(x)          # [B, T, lstm_hidden]
        x = self.embedding(lstm_out)        # [B, T, embed_dim]
        x = x + self.pos_encoder            # adiciona positional encoding
        x = self.transformer(x)             # [B, T, embed_dim]
        x = x[:, -1, :]                     # pooling pelo último passo
        return self.head(x)                 # [B, pred_len]

# -----------------------------
# Função principal de treino/avaliação
# -----------------------------
def Path_HybridTransformer(serie_size=-1, window_size=6, predict_steps=1, batch_size=32, epochs=20,
                     input_size=1, lstm_hidden=32, embed_dim=16, num_heads=2, num_layers=1, test_ratio=0.2,
                     folds=4):

    model = LSTMTransformerHybrid(
        input_dim=input_size,
        lstm_hidden=lstm_hidden,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pred_len=predict_steps,
        window_size=window_size
    )

    np.random.seed(42)

    # Dataset & DataLoader
    ceemdan_collector = Collector(base_path)
    
    dataset  = ceemdan_collector.read_data(
        file, serie_size, window_size, predict_steps, batch_size, freq_transform=False
    )

    split = int(len(dataset) * (1 - test_ratio))
    train_val_ds = torch.utils.data.Subset(dataset, range(split))
    test_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    # -----------------------------
    # Validação cruzada temporal
    # -----------------------------
    tscv = TimeSeriesSplit(n_splits=folds)
    val_losses_all = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_val_ds)):
        print(f"Fold {fold+1}")
        
        train_ds = torch.utils.data.Subset(train_val_ds, train_idx)
        val_ds = torch.utils.data.Subset(train_val_ds, val_idx)

        train_ds = NormalizedDataset(train_ds, fit=True)
        val_ds = NormalizedDataset(
            val_ds,
            mean_X=train_ds.mean_X, std_X=train_ds.std_X,
            mean_y=train_ds.mean_y, std_y=train_ds.std_y,
            fit=False
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Inicializa modelo novo para o fold
        model_fold = LSTMTransformerHybrid(
            input_dim=input_size,
            lstm_hidden=lstm_hidden,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pred_len=predict_steps,
            window_size=window_size
        )
        
        # Treina
        train_loss, val_loss = train_model(model_fold, train_loader, val_loader, epochs=epochs)
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
    rmse_all = np.sqrt(val_losses_all)
    mean_rmse = np.mean(rmse_all)
    std_rmse = np.std(rmse_all)

    print(f"Validação Cruzada: μ MSE = {mean_val_loss:.4f}, σ MSE = {std_val_loss:.4f}")
    print(f"Validação Cruzada: μ RMSE = {mean_rmse:.4f}, σ RMSE = {std_rmse:.4f}")

    # -----------------------------
    # Treina modelo final em todo treino + val
    # -----------------------------
    train_val_ds = NormalizedDataset(train_val_ds, fit=True)
    test_ds = NormalizedDataset(
        test_ds,
        mean_X=train_val_ds.mean_X, std_X=train_val_ds.std_X,
        mean_y=train_val_ds.mean_y, std_y=train_val_ds.std_y,
        fit=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    full_train_loader = DataLoader(train_val_ds, batch_size=batch_size, shuffle=False)
    final_model = LSTMTransformerHybrid(
        input_dim=input_size,
        lstm_hidden=lstm_hidden,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        pred_len=predict_steps,
        window_size=window_size
    )
    train_model(final_model, full_train_loader, test_loader=None, epochs=epochs)  # Sem validação agora
    end_time = time.time()

    train_time = end_time - start_time
    print(f"Tempo de treino final: {train_time:.2f} segundos")

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

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)

    print(f"MSE final no conjunto de teste: {mse:.6f}")
    print(f"RMSE final no conjunto de teste: {rmse:.6f}")

    # Plot predictions vs ground truth
    plt.figure(figsize=(10, 4))
    plt.plot(actuals, label="True")
    plt.plot(preds, label="Predicted")
    plt.legend()
    plt.title("Model Predictions vs Actual on Test Set")
    plt.show()

    print("Finished...")

if __name__=='__main__':
    Path_HybridTransformer()
