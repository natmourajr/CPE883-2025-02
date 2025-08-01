"""
Time series KAN transform.

Summary: Kan Classification of 3W pipeline data using P-MON-CKP and T-JUS-CKP as inputs.

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
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timm.layers import PatchEmbed
from typing import Union, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from dataloaders.benchmark.collector import Collector3W

# Ajuste o caminho para a pasta 'kat' (onde está o katransformer.py)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'kat/')))
from katransformer import KATVisionTransformer

base_folder = os.getenv("BASE_FOLDER")


# PatchEmbed adaptado para séries multivariadas (2 canais)
class PatchEmbed1D(nn.Module):
    def __init__(self, img_size=None, patch_size=16, embed_dim=128, in_chans=2, **kwargs):
        super().__init__()
        seq_len = img_size[0] if isinstance(img_size, tuple) else (img_size or 128)
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2)
        return x


# Classificador baseado em KAT
class KATTimeSeriesClassifier(KATVisionTransformer):
    def __init__(self, seq_len, patch_size, num_classes, embed_dim=128, depth=6, num_heads=4):
        super().__init__(
            img_size=seq_len,
            patch_size=patch_size,
            in_chans=2,  # <- 2 features: P-MON-CKP e T-JUS-CKP
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            embed_layer=PatchEmbed1D,
        )


# Dataset personalizado a partir do 3W
class ThreeWSeriesDataset(Dataset):
    def __init__(self, df, seq_len, normalize=True):
        self.seq_len = seq_len
        self.data = []
        self.labels = []

        # Seleção e normalização
        p_series = df['P-MON-CKP'].values
        t_series = df['T-JUS-CKP'].values
        if normalize:
            p_series = (p_series - p_series.mean()) / (p_series.std() + 1e-8)
            t_series = (t_series - t_series.mean()) / (t_series.std() + 1e-8)

        combined_series = np.stack([p_series, t_series], axis=1)
        instance_ids = df['instance_id'].astype(int).values

        # Criar janelas deslizantes
        for i in range(0, len(df) - seq_len):
            window = combined_series[i : i + seq_len]
            label = instance_ids[i + seq_len - 1]
            self.data.append(window)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# Treinamento
def train_model(model, loader, num_epochs=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/total:.4f} | Accuracy={accuracy:.2f}%")

    return model


# Avaliação
def evaluate_model(model, loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            correct += (predicted == yb).sum().item()
            total += yb.size(0)

    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")

    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues', values_format='d')

    # Aumenta tamanho da fonte
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for text in disp.text_.ravel():
        text.set_fontsize(12)
    
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.show()


# --- Execução Principal ---
if __name__ == "__main__":

    seq_len = 64
    patch_size = 8
    num_classes = 6    # <- Ajuste conforme seu dataset real
    batch_size = 128
    num_epochs = 10

    base_folder = os.getenv("BASE_FOLDER")
    dataset = Collector3W(
        data_path=os.path.join(base_folder, 'dataloaders/benchmark/_3w_dataset/data'),
        undesirable_event_code=1,
        train=True
    )
    test_dataset = Collector3W(
        data_path=os.path.join(base_folder, 'dataloaders/benchmark/_3w_dataset/data'),
        undesirable_event_code=1,
        train=False
    )

    df = dataset.df_samples
    df_test = test_dataset.df_samples

    import ipdb
    ipdb.set_trace()

    train_dataset = ThreeWSeriesDataset(df, seq_len=seq_len)
    test_dataset  = ThreeWSeriesDataset(df_test, seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = KATTimeSeriesClassifier(seq_len=seq_len, patch_size=patch_size, num_classes=num_classes)

    trained_model = train_model(model, train_loader, num_epochs=num_epochs)

    evaluate_model(trained_model, test_loader)
