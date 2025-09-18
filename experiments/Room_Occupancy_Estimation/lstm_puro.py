import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from collections import Counter

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from datetime import datetime

# 🔹 Importa o modelo separado
from models.Room_Occupancy_Estimation.lstm_puro import Pure_LSTM

# ==============================================================
# Carregamento e Pré-processamento dos dados
# ==============================================================

room_occupancy_estimation = fetch_ucirepo(id=864)
X_df = room_occupancy_estimation.data.features
y_df = room_occupancy_estimation.data.targets

X_df = X_df.select_dtypes(include=np.number)

# Normaliza os dados de entrada
X_norm = (X_df - X_df.mean()) / X_df.std()

# Define o tamanho da janela de tempo
window_size = 25
features_size = X_norm.shape[1]

# Função para criar janelas
def create_sequences(input_data, output_data, window_size):
    in_seq, out_seq = [], []
    L = len(input_data)
    for i in range(L - window_size):
        input_seq = input_data[i:i + window_size]
        output_seq = output_data[i + window_size]
        in_seq.append(input_seq)
        out_seq.append(output_seq)
    return np.array(in_seq), np.array(out_seq)

# Prepara os dados
X_data = X_norm.values
y_data = y_df.values.flatten()

X_seq, y_seq = create_sequences(X_data, y_data, window_size)

# Tensores
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.long)
print("Classes únicas em y:", np.unique(y_data))

# ==============================================================
# Divisão e balanceamento inicial
# ==============================================================

X_flat_before_split = X_tensor.view(X_tensor.shape[0], -1)

X_train, X_val, y_train, y_val = train_test_split(
    X_flat_before_split, y_tensor,
    test_size=0.2,
    random_state=42,
    stratify=y_tensor
)

print("\nDistribuição das classes no conjunto de treino original:")
print(Counter(y_train.numpy()))

# Aplica SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.numpy(), y_train.numpy())

print("\nDistribuição das classes no conjunto de treino após o oversampling:")
print(Counter(y_train_resampled))

# Remodela
X_train_final = torch.tensor(X_train_resampled, dtype=torch.float32).view(-1, window_size, features_size)
y_train_final = torch.tensor(y_train_resampled, dtype=torch.long)
X_val_final = X_val.view(-1, window_size, features_size)
y_val_final = y_val

# DataLoaders
train_dataset = TensorDataset(X_train_final, y_train_final)
val_dataset = TensorDataset(X_val_final, y_val_final)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"\nDimensões do tensor de treino (rebalanceado): {X_train_final.shape}")
print(f"Dimensões do tensor de validação (original): {X_val_final.shape}")
print(f"Número de amostras de treino (rebalanceado): {len(train_dataset)}")
print(f"Número de amostras de validação (original): {len(val_dataset)}")

# ==============================================================
# Treinamento com Validação Cruzada
# ==============================================================

n_classes = len(np.unique(y_data))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hiperparâmetros
lr = 5e-4
num_epochs = 100
patience = 15
criterion = nn.CrossEntropyLoss()

X_flat = X_tensor.view(X_tensor.shape[0], -1)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
all_train_losses = []
all_val_losses = []
fold_accuracies = []
fold_reports = []

print("Iniciando treinamento com Validação Cruzada K-Fold...")

for fold, (train_index, val_index) in enumerate(kf.split(X_flat)):
    print(f"\n===== Fold {fold + 1}/{n_splits} =====")

    X_train_fold, X_val_fold = X_flat[train_index], X_flat[val_index]
    y_train_fold, y_val_fold = y_tensor[train_index], y_tensor[val_index]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold.numpy(), y_train_fold.numpy())

    X_train_final = torch.tensor(X_train_resampled, dtype=torch.float32).view(-1, window_size, features_size)
    y_train_final = torch.tensor(y_train_resampled, dtype=torch.long)
    X_val_final = X_val_fold.view(-1, window_size, features_size)
    y_val_final = y_val_fold

    train_dataset = TensorDataset(X_train_final, y_train_final)
    val_dataset = TensorDataset(X_val_final, y_val_final)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 🔹 Usa o modelo importado
    model_pure = Pure_LSTM(
        input_dim=features_size,
        hidden_dim=128,
        num_layers=1,
        output_dim=n_classes,
        dropout=0.4
    ).to(device)

    optimizer = optim.Adam(model_pure.parameters(), lr=lr)

    loss_list_pure = []
    val_loss_list_pure = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model_pure.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_pure(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        loss_list_pure.append(train_loss)

        model_pure.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_pure(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list_pure.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model_pure.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping at epoch", epoch + 1)
                break

    model_pure.load_state_dict(best_model_state)
    model_pure.eval()

    y_pred_fold = []
    y_true_fold = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_pure(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_fold.extend(preds)
            y_true_fold.extend(labels.cpu().numpy())

    fold_accuracy = accuracy_score(y_true_fold, y_pred_fold)
    fold_accuracies.append(fold_accuracy)

    fold_report = classification_report(y_true_fold, y_pred_fold, zero_division=0, output_dict=True)
    fold_reports.append(fold_report)

    print(f"\n--- Resultados do Fold {fold + 1} ---")
    print(f"Acurácia: {fold_accuracy:.4f}")
    print(f"Precision (média): {fold_report['weighted avg']['precision']:.4f}")
    print(f"Recall (média): {fold_report['weighted avg']['recall']:.4f}")
    print(f"F1-Score (média): {fold_report['weighted avg']['f1-score']:.4f}")

    all_y_true.extend(y_true_fold)
    all_y_pred.extend(y_pred_fold)
    all_train_losses.append(loss_list_pure)
    all_val_losses.append(val_loss_list_pure)

print("\nTreinamento com Validação Cruzada concluído.")

# ==============================================================
# Relatórios e Plots
# ==============================================================

print("\n" + "="*50)
print("RELATÓRIO DETALHADO POR FOLD")
print("="*50)

for fold in range(n_splits):
    print(f"\n--- Fold {fold + 1} ---")
    print(f"Acurácia: {fold_accuracies[fold]:.4f}")
    print(f"Precision: {fold_reports[fold]['weighted avg']['precision']:.4f}")
    print(f"Recall: {fold_reports[fold]['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {fold_reports[fold]['weighted avg']['f1-score']:.4f}")

print("\n" + "="*50)
print("ESTATÍSTICAS GERAIS DA VALIDAÇÃO CRUZADA")
print("="*50)
print(f"Acurácia média: {np.mean(fold_accuracies):.4f} (±{np.std(fold_accuracies):.4f})")
print(f"Precision média: {np.mean([report['weighted avg']['precision'] for report in fold_reports]):.4f}")
print(f"Recall médio: {np.mean([report['weighted avg']['recall'] for report in fold_reports]):.4f}")
print(f"F1-Score médio: {np.mean([report['weighted avg']['f1-score'] for report in fold_reports]):.4f}")

# Curva de perda
max_len = max(len(l) for l in all_train_losses)
avg_train_loss = np.nanmean([np.pad(l, (0, max_len - len(l)), 'constant', constant_values=np.nan) for l in all_train_losses], axis=0)
avg_val_loss = np.nanmean([np.pad(l, (0, max_len - len(l)), 'constant', constant_values=np.nan) for l in all_val_losses], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(avg_train_loss, label='Média de Treino')
plt.plot(avg_val_loss, label='Média de Validação')
plt.title('Curva de Perda Média (Treino vs Validação) - LSTM Pura')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Relatório final
print("\n===== Relatório de Classificação Geral (Todos os Folds) =====")
print(classification_report(all_y_true, all_y_pred, zero_division=0))

print("\n===== Matriz de Confusão em Porcentagem =====")
cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=False,
            xticklabels=np.unique(all_y_true), yticklabels=np.unique(all_y_true))
plt.title('Matriz de Confusão Normalizada (Todos os Folds)')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, n_splits + 1), fold_accuracies, color='skyblue', alpha=0.7)
plt.axhline(y=np.mean(fold_accuracies), color='red', linestyle='--',
            label=f'Média: {np.mean(fold_accuracies):.4f}')
plt.xlabel('Fold')
plt.ylabel('Acurácia')
plt.title('Acurácia por Fold na Validação Cruzada')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, n_splits + 1))
plt.ylim(0, 1)
plt.show()
