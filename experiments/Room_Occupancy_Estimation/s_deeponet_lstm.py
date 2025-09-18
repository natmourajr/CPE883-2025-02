# preprocess_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE
from collections import Counter

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Adiciona a pasta models ao path para importação
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Importar modelos do arquivo s_deeponet_lstm.py na pasta models
from models.Room_Occupancy_Estimation.s_deeponet_lstm import BranchNet_LSTM, TrunkNet, S_DeepONet_LSTM

# Carregamento e Pré-processamento dos dados com oversampling
room_occupancy_estimation = fetch_ucirepo(id=864)
X_df = room_occupancy_estimation.data.features
y_df = room_occupancy_estimation.data.targets

X_df = X_df.select_dtypes(include=np.number)

# Normaliza os dados de entrada
X_norm = (X_df - X_df.mean()) / X_df.std()

# Define o tamanho da janela de tempo
window_size = 25
features_size = X_norm.shape[1]

# Cria as janelas de dados (sequências)
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

# Cria as sequências
X_seq, y_seq = create_sequences(X_data, y_data, window_size)

# Converte os arrays para tensores PyTorch
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.long)
print("Classes únicas em y:", np.unique(y_data))

# Achata o tensor de entrada para que o SMOTE possa ser aplicado
X_flat_before_split = X_tensor.view(X_tensor.shape[0], -1)

# Divide os dados em treino e validação ANTES do oversampling
X_train, X_val, y_train, y_val = train_test_split(
    X_flat_before_split, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor
)
print("\nDistribuição das classes no conjunto de treino original:")
print(Counter(y_train.numpy()))

# Aplica o SMOTE no conjunto de treino
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.numpy(), y_train.numpy())

print("\nDistribuição das classes no conjunto de treino após o oversampling:")
print(Counter(y_train_resampled))

# Remodela os tensores de volta para o formato de série temporal para LSTM/GRU
X_train_final = torch.tensor(X_train_resampled, dtype=torch.float32).view(-1, window_size, features_size)
y_train_final = torch.tensor(y_train_resampled, dtype=torch.long)

# O conjunto de validação NÃO é rebalanceado
X_val_final = X_val.view(-1, window_size, features_size)
y_val_final = y_val

# Cria os DataLoaders a partir dos dados reamostrados
train_dataset = TensorDataset(X_train_final, y_train_final)
val_dataset = TensorDataset(X_val_final, y_val_final)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Verificação das dimensões finais
print(f"\nDimensões do tensor de treino (rebalanceado): {X_train_final.shape}")
print(f"Dimensões do tensor de validação (original): {X_val_final.shape}")
print(f"Número de amostras de treino (rebalanceado): {len(train_dataset)}")
print(f"Número de amostras de validação (original): {len(val_dataset)}")

# Função de Treinamento e Validação Simplificada

def train_and_evaluate_simple(model, train_loader, val_loader, num_epochs, lr, n_classes, device, class_weights):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    train_losses, val_losses = [], []

    print("Iniciando o treinamento...")
    for epoch in range(num_epochs):
        # Treino
        model.train()
        running_loss = 0.0
        for u_batch, y_batch, labels in train_loader:
            u_batch, y_batch, labels = u_batch.to(device), y_batch.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(u_batch, y_batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validação
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for u_batch, y_batch, labels in val_loader:
                u_batch, y_batch, labels = u_batch.to(device), y_batch.to(device), labels.to(device)
                outputs = model(u_batch, y_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    print("\nTreinamento concluído.")
    return train_losses, val_losses, all_labels, all_preds

# Execução com K-Fold Cross-Validation

# Parâmetros
num_epochs = 150
lr = 1e-4
n_classes = len(np.unique(y_tensor.numpy()))
device = "cuda" if torch.cuda.is_available() else "cpu"
num_folds = 5 

# Concatenar todos os dados para a validação cruzada
X_all = torch.cat((X_train_final, X_val_final), dim=0)
y_all = torch.cat((y_train_final, y_val_final), dim=0)
time_coords = torch.arange(window_size, dtype=torch.float32).unsqueeze(0).repeat(X_all.shape[0], 1)
full_dataset = TensorDataset(X_all, time_coords, y_all)

# Inicializar K-Fold
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Listas para armazenar os resultados de cada fold
fold_accuracies = []
fold_f1_scores = [] 
all_labels_folds = []
all_preds_folds = []


# Iterar sobre cada fold
for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f'--- Fold {fold + 1}/{num_folds} ---')

    # Criar subconjuntos de dados para treino e validação
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Criar DataLoaders para este fold
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_subsampler)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_subsampler)

    # Inicializar o modelo para o fold
    model = S_DeepONet_LSTM(
        branch_input_dim=features_size,
        branch_hidden_dim=128,
        num_layers_lstm=1,
        trunk_input_dim=window_size,
        trunk_hidden_layers=[64, 32],
        output_dim=n_classes,
        dropout=0.5
    ).to(device)

    # Pesos de classe
    y_train_fold = y_all[train_ids].numpy()
    class_counts = np.bincount(y_train_fold)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Treinar e avaliar o modelo no fold atual
    train_losses, val_losses, all_labels, all_preds = train_and_evaluate_simple(
        model, train_loader, val_loader, num_epochs, lr, n_classes, device, class_weights
    )

    # Calcular e armazenar as métricas para este fold
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  

    fold_accuracies.append(accuracy)
    fold_f1_scores.append(f1) 

    print(f'Acurácia para o Fold {fold + 1}: {accuracy:.4f}')
    print(f'F1 Score para o Fold {fold + 1}: {f1:.4f}')

    # Armazenar labels e previsões para a matriz de confusão geral
    all_labels_folds.extend(all_labels)
    all_preds_folds.extend(all_preds)

# Resultados Finais da Validação Cruzada

print('\n--- Resultados Finais ---')
print(f'Acurácias por Fold: {[f"{acc:.4f}" for acc in fold_accuracies]}')
print(f'F1 Scores por Fold: {[f"{f1:.4f}" for f1 in fold_f1_scores]}')
print(f'Acurácia Média: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}')
print(f'F1 Score Médio: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}')

# ---
# 1. Matriz de Confusão Média (Combinando todos os folds)

# Recalcula a matriz de confusão com todas as previsões
cm = confusion_matrix(all_labels_folds, all_preds_folds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=range(n_classes), yticklabels=range(n_classes))
plt.title('Matriz de Confusão Média (Todos os Folds)')
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.show()

# ---
# 2. Relatório de Classificação Detalhado
print('\n--- Relatório de Classificação (Todos os Folds) ---')
print(classification_report(all_labels_folds, all_preds_folds,
                           target_names=[f'Classe {i}' for i in range(n_classes)]))

# ---
# 3. Curva de Perda (Você pode plotar a curva do último fold ou a média)
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Perda de Treino')
plt.plot(val_losses, label='Perda de Validação')
plt.title('Curva de Perda (Último Fold)')
plt.xlabel('Época')
plt.ylabel('Perda (Loss)')
plt.legend()
plt.grid(True)
plt.show()