# preprocess_train_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
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

# Importar modelos do arquivo mlp_model.py na pasta Room_Occupancy_Estimation
from models.Room_Occupancy_Estimation.mlp_puro import MLP_Model

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

# Treinamento, Avaliação e Visualização com Validação Cruzada

# Define o número de folds (K) para a validação cruzada
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listas para armazenar as métricas de cada fold
all_val_reports = []
all_val_cms = []
all_val_losses_final = []
all_train_losses_full = []
all_val_losses_full = []
all_val_accuracies = []
all_val_f1_scores = []

# Diretório para salvar resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir_base = f"/content/drive/MyDrive/resultados/cv_mlp/{timestamp}"
os.makedirs(save_dir_base, exist_ok=True)

# Converte os dados para tensores do PyTorch
X_data_tensor = torch.tensor(X_data, dtype=torch.float32)
y_data_tensor = torch.tensor(y_data, dtype=torch.long)
dataset = TensorDataset(X_data_tensor, y_data_tensor)

# Calcula a dimensão de entrada da MLP com base nos dados
sample_batch = next(iter(DataLoader(dataset, batch_size=32, shuffle=False)))[0]
mlp_input_dim = sample_batch.view(sample_batch.size(0), -1).size(1)
print(f"A dimensão de entrada para a MLP será: {mlp_input_dim}")

# Importar métricas adicionais
from sklearn.metrics import f1_score

# --- Início do loop de validação cruzada ---
for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
    print(f"\n--- Iniciando o Fold {fold + 1}/{n_splits} ---")

    # Cria subconjuntos de dados para treino e validação
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)

    # Cria os DataLoaders para este fold
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instanciação do modelo MLP dentro do loop
    n_classes = len(np.unique(y_data))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pesos de classe para este fold específico
    y_train_fold = y_data_tensor[train_index].numpy()
    class_counts_fold = np.array([np.sum(y_train_fold == i) for i in range(n_classes)])
    class_weights_fold = 1.0 / class_counts_fold
    class_weights_fold = class_weights_fold / class_weights_fold.sum()
    class_weights_fold = torch.tensor(class_weights_fold, dtype=torch.float32)

    model = MLP_Model(
        input_dim=mlp_input_dim,
        hidden_layers=[512, 256, 128],
        output_dim=n_classes,
        activation_fn="ReLU",
        output_activation=None,
        dropout=0.3,
        use_batchnorm=True,
        lr=1e-4,
        weight_decay=1e-5,
        num_epochs=100,
        class_weights=class_weights_fold,
        patience=15
    ).to(device)

    # Listas para as perdas deste fold
    fold_train_losses = []
    fold_val_losses = []

    # Loop de treinamento e validação para este fold
    for epoch in range(model.num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.view(X_batch.size(0), -1).to(device), y_batch.to(device)
            model.optimizer.zero_grad()
            outputs = model.forward(X_batch)
            loss = model.criterion(outputs, y_batch)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item()
        fold_train_losses.append(running_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.view(X_batch.size(0), -1).to(device), y_batch.to(device)
                outputs = model.forward(X_batch)
                loss = model.criterion(outputs, y_batch)
                val_loss += loss.item()
        fold_val_losses.append(val_loss / len(val_loader))

        if (epoch + 1) % 10 == 0:
            print(f"Fold {fold+1}, Epoch {epoch+1}/{model.num_epochs}, Train Loss: {fold_train_losses[-1]:.4f}, Val Loss: {fold_val_losses[-1]:.4f}")

    # Salva a curva de perda para este fold
    plt.figure(figsize=(10,6))
    plt.plot(fold_train_losses, label="Treino")
    plt.plot(fold_val_losses, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title(f"Curva de Perda - Fold {fold+1}")
    plt.grid(alpha=0.5)
    plt.legend()
    curve_path = os.path.join(save_dir_base, f"curva_perda_fold_{fold+1}.png")
    plt.savefig(curve_path)
    plt.close()
    print(f"Curva de perda do Fold {fold+1} salva em: {curve_path}")

    # Avaliação final no conjunto de validação do fold
    model.eval()
    y_pred_fold = []
    y_true_fold = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.view(xb.size(0), -1).to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_pred_fold.extend(preds)
            y_true_fold.extend(yb.cpu().numpy())

    # Calcula métricas adicionais
    accuracy_fold = accuracy_score(y_true_fold, y_pred_fold)
    f1_fold = f1_score(y_true_fold, y_pred_fold, average='weighted')

    # Armazena as métricas do fold
    report_fold = classification_report(y_true_fold, y_pred_fold, labels=range(n_classes), zero_division=0, output_dict=True)
    cm_fold = confusion_matrix(y_true_fold, y_pred_fold, labels=range(n_classes))
    all_val_reports.append(report_fold)
    all_val_cms.append(cm_fold)
    all_val_accuracies.append(accuracy_fold)
    all_val_f1_scores.append(f1_fold)

    # Armazena a perda de validação final e as curvas completas para a análise final
    all_val_losses_final.append(fold_val_losses[-1])
    all_train_losses_full.append(fold_train_losses)
    all_val_losses_full.append(fold_val_losses)

    # Imprime os resultados deste fold
    print(f"\n===== Relatório de Classificação - Fold {fold+1} =====")
    print(f"Acurácia: {accuracy_fold:.4f}")
    print(f"F1 Score: {f1_fold:.4f}")
    print(classification_report(y_true_fold, y_pred_fold, labels=range(n_classes), zero_division=0))
    print(f"===== Matriz de Confusão - Fold {fold+1} =====")
    print(cm_fold)

# --- Fim do loop de validação cruzada ---

# --- Resultados Finais da Validação Cruzada ---
print("\n--- Resultados Finais da Validação Cruzada ---")

# Calcular estatísticas finais
mean_accuracy = np.mean(all_val_accuracies)
std_accuracy = np.std(all_val_accuracies)
mean_f1 = np.mean(all_val_f1_scores)
std_f1 = np.std(all_val_f1_scores)
mean_val_loss = np.mean(all_val_losses_final)
std_val_loss = np.std(all_val_losses_final)

print(f"\n=== ESTATÍSTICAS FINAIS ===")
print(f"Acurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"F1 Score Médio: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Perda de Validação Média: {mean_val_loss:.4f} ± {std_val_loss:.4f}")

print(f"\nAcurácias por Fold: {[f'{acc:.4f}' for acc in all_val_accuracies]}")
print(f"F1 Scores por Fold: {[f'{f1:.4f}' for f1 in all_val_f1_scores]}")

# Visualizar F1 Scores por fold
plt.figure(figsize=(10, 6))
plt.bar(range(1, n_splits + 1), all_val_f1_scores, color='orange', alpha=0.7)
plt.axhline(y=mean_f1, color='red', linestyle='--', label=f'Média: {mean_f1:.4f}')
plt.title('F1 Score por Fold - Validação Cruzada', fontsize=16, fontweight='bold')
plt.xlabel('Fold', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, n_splits + 1))
f1_bar_path = os.path.join(save_dir_base, "f1_score_por_fold.png")
plt.savefig(f1_bar_path)
plt.show()
print(f"Gráfico de F1 Score por fold salvo em: {f1_bar_path}")

# --- Matriz de Confusão Média com Desvio Padrão ---
all_val_cms_array = np.array(all_val_cms)
mean_cm = np.mean(all_val_cms_array, axis=0)
std_cm = np.std(all_val_cms_array, axis=0)
cm_labels = np.array([[f'{mean_cm[i, j]:.2f} ± {std_cm[i, j]:.2f}' for j in range(n_classes)] for i in range(n_classes)])
plt.figure(figsize=(10, 8))
sns.heatmap(mean_cm, annot=cm_labels, fmt='', cmap='Blues',
            xticklabels=[f'Classe {i}' for i in range(n_classes)],
            yticklabels=[f'Classe {i}' for i in range(n_classes)])
plt.title("Matriz de Confusão Média com Desvio Padrão (5 Folds)")
plt.xlabel("Predito")
plt.ylabel("Real")
conf_path_avg_std = os.path.join(save_dir_base, "matriz_confusao_media_std.png")
plt.savefig(conf_path_avg_std)
plt.show()
print(f"Matriz de confusão média com desvio padrão salva em: {conf_path_avg_std}")

# --- Matriz de Confusão Normalizada com Média e Desvio Padrão ---
normalized_cms = [cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] for cm in all_val_cms]
normalized_cms_array = np.array(normalized_cms)
mean_normalized_cm = np.mean(normalized_cms_array, axis=0)
std_normalized_cm = np.std(normalized_cms_array, axis=0)
normalized_cm_labels = np.array([[f'{mean_normalized_cm[i, j]:.2%} ± {std_normalized_cm[i, j]:.2%}' for j in range(n_classes)] for i in range(n_classes)])
plt.figure(figsize=(10, 8))
sns.heatmap(mean_normalized_cm, annot=normalized_cm_labels, fmt='', cmap='Blues',
            xticklabels=[f'Classe {i}' for i in range(n_classes)],
            yticklabels=[f'Classe {i}' for i in range(n_classes)])
plt.title("Matriz de Confusão em Porcentagem Média com Desvio Padrão (5 Folds)")
plt.xlabel("Predito")
plt.ylabel("Real")
conf_path_normalized_avg_std = os.path.join(save_dir_base, "matriz_confusao_normalized_media_std.png")
plt.savefig(conf_path_normalized_avg_std)
plt.show()
print(f"Matriz de confusão normalizada com média e desvio padrão salva em: {conf_path_normalized_avg_std}")

# --- Curva de Perda do Melhor Fold ---

# Encontra o índice do fold com a menor perda de validação final
best_fold_index = np.argmin(all_val_losses_final)
best_fold_loss = all_val_losses_final[best_fold_index]
best_fold_f1 = all_val_f1_scores[best_fold_index]

# Recupera as curvas de perda de treino e validação do melhor fold
best_train_losses = all_train_losses_full[best_fold_index]
best_val_losses = all_val_losses_full[best_fold_index]

print(f"\nO melhor fold foi o Fold {best_fold_index + 1}")
print(f"Perda de validação final: {best_fold_loss:.4f}")
print(f"F1 Score: {best_fold_f1:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(best_train_losses, label=f"Treino (Melhor Fold)")
plt.plot(best_val_losses, label=f"Validação (Melhor Fold)")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title(f"Curva de Perda do Melhor Fold ({best_fold_index + 1}) - F1: {best_fold_f1:.4f}")
plt.grid(alpha=0.5)
plt.legend()
best_curve_path = os.path.join(save_dir_base, "curva_perda_melhor_fold.png")
plt.savefig(best_curve_path)
plt.show()
print(f"Curva de perda do melhor fold salvo em: {best_curve_path}")

# --- Relatório Consolidado de Todos os Folds ---
print("\n=== RELATÓRIO CONSOLIDADO ===")
print(f"Número de Folds: {n_splits}")
print(f"Acurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"F1 Score Médio: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"Perda de Validação Média: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
print(f"Melhor Fold: {best_fold_index + 1} (F1 Score: {best_fold_f1:.4f})")

# Salvar resultados em arquivo
results_file = os.path.join(save_dir_base, "resultados_finais.txt")
with open(results_file, 'w') as f:
    f.write("=== RESULTADOS DA VALIDAÇÃO CRUZADA ===\n\n")
    f.write(f"Número de Folds: {n_splits}\n")
    f.write(f"Acurácia Média: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n")
    f.write(f"F1 Score Médio: {mean_f1:.4f} ± {std_f1:.4f}\n")
    f.write(f"Perda de Validação Média: {mean_val_loss:.4f} ± {std_val_loss:.4f}\n")
    f.write(f"Melhor Fold: {best_fold_index + 1} (F1 Score: {best_fold_f1:.4f})\n\n")

    f.write("Acurácias por Fold:\n")
    for i, acc in enumerate(all_val_accuracies, 1):
        f.write(f"Fold {i}: {acc:.4f}\n")

    f.write("\nF1 Scores por Fold:\n")
    for i, f1_score_val in enumerate(all_val_f1_scores, 1):
        f.write(f"Fold {i}: {f1_score_val:.4f}\n")

print(f"Resultados finais salvos em: {results_file}")