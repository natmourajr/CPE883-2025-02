import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
from collections import Counter
import warnings

# Importa classes do modelo
from modelo import RoomOccupancyDataset, KANTransformer

# Ignorar avisos do scikit-learn
warnings.filterwarnings('ignore')

# Configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# --- 1. Carregar e Pré-processar o Dataset ---
print("--- 1. Carregando e Pré-processando o Dataset ---")
room_occupancy = fetch_ucirepo(id=864)
X = room_occupancy.data.features
y = room_occupancy.data.targets

df = pd.concat([X, y], axis=1)
print(f"Dataset shape: {df.shape}")

print("\nDistribuição da ocupação:")
print(df['Room_Occupancy_Count'].value_counts().sort_index())

# Features
features = ['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp',
            'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound',
            'S5_CO2', 'S5_CO2_Slope', 'S6_PIR', 'S7_PIR']

X = df[features].values
y = df['Room_Occupancy_Count'].values

# Balancear com SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print("Distribuição após SMOTE:", Counter(y_balanced))

# Normalizar
scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)


# --- Função de treino cross-validation ---
def cross_validation_train(X, y, n_splits=5, epochs=30, batch_size=64):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results = []
    all_train_losses, all_val_losses, all_confusion_matrices, all_f1_scores = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_dataset = RoomOccupancyDataset(X_train, y_train)
        val_dataset = RoomOccupancyDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = KANTransformer(
            input_dim=X.shape[1],
            hidden_dim=128,
            num_layers=4,
            num_classes=len(np.unique(y)),
            dropout=0.2
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # --- Treino ---
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * features.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # --- Validação ---
            model.eval()
            val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * features.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                val_f1 = f1_score(val_targets, val_preds, average='weighted')
                print(f'Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val F1={val_f1:.4f}')

            scheduler.step(val_loss)

        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        cm = confusion_matrix(val_targets, val_preds)

        fold_results.append({
            'fold': fold + 1,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_accuracy': val_accuracy,
            'final_val_f1': val_f1
        })

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_confusion_matrices.append(cm)
        all_f1_scores.append(val_f1)

        print(f"Fold {fold+1}: Acc={val_accuracy:.4f}, F1={val_f1:.4f}")

    return fold_results, all_train_losses, all_val_losses, all_confusion_matrices, all_f1_scores


# --- Executar Cross-Validation ---
fold_results, all_train_losses, all_val_losses, all_confusion_matrices, all_f1_scores = cross_validation_train(
    X_balanced, y_balanced, n_splits=5, epochs=30, batch_size=64
)
