# modules/Evaluation/evaluator.py

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import os
import matplotlib.pyplot as plt

from modules.Preprocessing.transforms import get_image_transforms
from modules.Utils.utils import EarlyStopping
from dataloaders.xray.dataloader import TuberculosisDataset

def plot_roc_curve(y_true, y_probs, fold, set_name, save_dir):
    """Gera e salva o gráfico da Curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos (Recall)')
    plt.title(f'Curva ROC - Fold {fold} - Conjunto de {set_name}')
    plt.legend(loc="lower right")
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'roc_curve_{set_name}.png'))
    plt.close()

def run_kfold_evaluation(model_class, model_name, config, experiment_dir, criterion=None):
    print(f"\n===== INICIANDO AVALIAÇÃO K-FOLD PARA O MODELO: {model_name} =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")

    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    k_folds = config['cross_validation']['n_splits']
    
    y_labels = full_dataset.metadata['label'].values
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(np.zeros(len(y_labels)), y_labels)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{k_folds} ---")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=True)
        eval_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=False)
        
        train_subset.dataset.transform = train_transforms
        val_subset.dataset.transform = eval_transforms
        
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        
        # Passa o 'model_config' e o 'device' para a classe do modelo
        model = model_class(model_config=config, num_classes=2, device=device).to(device)
        lr = config['training'].get('learning_rate', 0.001)
        wd = float(config['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        # Usa uma perda customizada se for passada (para CapsNet), senão usa o padrão
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
        
        # Inicializa o Early Stopping para monitorar a perda de validação
        patience = config['training'].get('early_stopping_patience', 7)
        early_stopper = EarlyStopping(patience=patience, verbose=True, path=os.path.join(fold_dir, 'best_model.pt'))

        epochs = config['training']['epochs']
        for epoch in range(epochs):
            model.train()
            # Loop de treino 
            train_loss_sum = 0.0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)

                if "CapsNet" in model_name:
                    labels_one_hot = F.one_hot(labels, num_classes=model.num_classes).float()
                    y_pred, reconstruction = model(data, labels_one_hot)
                    loss = criterion(labels_one_hot, y_pred, data, reconstruction)
                else:
                    y_pred = model(data)
                    loss = criterion(y_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    
                    if "CapsNet" in model_name:
                        labels_one_hot = F.one_hot(labels, num_classes=model.num_classes).float()
                        y_pred, reconstruction = model(data)
                        loss = criterion(labels_one_hot, y_pred, data, reconstruction)
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
            avg_train_loss = train_loss_sum / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Época {epoch + 1}/{epochs}, Perda de Treino: {avg_train_loss:.6f}, Perda de Validação: {avg_val_loss:.6f}")

            # Lógica do Early Stopping
            early_stopper(avg_val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping ativado!")
                break
        
        print("Carregando o melhor modelo salvo para avaliação final do fold...")
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))

        model.eval()
        sets_to_evaluate = {'Treino': train_loader, 'Validação': val_loader}
        fold_metrics = {}

        for set_name, data_loader in sets_to_evaluate.items():
            y_true, y_probs = [], []
            with torch.no_grad():
                for data, labels in data_loader:
                    data, labels = data.to(device), labels.to(device)
                    
                    if "CapsNet" in model_name:
                        y_pred, _ = model(data)
                        probabilities = y_pred[:, 1]
                    else:
                        outputs = model(data)
                        probabilities = F.softmax(outputs, dim=1)[:, 1]
                    
                    y_true.extend(labels.cpu().numpy())
                    y_probs.extend(probabilities.cpu().numpy())
            
            plot_roc_curve(y_true, y_probs, fold_num, set_name, fold_dir)
            auc_score = roc_auc_score(y_true, y_probs)
            fold_metrics[f'auc_{set_name.lower()}'] = auc_score
            print(f"  AUC do conjunto de {set_name}: {auc_score:.4f}")

        fold_results.append(fold_metrics)

    # Análise final e correta dos resultados
    val_aucs = [float(result['auc_validação']) for result in fold_results]
    mean_auc = float(np.mean(val_aucs))
    std_auc  = float(np.std(val_aucs))
    
    print("\n" + "-"*50)
    print(f"RESULTADO FINAL PARA O MODELO: {model_name}")
    print(f"AUC Médio (Validação): {mean_auc:.4f} (+/- {std_auc:.4f})")
    print("-" * 50)
    
    return {"mean_validation_auc": mean_auc, "std_validation_auc": std_auc}
