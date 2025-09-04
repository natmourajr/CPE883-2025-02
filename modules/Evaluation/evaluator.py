# modules/Evaluation/evaluator.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
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

def plot_loss_curves(train_history, val_history, fold, save_dir):
    """Gera e salva o gráfico das curvas de perda de treino e validação."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label='Perda de Treino')
    plt.plot(val_history, label='Perda de Validação')
    plt.title(f'Curvas de Perda - Fold {fold}')
    plt.xlabel('Épocas')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
def run_kfold_evaluation(model_class, model_name, config, experiment_dir, dev_indices, holdout_indices, criterion=None):
    print(f"\n===== INICIANDO AVALIAÇÃO K-FOLD PARA O MODELO: {model_name} =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")

    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    
    dev_labels = full_dataset.metadata['label'].iloc[dev_indices].values
    k_folds = config['cross_validation']['n_splits']
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    fold_results = []

    # Cria o DataLoader para o conjunto de teste final (hold-out) UMA VEZ
    holdout_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=False)
    holdout_subset = Subset(TuberculosisDataset(data_dir=config['dataset']['path'], transform=holdout_transforms), holdout_indices)
    holdout_loader = DataLoader(holdout_subset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    for fold, (train_relative_indices, val_relative_indices) in enumerate(kf.split(np.zeros(len(dev_indices)), dev_labels)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{k_folds} ---")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        # Converte os índices relativos do K-Fold para os índices absolutos do dataset original
        train_indices_abs = [dev_indices[i] for i in train_relative_indices]
        val_indices_abs = [dev_indices[i] for i in val_relative_indices]
        
        # Cria os DataLoaders para o fold atual
        train_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=True)
        eval_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=False)
        train_dataset_fold = Subset(TuberculosisDataset(data_dir=config['dataset']['path'], transform=train_transforms), train_indices_abs)
        val_dataset_fold = Subset(TuberculosisDataset(data_dir=config['dataset']['path'], transform=eval_transforms), val_indices_abs)
        train_loader = DataLoader(train_dataset_fold, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'])
        val_loader = DataLoader(val_dataset_fold, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], drop_last=True)

        model = model_class(model_config=config, num_classes=2, device=device).to(device)
        
        lr = config['training'].get('learning_rate', 0.001)
        wd = float(config['training'].get('weight_decay', 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        patience = config['training'].get('early_stopping_patience', 7)
        early_stopper = EarlyStopping(patience=patience, verbose=True, path=os.path.join(fold_dir, 'best_model.pt'))
        
        train_loss_history, val_loss_history = [], []
        
        # Medição de Custo Computacional
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        epochs = config['training']['epochs']
        for epoch in range(epochs):
            model.train()
            train_loss_sum = 0.0
            train_iterator = tqdm(train_loader, desc=f"  Treino Época {epoch + 1}/{epochs}", unit="batch")
            for data, metadata_batch in train_iterator:
                data, labels = data.to(device), metadata_batch['label'].to(device)
                if "CapsNet" in model_name:
                    labels_one_hot = F.one_hot(labels, num_classes=2).float()
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
            val_loss_sum = 0.0
            val_iterator = tqdm(val_loader, desc=f"  Validação Época {epoch + 1}/{epochs}", unit="batch")
            with torch.no_grad():
                for data, metadata_batch in val_iterator:
                    data, labels = data.to(device), metadata_batch['label'].to(device)
                    if "CapsNet" in model_name:
                        labels_one_hot = F.one_hot(labels, num_classes=2).float()
                        y_pred, reconstruction = model(data)
                        loss = criterion(labels_one_hot, y_pred, data, reconstruction)
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, labels)
                    val_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)
            avg_val_loss = val_loss_sum / len(val_loader)
            train_loss_history.append(avg_train_loss)
            val_loss_history.append(avg_val_loss)
            print(f"  Época {epoch + 1}/{epochs} -> Perda Treino: {avg_train_loss:.6f} | Perda Validação: {avg_val_loss:.6f}")

            early_stopper(avg_val_loss, model)
            if early_stopper.early_stop:
                print("Early stopping ativado!")
                break
        
        end_time = time.time()
        training_time_seconds = end_time - start_time
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        
        plot_loss_curves(train_loss_history, val_loss_history, fold_num, fold_dir)
        print("Carregando o melhor modelo salvo para avaliação final do fold...")
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
        
        fold_metrics = {
            'tempo_treino_seg': training_time_seconds,
            'pico_memoria_mb': peak_memory_mb
        }

        sets_to_evaluate = {'Validação': val_loader, 'Teste Final': holdout_loader}
        for set_name, data_loader in sets_to_evaluate.items():
            all_metadata, all_probs = [], []
            with torch.no_grad():
                iterator = tqdm(data_loader, desc=f"  Avaliação Final {set_name}", unit="batch")
                for data, metadata_batch in iterator:
                    data, labels = data.to(device), metadata_batch['label'].to(device)
                    if "CapsNet" in model_name:
                        y_pred, _ = model(data)
                        probabilities = y_pred[:, 1]
                    else:
                        outputs = model(data)
                        probabilities = F.softmax(outputs, dim=1)[:, 1]
                    all_probs.extend(probabilities.cpu().numpy())
                    for i in range(len(data)):
                        all_metadata.append({
                            'true_label': metadata_batch['label'][i].item(),
                            'age': metadata_batch['age'][i].item(),
                            'gender': metadata_batch['gender'][i]
                        })
            results_df = pd.DataFrame(all_metadata)
            results_df['probability'] = all_probs

            # Define a chave padronizada para este conjunto ('validation' ou 'holdout')
            set_key = 'validation' if set_name == 'Validação' else 'holdout'

            # Análise por Subgrupo
            age_bins = [0, 40, 60, 120]
            age_labels = ['0-40', '41-60', '61+']
            results_df['age_group'] = pd.cut(results_df['age'], bins=age_bins, labels=age_labels, right=True).astype(str)
            subgroups = {
                "geral": results_df, # Chave simplificada
                "genero_masculino": results_df[results_df['gender'] == 'Male'],
                "genero_feminino": results_df[results_df['gender'] == 'Female'],
                "idade_0_40": results_df[results_df['age_group'] == '0-40'],
                "idade_41_60": results_df[results_df['age_group'] == '41-60'],
                "idade_61+": results_df[results_df['age_group'] == '61+'],
            }
            
            print(f"\n--- Análise de Performance por Subgrupo ({set_name}) - Fold {fold_num} ---")
            for group_name, df_group in subgroups.items():
                y_true_group = df_group['true_label'].values
                y_probs_group = df_group['probability'].values
                
                if len(df_group) > 0 and len(np.unique(y_true_group)) > 1:
                    auc = roc_auc_score(y_true_group, y_probs_group)
                    # Salva a AUC do subgrupo com uma chave padronizada
                    fold_metrics[f'auc_{set_key}_{group_name}'] = auc
                    
                    # Para o grupo "Geral", calcula e salva as outras métricas
                    if group_name == "geral":
                        plot_roc_curve(y_true_group, y_probs_group, fold_num, set_name, fold_dir)
                        threshold = config['training'].get('decision_threshold', 0.5)
                        y_pred_class = (np.array(y_probs_group) >= threshold).astype(int)
                        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_class, labels=[0, 1]).ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        fold_metrics[f'sensitivity_{set_key}'] = sensitivity
                        fold_metrics[f'specificity_{set_key}'] = specificity
                    
                    print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: {auc:.4f} | N: {len(df_group)}")
                else:
                    fold_metrics[f'auc_{set_key}_{group_name}'] = np.nan
                    print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: N/A (dados insuficientes: {len(df_group)})")
        
        fold_results.append(fold_metrics)
    # --- Análise Final Consolidada ---
    print("\n" + "="*60)
    print(f"RESULTADO FINAL CONSOLIDADO PARA O MODELO: {model_name}")

    final_summary_dict = {}
    metrics_to_summarize = [
        # Métricas de Validação
        'auc_validation_geral', 'sensitivity_validation', 'specificity_validation',
        'auc_validation_genero_masculino', 'auc_validation_genero_feminino',
        'auc_validation_idade_0_40', 'auc_validation_idade_41_60', 'auc_validation_idade_61+',
        # Métricas do Teste Final
        'auc_holdout_geral', 'sensitivity_holdout', 'specificity_holdout',
        'auc_holdout_genero_masculino', 'auc_holdout_genero_feminino',
        'auc_holdout_idade_0_40', 'auc_holdout_idade_41_60', 'auc_holdout_idade_61+',
        # Métricas de Custo
        'tempo_treino_seg', 'pico_memoria_mb'
    ]

    for key in metrics_to_summarize:
        values = [res.get(key, np.nan) for res in fold_results]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        
        # Imprime apenas as métricas gerais principais, para não poluir o log final
        if 'geral' in key or 'sensitivity' in key or 'specificity' in key or 'tempo' in key or 'pico' in key:
            print(f"  -> {key.replace('_', ' ').title():<40}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        final_summary_dict[f'mean_{key}'] = float(mean_val)
        final_summary_dict[f'std_{key}'] = float(std_val)

    print("=" * 60)

    return final_summary_dict
