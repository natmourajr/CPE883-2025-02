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
import shutil
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

def save_roc_curve(y_true, y_probs, set_name, save_dir):
    """
    Gera, salva o gráfico da Curva ROC e salva os vetores fpr, tpr.
    Retorna a pontuação AUC.
    """
    if len(np.unique(y_true)) < 2:
        print(f"  AVISO: Pulando a Curva ROC para '{set_name}' (apenas uma classe presente).")
        return np.nan
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    
    # Salva os vetores da curva para plotagem posterior comparativa
    roc_data_path = os.path.join(save_dir, f'roc_vectors_{set_name}.npz')
    np.savez(roc_data_path, fpr=fpr, tpr=tpr, auc=auc)
    print(f"Vetores da Curva ROC para '{set_name}' salvos em: {roc_data_path}")
    
def find_optimal_threshold(y_true, y_probs):
    """
    Encontra o limiar de decisão ótimo usando .
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Ignora o primeiro threshold que pode ser > 1
    sp = np.sqrt(  np.sqrt(tpr*(1-fpr)) * (0.5*(tpr+(1-fpr)))  )
    knee = np.argmax(sp)
    optimal_threshold = thresholds[knee]  
    
    return optimal_threshold   

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

    # Carrega o dataset completo uma vez para referência
    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    
    # Prepara para a estratificação do K-Fold
    dev_labels = full_dataset.metadata['label'].iloc[dev_indices].values
    k_folds = config['cross_validation']['n_splits']
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    fold_results = []
    eval_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=False)

    # ---  Cria o DataLoader para o conjunto 'Operação' (todos os dados de desenvolvimento) UMA VEZ ---
    operacao_dataset = Subset(TuberculosisDataset(data_dir=config['dataset']['path'], transform=eval_transforms), dev_indices)
    operacao_loader = DataLoader(operacao_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    for fold, (train_relative_indices, val_relative_indices) in enumerate(kf.split(np.zeros(len(dev_indices)), dev_labels)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{k_folds} ---")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_indices_abs = [dev_indices[i] for i in train_relative_indices]
        val_indices_abs = [dev_indices[i] for i in val_relative_indices]

        train_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=True)
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
        
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        # ... (Loop de treinamento e validação por época - SEM ALTERAÇÕES) ...
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
        
        plot_loss_curves(train_loss_history, val_loss_history, fold_num, fold_dir)
        print("Carregando o melhor modelo salvo para avaliação final do fold...")
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
        
        fold_metrics = {
            'tempo_treino_seg': end_time - start_time,
            'pico_memoria_mb': torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0
        }

        # --- O conjunto de holdout NÃO é mais avaliado dentro do loop ---
        # --- O conjunto 'Operação' é avaliado aqui ---
        sets_to_evaluate = {'Validação': val_loader, 'Operação': operacao_loader}
        
        # --- Variável para armazenar o limiar ótimo encontrado na validação ---
        optimal_threshold_for_fold = 0.5 # Valor padrão caso a validação falhe

        for set_name, data_loader in sets_to_evaluate.items():
            all_metadata, all_probs = [], []
            model.eval()
            with torch.no_grad():
                iterator = tqdm(data_loader, desc=f"  Avaliação Final {set_name}", unit="batch")
                for data, metadata_batch in iterator:
                    data = data.to(device)
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
            
            set_key = 'validation' if set_name == 'Validação' else 'operacao'

            # Encontra o limiar ótimo no conjunto de VALIDAÇÃO
            if set_name == 'Validação':
                val_y_true = [m['true_label'] for m in all_metadata]
                val_y_probs = all_probs
                if len(np.unique(val_y_true)) > 1:
                    optimal_threshold_for_fold = find_optimal_threshold(val_y_true, val_y_probs)
                    fold_metrics['optimal_threshold'] = float(optimal_threshold_for_fold)
                    print(f"  -> Limiar Ótimo encontrado no conjunto de Validação: {optimal_threshold_for_fold:.4f}")
                else:
                    print("  -> AVISO: Não foi possível calcular o limiar ótimo (apenas uma classe na validação). Usando 0.5.")
                    fold_metrics['optimal_threshold'] = 0.5

            # ... (Análise por subgrupo - SEM ALTERAÇÕES na lógica interna) ...
            age_bins = [0, 40, 60, 120]
            age_labels = ['0-40', '41-60', '61+']
            results_df['age_group'] = pd.cut(results_df['age'], bins=age_bins, labels=age_labels, right=True).astype(str)
            subgroups = {
                "geral": results_df,
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
                    fold_metrics[f'auc_{set_key}_{group_name}'] = auc
                    
                    if group_name == "geral":
                        save_roc_curve(y_true_group, y_probs_group, set_name, fold_dir)
                        threshold = config['training'].get('decision_threshold', 0.5)
                        y_pred_class = (np.array(y_probs_group) >= threshold).astype(int)
                        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_class, labels=[0, 1]).ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        fold_metrics[f'sensitivity_{set_key}'] = float(sensitivity)
                        fold_metrics[f'specificity_{set_key}'] = float(specificity)
                    
                    print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: {auc:.4f} | N: {len(df_group)}")
                else:
                    fold_metrics[f'auc_{set_key}_{group_name}'] = np.nan
                    print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: N/A (dados insuficientes: {len(df_group)})")
        
        fold_results.append(fold_metrics)
        
    # --- FIM DO LOOP DE K-FOLD ---
    
    # ---  Bloco para encontrar o melhor fold com base no desempenho em 'Operação' ---
    print("\n" + "="*60)
    print("SELEÇÃO DO MELHOR MODELO COM BASE NO CONJUNTO DE OPERAÇÃO")
    best_fold_index = np.nanargmax([res.get('auc_operacao_geral', 0.0) for res in fold_results])
    best_fold_num = best_fold_index + 1
    best_model_threshold = fold_results[best_fold_index].get('optimal_threshold', 0.5)
    best_operacao_auc = fold_results[best_fold_index].get('auc_operacao_geral')
    print(f"Melhor desempenho no conjunto 'Operação' foi no Fold {best_fold_num} (AUC Geral: {best_operacao_auc:.4f})")
    print(f"Limiar ótimo deste modelo (encontrado na validação do Fold {best_fold_num}): {best_model_threshold:.4f}")
    print("Este modelo será usado para a avaliação final no conjunto de Hold-Out.")
    
    # --- Bloco para avaliação final do MELHOR MODELO no conjunto Hold-Out ---
    print("\n" + "="*60)
    print("AVALIAÇÃO FINAL DO MELHOR MODELO NO CONJUNTO HOLD-OUT")
    
    # Carrega o modelo campeão
    best_model_path = os.path.join(experiment_dir, f"fold_{best_fold_num}", 'best_model.pt')
    final_model_save_path = os.path.join(experiment_dir, 'best_overall_model.pt')
    shutil.copy(best_model_path, final_model_save_path)
    print(f"Melhor modelo geral salvo em: {final_model_save_path}")
    final_model = model_class(model_config=config, num_classes=2, device=device).to(device)
    final_model.load_state_dict(torch.load(best_model_path))
    final_model.eval()

    # Prepara o dataloader do hold-out
    holdout_transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=False)
    holdout_subset = Subset(TuberculosisDataset(data_dir=config['dataset']['path'], transform=holdout_transforms), holdout_indices)
    holdout_loader = DataLoader(holdout_subset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])
    
    # Executa a avaliação no hold-out
    holdout_metadata, holdout_probs = [], []
    with torch.no_grad():
        iterator = tqdm(holdout_loader, desc="  Avaliação Final no Hold-Out", unit="batch")
        for data, metadata_batch in iterator:
            data = data.to(device)
            if "CapsNet" in model_name:
                # A CapsNet retorna uma tupla (predições, reconstrução)
                # Durante a inferência, só nos importam as predições (primeiro elemento)
                outputs, _ = final_model(data)
                # A saída da CapsNet já representa a probabilidade (norma do vetor)
                probabilities = outputs[:, 1]
            else:
                # Para ResNet, CKAN, etc., que retornam um único tensor de logits
                outputs = final_model(data)
                probabilities = F.softmax(outputs, dim=1)[:, 1]
            holdout_probs.extend(probabilities.cpu().numpy())
            for i in range(len(data)):
                holdout_metadata.append({
                    'true_label': metadata_batch['label'][i].item(),
                    'age': metadata_batch['age'][i].item(),
                    'gender': metadata_batch['gender'][i]
                })

    results_df_holdout = pd.DataFrame(holdout_metadata)
    results_df_holdout['probability'] = holdout_probs
    
    final_holdout_metrics = {}
    
    # Análise por subgrupo no Hold-Out
    results_df_holdout['age_group'] = pd.cut(results_df_holdout['age'], bins=[0, 40, 60, 120], labels=['0-40', '41-60', '61+'], right=True).astype(str)
    subgroups_holdout = {
        "geral": results_df_holdout,
        "genero_masculino": results_df_holdout[results_df_holdout['gender'] == 'Male'],
        "genero_feminino": results_df_holdout[results_df_holdout['gender'] == 'Female'],
        "idade_0_40": results_df_holdout[results_df_holdout['age_group'] == '0-40'],
        "idade_41_60": results_df_holdout[results_df_holdout['age_group'] == '41-60'],
        "idade_61+": results_df_holdout[results_df_holdout['age_group'] == '61+'],
    }
    
    print("\n--- Resultados de Performance no Conjunto Hold-Out ---")
    holdout_dir = os.path.join(experiment_dir, "holdout_results")
    os.makedirs(holdout_dir, exist_ok=True)
    
    for group_name, df_group in subgroups_holdout.items():
        y_true_group = df_group['true_label'].values
        y_probs_group = df_group['probability'].values
        
        if len(df_group) > 0 and len(np.unique(y_true_group)) > 1:
            auc = roc_auc_score(y_true_group, y_probs_group)
            final_holdout_metrics[f'auc_holdout_{group_name}'] = auc
            
            if group_name == "geral":
                plot_roc_curve(y_true_group, y_probs_group, "Final", "Hold-Out", holdout_dir)
                save_roc_curve(y_true_group, y_probs_group, "Hold-Out", holdout_dir)
                print(f"  -> Usando limiar de {best_model_threshold:.4f} para métricas de classificação no Hold-Out.")
                y_pred_class = (np.array(y_probs_group) >= best_model_threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_class, labels=[0, 1]).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                final_holdout_metrics[f'sensitivity_holdout'] = float(sensitivity)
                final_holdout_metrics[f'specificity_holdout'] = float(specificity)
            
            print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: {auc:.4f} | N: {len(df_group)}")
        else:
            final_holdout_metrics[f'auc_holdout_{group_name}'] = np.nan
            print(f"  -> {group_name.replace('_', ' ').title():<20} | AUC: N/A (dados insuficientes: {len(df_group)})")
            
    # ---  O retorno da função agora inclui o sumário da validação cruzada E os resultados finais do hold-out ---
    
    # Sumariza os resultados da validação cruzada (como antes)
    cv_summary_dict = {}
    metrics_to_summarize = [key for key in fold_results[0].keys()] # Pega todas as métricas do primeiro fold
    
    for key in metrics_to_summarize:
        values = [res.get(key, np.nan) for res in fold_results]
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        cv_summary_dict[f'mean_{key}'] = float(mean_val)
        cv_summary_dict[f'std_{key}'] = float(std_val)

    # Combina os dois dicionários em um resultado final
    final_results = {
        'cross_validation_summary': cv_summary_dict,
        'final_holdout_results': final_holdout_metrics
    }

    print("\n" + "="*60)
    print(f"SUMÁRIO FINAL PARA O MODELO: {model_name}")
    print("Métricas abaixo são a MÉDIA (+/- DESV. PADRÃO) dos resultados da Validação Cruzada.")
    print(f"AUC Operação Geral (Média CV): {cv_summary_dict['mean_auc_operacao_geral']:.4f} (+/- {cv_summary_dict['std_auc_operacao_geral']:.4f})")
    print(f"AUC Validação Geral (Média CV): {cv_summary_dict['mean_auc_validation_geral']:.4f} (+/- {cv_summary_dict['std_auc_validation_geral']:.4f})")
    print("="*60)
    
    return final_results