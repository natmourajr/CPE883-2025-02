# modules/Evaluation/evaluator.py

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os

from modules.DataLoader.dataloader import TuberculosisDataset
from modules.Utils.utils import EarlyStopping
from modules.Preprocessing.transforms import get_image_transforms

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


def run_kfold_evaluation_dry_run(model_class, model_name, config):
    """
    VERSÃO DE TESTE RÁPIDO ("DRY RUN")
    Verifica se a estrutura K-Fold e os DataLoaders estão funcionando,
    sem executar o treinamento completo.
    """
    print(f"\n===== MODO DE TESTE RÁPIDO PARA O MODELO: {model_name} =====")
    
    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    
    k_folds = config['cross_validation']['n_splits']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    # Loop principal da validação cruzada
    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        print("-" * 50)
        print(f"Verificando Fold {fold + 1}/{k_folds}")
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # Apenas as transformações de avaliação são necessárias para este teste
        eval_transforms = get_image_transforms(image_size=config['model']['image_size'], is_train=False)
        train_subset.dataset.transform = eval_transforms
        val_subset.dataset.transform = eval_transforms
        
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        
        print(f"  -> Tamanho do treino: {len(train_subset)}, Tamanho da validação: {len(val_subset)}")
        
        # --- TESTE DE CARREGAMENTO DE DADOS ---
        #Tenta carregar um lote de cada loader para ver se funciona
        try:
            train_images, train_labels = next(iter(train_loader))
            print(f"  -> SUCESSO! Carregou um lote de treino com shape de imagens: {train_images.shape}")
            
            val_images, val_labels = next(iter(val_loader))
            print(f"  -> SUCESSO! Carregou um lote de validação com shape de imagens: {val_images.shape}")
        except Exception as e:
            print(f"  -> ERRO ao tentar carregar um lote de dados: {e}")
            # Se der erro aqui, o problema está no dataloader ou nos dados.
        # --- FIM DO TESTE DE CARREGAMENTO ---

    print("-" * 50)
    print("Teste da estrutura K-Fold concluído com sucesso!")
    
    # Retorna um dicionário vazio pois não calcula métricas
    return {}

    # device define o dispositivo de hardware (CPU, cuda, se configurado)

def run_kfold_evaluation(model_class, model_name, config, experiment_dir):
    print(f"\n===== INICIANDO AVALIAÇÃO K-FOLD PARA O MODELO: {model_name} =====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treinamento: {device}")

    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    k_folds = config['cross_validation']['n_splits']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        fold_num = fold + 1
        print(f"\n--- Fold {fold_num}/{k_folds} ---")

        fold_dir = os.path.join(experiment_dir, f"fold_{fold_num}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_transforms = get_image_transforms(image_size=config['model']['image_size'], is_train=True)
        eval_transforms = get_image_transforms(image_size=config['model']['image_size'], is_train=False)
        
        train_subset.dataset.transform = train_transforms
        val_subset.dataset.transform = eval_transforms
        
        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        
        model = model_class(model_config=config['model'], num_classes=2, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        patience = config['training'].get('early_stopping_patience', 7)
        early_stopper = EarlyStopping(patience=patience, verbose=True, path=os.path.join(fold_dir, 'best_model.pt'))

        epochs = config['training']['epochs']
        for epoch in range(epochs):
            model.train()
            # Loop de treino
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"  Época {epoch + 1}/{epochs}, Perda de Validação: {avg_val_loss:.6f}")

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
                    outputs = model(data)
                    probabilities = F.softmax(outputs, dim=1)[:, 1]
                    y_true.extend(labels.cpu().numpy())
                    y_probs.extend(probabilities.cpu().numpy())
            
            plot_roc_curve(y_true, y_probs, fold_num, set_name, fold_dir)
            auc_score = roc_auc_score(y_true, y_probs)
            fold_metrics[f'auc_{set_name.lower()}'] = auc_score
            print(f"  AUC do conjunto de {set_name}: {auc_score:.4f}")

        fold_results.append(fold_metrics)

    val_aucs = [result['auc_validação'] for result in fold_results]
    
    mean_auc = np.mean(val_aucs)
    std_auc = np.std(val_aucs)
    
    print("\n" + "-"*50)
    print(f"RESULTADO FINAL PARA O MODELO: {model_name}")
    print(f"AUC Médio (Validação): {mean_auc:.4f} (+/- {std_auc:.4f})")
    print("-" * 50)
    
    return {"mean_validation_auc": mean_auc, "std_validation_auc": std_auc}