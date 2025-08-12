# modules/Evaluation/evaluator.py

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from modules.DataLoader.dataloader import TuberculosisDataset
from modules.Preprocessing.transforms import get_image_transforms

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


def run_kfold_evaluation(model_class, model_name, config, device):
    print(f"\n===== INICIANDO AVALIAÇÃO K-FOLD PARA O MODELO: {model_name} =====")
      
    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'])
    
    k_folds = config['cross_validation']['n_splits']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config['dataset']['random_seed'])
    
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_subset.dataset.transform = get_image_transforms(image_size=config['model']['image_size'], is_train=True)
        val_subset.dataset.transform = get_image_transforms(image_size=config['model']['image_size'], is_train=False)

        train_loader = DataLoader(train_subset, batch_size=config['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['training']['batch_size'], shuffle=False)
        
        model = model_class(num_classes=2).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        best_val_accuracy = 0.0
        
        epochs = config['training']['epochs']
        print(f"Treinando por {epochs} épocas...")
        for epoch in range(epochs):
            model.train()
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * correct / total
            print(f"  Época {epoch + 1}/{epochs}, Acurácia de Validação: {val_accuracy:.2f}%")
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
        
        print(f"Melhor acurácia de validação do Fold {fold + 1}: {best_val_accuracy:.2f}%")
        fold_results.append(best_val_accuracy)

    mean_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    print("\n" + "-"*50)
    print(f"RESULTADO FINAL PARA O MODELO: {model_name}")
    print(f"Acurácia Média: {mean_accuracy:.2f}% (+/- {std_accuracy:.2f}%)")
    print("-" * 50)
    
    return {"mean": mean_accuracy, "std": std_accuracy}    