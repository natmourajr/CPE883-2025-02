# experiments/ResNet_Baseline/run_experiment.py

import yaml
import torch
import torch.nn as nn
from torchvision import models
import sys
import os
from datetime import datetime
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
# Adiciona o diretório raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.Evaluation.evaluator import run_kfold_evaluation
from models.ResNet_Xray.resnetbaseline import ResNetBaseline
from dataloaders.xray.dataloader import TuberculosisDataset

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config()
    model_name = "ResNet-18_Baseline" # Mude para cada modelo
    
    # ---  SETUP DO EXPERIMENTO  ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{model_name}/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")
    shutil.copy('config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    
    # --- NOVA LÓGICA: DIVISÃO DO CONJUNTO DE TESTE FINAL (HOLD-OUT) ---
    print("\n--- Separando o conjunto de Teste Final (Hold-Out) ---")
    
    # Carrega os metadados completos
    full_dataset_metadata = TuberculosisDataset(data_dir=config['dataset']['path']).metadata
    
    # Cria as faixas etárias e a "super-categoria" para estratificação
    age_bins = [0, 20, 40, 60, 100]
    age_labels = ['0-20', '21-40', '41-60', '61+']
    full_dataset_metadata['age_group'] = pd.cut(full_dataset_metadata['age'], bins=age_bins, labels=age_labels, right=True).astype(str)
    full_dataset_metadata['stratify_group'] = full_dataset_metadata['gender'] + '_' + full_dataset_metadata['age_group']

    all_indices = list(range(len(full_dataset_metadata)))
    
    # Divide os ÍNDICES em desenvolvimento e teste final, estratificando pela "super-categoria"
    dev_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=100, # Define o tamanho do conjunto de teste final
        random_state=config['dataset']['random_seed'],
        stratify=full_dataset_metadata['stratify_group']
    )
    
    print(f"Dataset dividido: {len(dev_indices)} amostras para Desenvolvimento (Treino/Validação com K-Fold)")
    print(f"Dataset dividido: {len(holdout_indices)} amostras para Teste Final (Hold-Out)")
    

    np.save(os.path.join(experiment_dir, 'holdout_indices.npy'), holdout_indices)
    
    # --- EXECUÇÃO DA AVALIAÇÃO K-FOLD ---
    results = run_kfold_evaluation(
        model_class=ResNetBaseline,
        model_name=model_name, 
        config=config,
        experiment_dir=experiment_dir,
        dev_indices=dev_indices, 
        holdout_indices=holdout_indices 

    )

    results_path = os.path.join(experiment_dir, 'summary_results.json')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResultados de sumarização salvos em: {results_path}")

if __name__ == '__main__':
    main()