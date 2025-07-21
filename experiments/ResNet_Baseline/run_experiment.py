# experiments/ResNet_Baseline/run_experiment.py

import yaml
import torch
import torch.nn as nn
from torchvision import models
import sys
import os

# Adiciona o diretório raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.Evaluation.evaluator import run_kfold_evaluation
from modules.models.ResNet_Baseline.resnetbaseline import ResNetBaseline

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config()

    # Cria uma pasta única para este experimento com base no tempo
    model_name = "ResNet-18_Baseline" 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{model_name}/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")

    # Salva uma cópia da configuração usada para reprodutibilidade
    shutil.copy('config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    
    results = run_kfold_evaluation(
        model_class=ResNetBaseline, 
        model_name=model_name, 
        config=config,
        experiment_dir=experiment_dir 
    )

    results_path = os.path.join(experiment_dir, 'summary_results.json')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResultados de sumarização salvos em: {results_path}")

if __name__ == '__main__':
    main()