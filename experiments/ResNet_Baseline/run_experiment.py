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
from modules.models.ResNet.resnetbaseline import ResNetBaseline

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()

    # Chama o avaliador K-Fold, passando a classe do modelo ResNet
    run_kfold_evaluation(
        model_class=ResNetBaseline, 
        model_name="ResNet-18_Baseline", 
        config=config,
        device="cuda"
    )

if __name__ == '__main__':
    main()