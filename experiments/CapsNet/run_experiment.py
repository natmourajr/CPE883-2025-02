# experiments/CapsNet/run_experiment.py

import yaml
import torch
import torch.nn as nn
import sys
import os

# Adiciona o diretório raiz do projeto ao path do Python para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.Evaluation.evaluator import run_kfold_evaluation

# ============================================================
# DEFINA A ARQUITETURACAPSNET AQUI
# Por enquanto, um esqueleto simples para teste.
# ============================================================
class CapsNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CapsNet, self).__init__()
        # Substituir este código pela arquitetura CapsNet real
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)
        print("Modelo CapsNet de Exemplo Inicializado!")

    def forward(self, x):
        return self.fc(self.flatten(x))
# ============================================================

def load_config():
    """Carrega o arquivo de configuração da raiz do projeto."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Ponto de entrada para executar a avaliação K-Fold para a CapsNet.
    """
    # Carrega a configuração geral do projeto
    config = load_config()
    
    # Chama a função de avaliação, passando a CLASSE do modelo
    run_kfold_evaluation(
        model_class=CapsNet, 
        model_name="CapsNet", 
        config=config,
        device="cuda"
    )

if __name__ == '__main__':
    main()