# experiments/CKAN/run_experiment.py

import yaml
import torch
import sys
import os

# Adiciona o diretório raiz do projeto ao path do Python para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..',)))

from modules.Evaluation.evaluator import run_kfold_evaluation
from modules.models.CKAN.ckan import CKAN

def load_config():
    """Carrega o arquivo de configuração da raiz do projeto."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    """
    Ponto de entrada para executar a avaliação K-Fold para a CKAN.
    """
    # Carrega a configuração geral do projeto
    config = load_config()
    
    # Chama a função de avaliação, passando a CLASSE do modelo
    run_kfold_evaluation(
        model_class=CKAN, 
        model_name="CKAN", 
        config=config,
        device="cpu"
    )

if __name__ == '__main__':
    main()