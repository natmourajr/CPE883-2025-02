# experiments/CapsNet/run_experiment.py

import yaml
import torch
import sys
import os
from datetime import datetime
import shutil
from contextlib import redirect_stdout
# Adiciona o diretório raiz do projeto ao path do Python para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.Evaluation.evaluator import run_kfold_evaluation
from models.CapsNet.capsnet_xray import CapsNet
from models.CapsNet.losses_xray import CapsuleLoss 


def load_config():
    """Carrega o arquivo de configuração da raiz do projeto."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    
    model_name = "CapsNet"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{model_name}/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")

    shutil.copy('config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    
    # Salva o resumo da arquitetura em um arquivo de texto
    try:
        model_instance = CapsNet(model_config=config['model'], num_classes=2)
        batch_size = config['training']['batch_size']
        image_size = config['model']['image_size']
        input_size = (batch_size, 3, image_size, image_size)
        architecture_summary_path = os.path.join(experiment_dir, 'architecture_summary.txt')
        
        with open(architecture_summary_path, 'w') as f:
            with redirect_stdout(f):
                summary(model_instance, input_size=input_size, col_names=["input_size", "output_size", "num_params", "trainable"])
        print(f"Resumo da arquitetura salvo em: {architecture_summary_path}")
    except Exception as e:
        print(f"Não foi possível salvar o resumo da arquitetura: {e}")

    # O valor de lam_recon é um hiperparâmetro da CapsNet. O valor do paper é 0.0005, 
    # escalado pela área da imagem.
    lam_recon_scale = 0.0005 
    lam_recon = lam_recon_scale * (config['model']['image_size'] ** 2)
    capsule_loss = CapsuleLoss(lam_recon=lam_recon)

    # Chama a função de avaliação, passando a classe do modelo e a perda customizada
    results = run_kfold_evaluation(
        model_class=CapsNet, 
        model_name=model_name, 
        config=config,
        experiment_dir=experiment_dir,
        criterion=capsule_loss # <-- Passa o objeto de perda customizada
    )

    results_path = os.path.join(experiment_dir, 'summary_results.json')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResultados de sumarização salvos em: {results_path}")


if __name__ == '__main__':
    main()