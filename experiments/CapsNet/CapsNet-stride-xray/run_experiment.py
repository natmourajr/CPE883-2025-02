# experiments/CapsNet/run_experiment.py

import yaml
import torch
import sys
import os
from datetime import datetime
import shutil
from contextlib import redirect_stdout
# gerar nova imgem docker com o torchinfo
#from torchinfo import summary 
# Adiciona o diretório raiz do projeto ao path do Python para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))

from modules.Evaluation.evaluator import run_kfold_evaluation
from models.CapsNet.capsnet_xray import CapsNetStrided
from models.CapsNet.losses_xray import CapsuleLoss 


def load_config():
    """Carrega o arquivo de configuração da raiz do projeto."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
        
def main():
    # --- 1. SETUP DO EXPERIMENTO ---
    config = load_config()
    model_name = "CapsNet_Strided"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{model_name}/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Artefatos deste experimento serão salvos em: {experiment_dir}")

    # Salva uma cópia da configuração usada para reprodutibilidade
    shutil.copy('config.yaml', os.path.join(experiment_dir, 'config.yaml'))
    

    image_size = config['preprocessing']['image_size']

    # --- 2. SALVAR RESUMO DA ARQUITETURA ---
    try:
        # Passa a seção de config específica da CapsNet
        model_instance = CapsNetStrided(model_config=config['architectures'], num_classes=2)
        batch_size = config['training']['batch_size']
        image_size = config['preprocessing']['image_size']
        input_size = (batch_size, 3, image_size, image_size)
        architecture_summary_path = os.path.join(experiment_dir, 'architecture_summary.txt')
        
        with open(architecture_summary_path, 'w') as f:
            with redirect_stdout(f):
                #summary(model_instance, input_size=input_size, col_names=["input_size", "output_size", "num_params", "trainable"])
                print("torchinfo nao disponivel nesta versao")
        #print(f"Resumo da arquitetura salvo em: {architecture_summary_path}")
        print("torchinfo nao disponivel nesta versao")
    except Exception as e:
        print(f"Não foi possível salvar o resumo da arquitetura: {e}")

    # --- 3. CONFIGURAÇÃO DA PERDA (LOSS) ---
    # 3. LÊ O VALOR DE LAMBDA DO CONFIG, NÃO DEIXA FIXO NO CÓDIGO
    lam_recon_scale = config['architectures']['CapsNet']['lambda_reconstruction']
    lam_recon = lam_recon_scale * (image_size ** 2)
    capsule_loss = CapsuleLoss(lam_recon=lam_recon)

    # --- 4. EXECUÇÃO DA AVALIAÇÃO ---
    results = run_kfold_evaluation(
        model_class=CapsNetStrided, 
        model_name=model_name, 
        config=config,
        experiment_dir=experiment_dir,
        criterion=capsule_loss
    )

    # --- 5. SALVAMENTO DOS RESULTADOS ---
    results_path = os.path.join(experiment_dir, 'summary_results.json')
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    print(f"\nResultados de sumarização salvos em: {results_path}")


if __name__ == '__main__':
    main()
