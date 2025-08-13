# experiments/CapsNet/gradient_sanity_check.py

import yaml
import torch
import torch.nn as nn
import sys
import os

import torch.nn.functional as F

# Adiciona o diretório raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))

from dataloaders.xray.dataloader import TuberculosisDataset
from modules.Preprocessing.transforms import get_image_transforms
from models.CapsNet.capsnet_xray import CapsNet
from models.CapsNet.losses_xray import CapsuleLoss
from torch.utils.data import DataLoader

from modules.Utils.utils import print_grad_stats

def load_config(config_path='config.yaml'):
    """
    # 2. FUNÇÃO SIMPLIFICADA E ROBUSTA
    Carrega o arquivo de configuração. 
    Assume que o script é executado da raiz do projeto (/app no Docker).
    """
    try:
        with open(config_path, 'r') as file:
            print(f"Configuração '{config_path}' carregada com sucesso.")
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"\nERRO: Arquivo de configuração '{config_path}' não encontrado!")
        print(f"O script tentou procurar em: {os.path.abspath(config_path)}")
        print("Certifique-se de que o 'config.yaml' está na raiz do seu projeto.")
        sys.exit(1)

def main():
    print("="*60)
    print("INICIANDO TESTE DE SANIDADE DO GRADIENTE (OVERFITTING EM LOTE ÚNICO)")
    print("="*60)
    
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 1. Prepara o DataLoader para pegar um único lote
    transforms = get_image_transforms(image_size=config['preprocessing']['image_size'], is_train=True)
    full_dataset = TuberculosisDataset(data_dir=config['dataset']['path'], transform=transforms)
    train_loader = DataLoader(full_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    print("\nCarregando um único lote de dados para o teste...")
    single_batch_data, single_batch_labels = next(iter(train_loader))
    single_batch_data = single_batch_data.to(device)
    single_batch_labels = single_batch_labels.to(device)
    print(f"Shape do lote de dados: {single_batch_data.shape}")
    
    # 2. Inicializa o modelo, a perda e o otimizador
    model = CapsNet(model_config=config, num_classes=2, device=device).to(device)
    
    lam_recon = 0.0 # Desativamos a reconstrução para simplificar o teste
    criterion = CapsuleLoss(lam_recon=lam_recon)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training'].get('learning_rate', 0.001))
    
    print("\nIniciando treinamento repetido no mesmo lote por 100 iterações...")
    print("O valor da perda (Loss) deve diminuir drasticamente.")
    print("-" * 60)
    
    # 3. Loop de treinamento no mesmo lote
    for i in range(100):
        model.train()
        
        # Prepara os rótulos em one-hot
        labels_one_hot = F.one_hot(single_batch_labels, num_classes=model.num_classes).float()
        
        # Forward pass
        y_pred, reconstruction = model(single_batch_data, labels_one_hot)
        
        # Cálculo da perda
        loss = criterion(labels_one_hot, y_pred, single_batch_data, reconstruction)
        
        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print_grad_stats(model)

        
        if (i + 1) % 10 == 0:
            print(f"Iteração {i + 1}/100, Perda: {loss.item():.6f}")

    print("-" * 60)
    print("Teste concluído.")


if __name__ == '__main__':
    main()
