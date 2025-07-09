# experiments/CapsNet/demo_dataloader.py

import yaml
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# Adiciona o diretório raiz do projeto ao path do Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules.DataLoader.dataloader import TuberculosisDataset
from modules.Preprocessing.transforms import get_image_transforms

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def visualize_batch(images, labels, class_names, title):
    plt.figure(figsize=(12, 12))
    plt.suptitle(title, fontsize=16)
    for i in range(min(len(images), 9)):
        ax = plt.subplot(3, 3, i + 1)
        # Atenção: para visualizar corretamente uma imagem normalizada, precisaríamos
        # "desnormalizar". Para manter simples, a imagem pode parecer com cores estranhas.
        img = images[i].permute(1, 2, 0).numpy()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = std * img + mean # Desnormaliza
        img = img.clip(0, 1)   # Garante que os valores fiquem entre 0 e 1
        plt.imshow(img)
        plt.title(f"Label: {class_names[labels[i].item()]}")
        plt.axis("off")
    
    save_path = f"experiments/CapsNet/{title.lower().replace(' ', '_')}_visualization.png"
    plt.savefig(save_path)
    print(f"\nVisualização salva em '{save_path}'")

def main():
    print("Iniciando a demonstração do DataLoader com módulo de pré-processamento...")
    
    # ... (código de carregar config sem alteração) ...
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)
    dataset_config = config['dataset']
    training_config = config['training']
    model_config = config['model']
    
    # --- MUDANÇA NA CRIAÇÃO DAS TRANSFORMAÇÕES ---
    # Agora chamamos nossa função centralizada para obter os pipelines
    train_transforms = get_image_transforms(image_size=model_config['image_size'], is_train=True)
    eval_transforms = get_image_transforms(image_size=model_config['image_size'], is_train=False)
    
    print("\nCriando os datasets de treino, validação e teste...")
    
    # Dataset de Treinamento (usa transformações com data augmentation)
    train_dataset = TuberculosisDataset(
        data_dir=dataset_config['path'], 
        transform=train_transforms, 
        mode='train',
        split_ratios=dataset_config['split_ratios'],
        random_seed=dataset_config['random_seed']
    )
    
    # Datasets de Validação e Teste (usam transformações sem data augmentation)
    val_dataset = TuberculosisDataset(
        data_dir=dataset_config['path'], 
        transform=eval_transforms, 
        mode='val',
        split_ratios=dataset_config['split_ratios'],
        random_seed=dataset_config['random_seed']
    )
    
    test_dataset = TuberculosisDataset(
        data_dir=dataset_config['path'], 
        transform=eval_transforms, 
        mode='test',
        split_ratios=dataset_config['split_ratios'],
        random_seed=dataset_config['random_seed']
    )
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=training_config['batch_size'], shuffle=True, num_workers=training_config['num_workers'])
    print("\n--- Informações dos Conjuntos de Dados ---")
    print(f"Tamanho do conjunto de Treino: {len(train_dataset)} amostras")
    print(f"Tamanho do conjunto de Validação: {len(val_dataset)} amostras")
    print(f"Tamanho do conjunto de Teste: {len(test_dataset)} amostras")

    print("\n--- Verificação de um Lote de Treinamento ---")
    train_images, train_labels = next(iter(train_loader))
    print(f"Formato do lote de imagens de treino: {train_images.shape}")
    
    class_names = {0: 'Normal', 1: 'Tuberculosis'}
    visualize_batch(train_images, train_labels, class_names, title="Batch de Treinamento com Augmentation")


if __name__ == '__main__':
    main()