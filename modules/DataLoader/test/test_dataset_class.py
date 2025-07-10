#/modules/DataLoader/test/test_dataset_class.py

import yaml
import torch
import matplotlib.pyplot as plt
import sys
import os

# Adiciona o diretório raiz do projeto ao path do Python para encontrar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Importa as classes e funções que vamos testar
from modules.DataLoader.dataloader import TuberculosisDataset
from modules.Preprocessing.transforms import get_image_transforms

def load_config():
    """
    Carrega o arquivo de configuração a partir de um caminho absoluto
    e conhecido dentro do container Docker.
    """
    config_path = '/app/config.yaml'
    
    try:
        with open(config_path, 'r') as file:
            print(f"Configuração '{config_path}' carregada com sucesso.")
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"\nERRO: Arquivo de configuração não encontrado em '{config_path}'!")
        print("Verifique se o arquivo 'config.yaml' realmente existe na raiz do seu projeto.")
        sys.exit(1)

def main():
    """
    Script de teste focado exclusivamente na classe TuberculosisDataset.
    """
    print("--- Iniciando Teste da Classe TuberculosisDataset ---")
    
    # 1. Carrega as configurações necessárias
    config = load_config()
    data_dir = config['dataset']['path']
    image_size = config['model']['image_size']
    
    # 2. Pega o pipeline de transformações (sem data augmentation)
    # is_train=False para um teste consistente
    transforms = get_image_transforms(image_size=image_size, is_train=False)
    
    # 3. Tenta instanciar o Dataset completo
    try:
        print(f"\nInstanciando o dataset a partir de: {data_dir}")
        full_dataset = TuberculosisDataset(data_dir=data_dir, transform=transforms)
        print("-> SUCESSO! Objeto TuberculosisDataset criado.")
    except Exception as e:
        print(f"-> FALHA ao criar o dataset: {e}")
        return

    # 4. Verifica o tamanho total do dataset
    total_images = len(full_dataset)
    print(f"-> Tamanho total do dataset verificado: {total_images} imagens.")
    
    # 5. Pega a PRIMEIRA amostra (índice 0) para testar o __getitem__
    print("\nTestando o método __getitem__ com a primeira amostra (índice 0)...")
    try:
        image, label = full_dataset[0]
        print("-> SUCESSO! Uma amostra foi carregada.")
        
        # 6. Verifica as propriedades da amostra carregada
        print(f"  - Tipo do dado da imagem: {type(image)}")
        print(f"  - Shape do tensor da imagem: {image.shape}") # Deve ser [3, image_size, image_size]
        print(f"  - Tipo do dado do rótulo: {type(label)}")
        print(f"  - Valor do rótulo: {label.item()}")
        
        # 7. Visualiza a amostra
        class_names = {0: 'Normal', 1: 'Tuberculosis'}
        plt.figure(figsize=(6, 6))
        
        # Desnormaliza a imagem para visualização correta
        img_to_show = image.permute(1, 2, 0).numpy()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_to_show = std * img_to_show + mean
        img_to_show = img_to_show.clip(0, 1)
        
        plt.imshow(img_to_show)
        plt.title(f"Primeira Amostra (Índice 0)\nLabel: {class_names[label.item()]}")
        plt.axis("off")
        
        save_path = "modules/DataLoader/test/single_sample_test.png"
        plt.savefig(save_path)
        print(f"\n-> Imagem de teste salva em: '{save_path}'")
        
    except Exception as e:
        print(f"-> FALHA ao carregar a amostra de índice 0: {e}")

    print("\n--- Teste da Classe TuberculosisDataset Concluído ---")


if __name__ == '__main__':
    main()