# modules/DataLoader/dataloader.py

import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch

class TuberculosisDataset(Dataset):
    """
    Dataset customizado para o dataset de Raio-X de Tuberculose (Shenzhen).
    - Usa um DataFrame do Pandas para gerenciar os metadados.
    - Realiza a divisão (split) dos dados em treinamento, validação e teste.
    - Permite divisão aleatória com semente (random_seed) para reprodutibilidade.
    """
    def __init__(self, data_dir, transform=None, mode='train', split_ratios=(0.7, 0.15, 0.15), random_seed=42):
        """
        Args:
            data_dir (string): Diretório contendo a pasta 'images'.
            transform (callable, optional): Transformações do torchvision a serem aplicadas.
            mode (string): O conjunto de dados a ser carregado. Opções: 'train', 'val', 'test'.
            split_ratios (tuple): Uma tupla com as proporções para treino, validação e teste.
            random_seed (int): Semente para o gerador de números aleatórios para garantir divisões reprodutíveis.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.mode = mode

        # Carrega todos os metadados em um DataFrame do Pandas
        full_metadata_df = self._load_metadata_as_dataframe()

        # Embaralha e divide o DataFrame
        train_df, val_df, test_df = self._split_dataframe(full_metadata_df, split_ratios, random_seed)

        # Seleciona o DataFrame correto com base no modo
        if self.mode == 'train':
            self.metadata = train_df
            print(f"Modo Treino: {len(self.metadata)} amostras.")
        elif self.mode == 'val':
            self.metadata = val_df
            print(f"Modo Validação: {len(self.metadata)} amostras.")
        else: # 'test'
            self.metadata = test_df
            print(f"Modo Teste: {len(self.metadata)} amostras.")
        
        # Reseta o índice do dataframe para garantir acesso sequencial com __getitem__
        self.metadata = self.metadata.reset_index(drop=True)

    def _load_metadata_as_dataframe(self):
        """ Carrega os metadados e retorna como um DataFrame do Pandas. """
        image_files = os.listdir(self.image_dir)
        data = []
        for file_name in image_files:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                label = int(file_name.split('_')[-1].split('.')[0])
                data.append({'file_name': file_name, 'label': label})
        
        df = pd.DataFrame(data)
        print(f"Total de {len(df)} imagens encontradas e carregadas no DataFrame.")
        return df

    def _split_dataframe(self, df, split_ratios, random_seed):
        """ Embaralha e divide um DataFrame em 3 conjuntos. """
        # Embaralha o DataFrame de forma reprodutível
        df_shuffled = df.sample(frac=1, random_state=random_seed)

        # Calcula os pontos de corte para a divisão
        train_ratio, val_ratio, _ = split_ratios
        train_end_idx = int(len(df_shuffled) * train_ratio)
        val_end_idx = train_end_idx + int(len(df_shuffled) * val_ratio)

        # Divide o DataFrame usando os índices
        train_df = df_shuffled.iloc[:train_end_idx]
        val_df = df_shuffled.iloc[train_end_idx:val_end_idx]
        test_df = df_shuffled.iloc[val_end_idx:]
        
        return train_df, val_df, test_df

    def __len__(self):
        """ Retorna o número de amostras no conjunto de dados (específico do modo). """
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Busca e retorna uma amostra do dataset no índice `idx`. """
        # Pega a linha do DataFrame correspondente ao índice
        sample_info = self.metadata.iloc[idx]
        image_name = sample_info['file_name']
        label = sample_info['label']
        
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {image_path}")
            return None, None

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label