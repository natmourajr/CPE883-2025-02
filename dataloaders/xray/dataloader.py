# modules/DataLoader/dataloader.py

import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
import torch

class TuberculosisDataset(Dataset):
    """
    Dataset customizado para o dataset de Raio-X de Tuberculose (Shenzhen).
    VERSÃO SIMPLIFICADA PARA VALIDAÇÃO CRUZADA.
    Esta classe agora representa o DATASET COMPLETO. A divisão em treino/validação
    é feita externamente  no módulo Evaluation.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Diretório contendo a pasta 'images'.
            transform (callable, optional): Transformações a serem aplicadas.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.metadata = self._load_metadata_as_dataframe()

    def _load_metadata_as_dataframe(self):
        """ Carrega os metadados e retorna como um DataFrame do Pandas. """
        image_files = os.listdir(self.image_dir)
        data = []
        for file_name in image_files:
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                label = int(file_name.split('_')[-1].split('.')[0])
                data.append({'file_name': file_name, 'label': label})
        
        df = pd.DataFrame(data)
        print(f"Dataset completo carregado: {len(df)} imagens.")
        return df

    def __len__(self):
        """ Retorna o número total de amostras no dataset. """
        return len(self.metadata)

    def __getitem__(self, idx):
        """ Busca e retorna uma amostra do dataset no índice `idx`. """
        sample_info = self.metadata.iloc[idx]
        image_name = sample_info['file_name']
        label = sample_info['label']
        
        image_path = os.path.join(self.image_dir, image_name)
        
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label