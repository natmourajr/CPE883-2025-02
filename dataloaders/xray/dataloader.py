# modules/DataLoader/dataloader.py

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class TuberculosisDataset(Dataset):
    """
    Dataset customizado que agora lê os metadados (incluindo idade e gênero)
    de um arquivo CSV.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Diretório que contém a pasta 'images'.
            transform (callable, optional): Transformações a serem aplicadas na imagem.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_dir = os.path.join(self.data_dir, 'images')
        
        metadata_path = 'data/shenzhen_metadata.csv'
        self.metadata = self._load_metadata_from_csv(metadata_path)

    def _load_metadata_from_csv(self, csv_path):
        """
        Carrega os metadados do arquivo CSV e os processa.
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Metadados carregados com sucesso de: {csv_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"ERRO: Arquivo de metadados não encontrado em '{csv_path}'. Verifique o caminho.")

        # Renomeia colunas para consistência interna no projeto
        if 'study_id' in df.columns:
            df = df.rename(columns={'study_id': 'file_name', 'sex': 'gender'})
        
        # Garante que o nome do arquivo inclua a extensão .png se não tiver
        if not df['file_name'].iloc[0].endswith('.png'):
            df['file_name'] = df['file_name'] + '.png'

        # 0 para 'normal', 1 para qualquer outra coisa (tuberculose)
        df['label'] = df['findings'].apply(lambda x: 0 if x == 'normal' else 1)

        print(f"Dataset completo carregado: {len(df)} imagens.")
        
        # Garante que as colunas necessárias existem
        required_cols = ['file_name', 'gender', 'age', 'label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O CSV precisa conter colunas que resultem em: {required_cols}")
            
        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Busca uma amostra completa: imagem e todos os seus metadados.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_info = self.metadata.iloc[idx]
        image_name = sample_info['file_name']
        
        label = torch.tensor(sample_info['label'], dtype=torch.long)
        age = torch.tensor(sample_info['age'], dtype=torch.float32) # Idade como float
        gender = sample_info['gender'] # Gênero como string ('Male'/'Female')
        
        image_path = os.path.join(self.image_dir, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"ERRO GRAVE: A imagem '{image_name}' listada nos metadados não foi encontrada em '{self.image_dir}'")
            # Retorna None para que possa ser filtrado depois, se necessário
            return None 

        if self.transform:
            image = self.transform(image)
        
        # Retorna a imagem e um dicionário com os metadados
        metadata_dict = {
            'label': label,
            'age': age,
            'gender': gender
        }
        
        return image, metadata_dict