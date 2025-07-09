import os
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio.v3 as iio

class RockSegmentationDataset(Dataset):
    """
    Dataset personalizado para segmentação de imagens de tomografia de rocha.
    Lê imagens .tiff em escala de cinza e suas máscaras multiclasse correspondentes.

    Classes:
    - 0: fundo
    - 1: rocha
    - 2: poros
    """

    def __init__(self, root_dir, input_dir='inputs', mask_dir='masks', transform=None):
        """
        Inicializa o dataset.

        Parâmetros:
        - root_dir: caminho da pasta principal contendo subpastas.
        - input_dir: subpasta com imagens de entrada (default='inputs').
        - mask_dir: subpasta com máscaras multiclasse (default='masks').
        - transform: função de transformação (opcional).
        """
        self.input_dir = os.path.join(root_dir, input_dir)
        self.mask_dir = os.path.join(root_dir, mask_dir)
        self.transform = transform

        # Lista de arquivos de entrada
        self.input_files = sorted([
            f for f in os.listdir(self.input_dir)
            if f.endswith('.tiff') or f.endswith('.tif')
        ])

    def __len__(self):
        """
        Retorna o número total de amostras.
        """
        return len(self.input_files)

    def __getitem__(self, idx):
        """
        Retorna uma amostra de dado: (imagem, máscara)
        """
        # Caminhos para imagem e máscara
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        mask_path = os.path.join(self.mask_dir, self.input_files[idx].replace('sample', 'mask'))

        # Leitura das imagens
        image = iio.imread(input_path).astype(np.float32) / 255.0  # Normaliza para [0, 1]
        mask = iio.imread(mask_path).astype(np.int64)  # Máscara com classes: 0, 1, 2

        # Adiciona canal à imagem (de [H, W] para [1, H, W])
        image = np.expand_dims(image, axis=0)

        # Aplica transformações, se houver
        if self.transform:
            image, mask = self.transform(image, mask)

        # Converte para tensores PyTorch
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)