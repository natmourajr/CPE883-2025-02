import os
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio.v3 as iio
from skimage.transform import resize

class RockSegmentationDatasetMulti(Dataset):
    """
    Dataset para segmentação de imagens de microtomografia de rochas com múltiplas amostras.
    Retorna: imagem, máscara, nome_da_amostra.

    Estrutura esperada:
    dataset/
        DatasetA/
            inputs/
            masks/
        DatasetB/
            inputs/
            masks/
        ...
    """

    def __init__(self, root_dir, output_shape=(512, 512), transform=None):
        """
        Parâmetros:
        - root_dir: caminho para a pasta contendo as amostras.
        - output_shape: tupla (altura, largura) para redimensionamento.
        - transform: transformação opcional a ser aplicada às amostras.
        """
        self.transform = transform
        self.output_shape = output_shape
        self.samples = []

        # Percorre todas as subpastas (DatasetA, DatasetB, ...)
        for dataset_folder in os.listdir(root_dir):
            dataset_path = os.path.join(root_dir, dataset_folder)
            input_dir = os.path.join(dataset_path, "inputs")
            mask_dir = os.path.join(dataset_path, "masks")

            if not os.path.isdir(input_dir) or not os.path.isdir(mask_dir):
                continue  # Pula se não for válido

            input_files = sorted([
                f for f in os.listdir(input_dir)
                if f.endswith('.tiff') or f.endswith('.tif')
            ])

            for file_name in input_files:
                input_path = os.path.join(input_dir, file_name)
                mask_path = os.path.join(mask_dir, file_name.replace("sample", "mask"))
                if os.path.exists(mask_path):
                    # Armazena também o nome da amostra (ex: "DatasetA")
                    self.samples.append((input_path, mask_path, dataset_folder))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, mask_path, amostra_nome = self.samples[idx]

        # Leitura das imagens
        image = iio.imread(input_path).astype(np.float32) / 255.0  # Normaliza
        mask = iio.imread(mask_path).astype(np.int64)

        # Redimensiona imagem e máscara para output_shape
        image = resize(
            image,
            self.output_shape,
            preserve_range=True,
            anti_aliasing=True
        )
        mask = resize(
            mask,
            self.output_shape,
            order=0,  # Mantém valores inteiros
            preserve_range=True,
            anti_aliasing=False
        ).astype(np.int64)

        # Adiciona canal à imagem
        image = np.expand_dims(image, axis=0)  # [1, H, W]

        if self.transform:
            image, mask = self.transform(image, mask)

        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.long),
            amostra_nome  # <- novo retorno
        )
