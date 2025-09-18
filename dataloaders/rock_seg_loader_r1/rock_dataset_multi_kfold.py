import os
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
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

    def __init__(self, root_dir, split=None, output_shape=(512, 512), transform=None, seed=42):
        """
        Parâmetros:
        - root_dir: caminho para a pasta contendo as amostras.
        - split: "train", "val", "trainval", "test" ou None (dataset completo).
        - output_shape: tupla (altura, largura) para redimensionamento.
        - transform: transformação opcional a ser aplicada às amostras.
        - seed: semente para garantir reprodutibilidade no split.
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

        # Faz o split somente se solicitado
        if split is not None:
            train_set, val_set, test_set = split_dataset(self.samples, seed=seed)

            if split == "train":
                self.samples = train_set
            elif split == "val":
                self.samples = val_set
            elif split == "trainval":
                self.samples = train_set + val_set  # concatena
            elif split == "test":
                self.samples = test_set
            else:
                raise ValueError(f"Split desconhecido: {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, mask_path, amostra_nome = self.samples[idx]
        filename = os.path.basename(input_path)  # ex: "sampleA0001.tiff"

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
            f"{amostra_nome}_{filename}"  # ex: "DatasetA_sampleA0001.tiff"
        )

# Split treino, validação e teste    
def split_dataset(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Divide uma lista de amostras em train/val/test com base nas proporções.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Soma das proporções deve ser 1"
    total_size = len(samples)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    dataset = list(samples)  # copia
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # random_split retorna Subset, então vamos extrair os índices
    train_samples = [dataset[i] for i in train_set.indices]
    val_samples = [dataset[i] for i in val_set.indices]
    test_samples = [dataset[i] for i in test_set.indices]

    return train_samples, val_samples, test_samples
