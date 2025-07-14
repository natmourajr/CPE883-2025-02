#import pytest
from rock_dataset_multi import RockSegmentationDatasetMulti
import torch

def test_loader_basic_properties():
    # Caminho para dataset de teste
    dataset = RockSegmentationDatasetMulti(
        root_dir= r"..\fake_dataset",   # <- use este caminho relativo no seu projeto
        output_shape=(512, 512)
    )

    assert len(dataset) == 2, "Dataset deveria ter 2 pares imagem-máscara"

    for i in range(len(dataset)):
        image, mask, amostra_nome = dataset[i]

        # Verificações básicas
        assert isinstance(image, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(amostra_nome, str)

        assert image.shape == (1, 512, 512), f"Shape da imagem inválido: {image.shape}"
        assert mask.shape == (512, 512), f"Shape da máscara inválido: {mask.shape}"

        # Verifica valores esperados na máscara
        assert mask.min() >= 0 and mask.max() <= 2, "Máscara com valores fora do intervalo 0–2"
