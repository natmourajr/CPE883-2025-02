# RockSegmentationDatasetMulti

Dataset customizado para segmentação de imagens de microtomografia de rochas com múltiplas amostras.

## Estrutura de Pastas
```
dataset/
   SampleA/
      inputs/
      masks/
SampleB/
    inputs/
    masks/
 ...  

```
- `inputs/` → imagens originais em formato `.tiff` ou `.tif`
- `masks/` → máscaras correspondentes (mesma resolução)

## Uso
```
from rock_seg_loader.rock_dataset_multi import RockSegmentationDatasetMulti, split_dataset
from torch.utils.data import DataLoader

dataset = RockSegmentationDatasetMulti(root_dir="path/to/dataset", output_shape=(128, 128))
train_set, val_set, test_set = split_dataset(dataset)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
Parâmetros
root_dir: caminho da pasta raiz do dataset.

output_shape: tupla (H, W) para redimensionamento das imagens.

transform: função opcional para aplicar transformações adicionais.

```

## Funções adicinais ao Dataloader anterior
split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42): divide o dataset em treino, validação e teste.
