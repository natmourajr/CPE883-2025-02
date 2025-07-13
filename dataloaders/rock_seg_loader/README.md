# RockSegLoader

**Dataloader para segmentação de imagens de microtomografia de rochas.**

Este DataLoader foi desenvolvido como parte da disciplina *Tópicos Especiais em Machine Learning (CPE883 - 2025/2)* e tem como objetivo facilitar o carregamento e o pré-processamento de conjuntos de dados de microtomografia com múltiplas amostras.

---

## 📁 Estrutura esperada dos dados

O diretório contendo os dados deve seguir a seguinte organização:

2_Dataset/
├── DatasetA/
│ ├── inputs/
│ │ ├── sampleA0001.tiff
│ │ ├── ...
│ └── masks/
│ ├── maskA0001.tiff
│ ├── ...
├── DatasetB/
│ ├── inputs/
│ └── masks/
...


Cada pasta representa uma amostra diferente (ex: `DatasetA`, `DatasetB`, ...), com subpastas `inputs/` (imagens) e `masks/` (máscaras multiclasse).

---

## 🧠 Classes esperadas nas máscaras

As máscaras são imagens com valores inteiros representando:

- `0`: fundo  
- `1`: rocha  
- `2`: poro  

---

## ⚙️ Funcionalidades

- Compatível com `torch.utils.data.Dataset`
- Redimensionamento automático das imagens e máscaras para tamanho fixo (default: 512x512)
- Normalização das imagens no intervalo `[0, 1]`
- Máscaras com valores discretos preservados
- Retorna o nome da amostra (ex: `DatasetA`) para rastreabilidade

---

## 🚀 Exemplo de uso

```python
from rock_dataset_multi import RockSegmentationDatasetMulti
from torch.utils.data import DataLoader

dataset_path = "caminho/para/2_Dataset"
dataset = RockSegmentationDatasetMulti(root_dir=dataset_path, output_shape=(512, 512))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for images, masks, amostras in dataloader:
    print(images.shape, masks.shape, amostras)
    break

