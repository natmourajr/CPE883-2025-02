# Projeto de Segmentação de Imagens de Microtomografia de Rochas

Este projeto tem como objetivo realizar segmentação semântica em imagens de microtomografia de rochas utilizando diferentes arquiteturas de deep learning. O pipeline foi construído de forma modular e experimental, permitindo testes controlados e rastreáveis com diferentes modelos.

```
📁 Estrutura do Projeto
Projeto_segmentacao/
│
├── rock_seg_loader/
│   └── rock_dataset_multi_rev2.py           # DataLoader com suporte multi-amostras e rastreabilidade
│
├── rock_seg_model/
│   ├── caps_rock_seg/                  # CapsNet
│   ├── ckan_rock_seg/                  # CKAN (KANs com convolução)
│   └── deeponet_rock_seg/              # DeepONet (com branch e trunk convolucionais)
│
├── metrics.py                          # Métricas (Dice, IoU)
│
├── main_caps.py                        # Loop de treino para CapsNet
├── main_ckan.py                        # Loop de treino para CKAN
├── main_deeponet.py                    # Loop de treino para DeepONet
│
└── results/
    ├── CapsNet/
    ├── CKAN/
    └── DeepONet/

```

## 📦 Dataset Loader com Rastreamento

O RockSegmentationDatasetMulti foi desenvolvido para facilitar o carregamento de múltiplas amostras, mantendo a rastreabilidade da origem de cada imagem e máscara.

✅ Funcionalidades:

Lê subpastas no formato:

```
dataset/
  DatasetA/
    inputs/
    masks/
  DatasetB/
    inputs/
    masks/
  ...
```

Redimensionamento automático (ex: 512×512)
Normalização das imagens
Retorna nome da amostra (ex: DatasetA_sampleA0001.tiff) — útil para:
Análise de resultados por amostra
Visualização e salvamento com rastreabilidade

## 🧠 Modelos Implementados
### 🧪 1. CapsNet (Segmentação com Cápsulas)

Baseado em cápsulas primárias com função squash e reconstrução via deconvolução.

Primeira camada convolucional extrai características iniciais

Camada PrimaryCaps forma cápsulas com dimensão configurável

Deconvolução + convolução final gera o mapa de classes

📄 Arquivo: rock_seg_model/caps_rock_seg/model_caps.py
⚙️ Configuração: caps_config.py

### 🧪 2. CKAN — Convolutional KANs (Knowledge-Embedded Neural Networks)

Arquitetura composta por camadas convolucionais baseadas em redes KANs, permitindo aprendizado com estruturas funcionais mais interpretáveis.

3 blocos de convolução CKAN (KAN_Convolutional_Layer)

Camada final Conv2d para predição das classes

Identity() substitui pooling para manter dimensionalidade (pode ser alterado)

📄 Arquivo: rock_seg_model/ckan_rock_seg/model_ckan.py
⚙️ Configuração: ckan_config.py

### 🧪 3. DeepONet com Convoluções

Implementação de um DeepONet modificado para imagens, usando redes convolucionais para os ramos branch e trunk, com fusão via produto escalar.

branch_net: convoluções sobre a imagem

trunk_net: também convolucional, processa mesma entrada (ou pode ser outro domínio)

Saída combinada via produto escalar → fc → reshape final para [B, C, H, W]

📄 Arquivo: rock_seg_model/deeponet_rock_seg/model_deeponet.py
⚙️ Configuração: deeponet_config.py

📈 Métricas

As métricas implementadas para avaliação da segmentação são:

Dice Score

IoU (Intersection over Union)

Implementadas no arquivo: metrics.py

#### ▶️ Como Rodar

Para treinar qualquer um dos modelos, execute os arquivos main_*.py. Por exemplo:

python main_caps.py        # Treina o modelo CapsNet
python main_ckan.py        # Treina o modelo CKAN
python main_deeponet.py    # Treina o modelo DeepONet


Certifique-se de ajustar o caminho do dataset em cada script, por exemplo:

root_dir=r"C:\Users\...\mini_dataset"

#### 📊 Resultados

Os resultados de predição e as curvas de perda são salvos automaticamente na pasta results/, organizados por modelo:
```
results/
├── CapsNet/
│   ├── epoch_1/
│   ├── epoch_2/
│   └── loss_curve.png
├── CKAN/
└── DeepONet/
```
Cada imagem salva inclui:
- Imagem original
- Máscara real
- Máscara predita
- Nome da amostra original (para rastreabilidade)

##### 🚧 Status
- [x] Dataset com rastreabilidade por nome
- [x] Implementação de CapsNet
- [x] Implementação de CKAN
- [x] Implementação de DeepONet
- [ ] Implementação do modelo baseline
- [x] Métricas (Dice, IoU)
- [ ] Generalização do loop de treino arquivo main_*.py
- [ ] Plotar curva loss de validação
- [ ] Salvar as predições e métricas de teste

##### 🧪 Requisitos
```
Python ≥ 3.8

PyTorch ≥ 1.12

Numpy, Matplotlib, Scikit-Image, imageio

tqdm
```
