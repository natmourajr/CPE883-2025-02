# CapsNet Segmentação — Experimento Inicial com Microtomografia de Rochas

Este repositório contém um **experimento inicial** com **Capsule Networks (CapsNet)** aplicadas à **segmentação de imagens de microtomografia de rochas**.  
O objetivo foi criar um **primeiro protótipo funcional** para servir como base no desenvolvimento do projeto final da disciplina **Tópicos Especiais em Machine Learning**.

---

## 📂 Estrutura do Projeto

```
CapsNet_segmenter.py # Script principal de treino/avaliação
rock_segmentation_dataset_multi.py # Dataloader reutilizado do projeto CKAN
dataset_rochas_mini/ # Mini conjunto de imagens para teste (inputs/ e masks/)
```

---

## 🧠 Modelo

O modelo é baseado em uma arquitetura **SegCaps simplificada**, composta por:
- Camada convolucional inicial (`Conv2D`).
- Cápsulas primárias (`PrimaryCaps`) com squash activation.
- Convolução transposta simples para ajuste de feature maps.
- Camada final (`Conv2D`) para previsão dos logits de cada classe.

---

## 📊 Resultados

### 🔹 Mini Dataset (20 imagens, 128×128)
| Época | Dice Score | IoU Score |
|-------|------------|-----------|
| 1     | 0.2746     | 0.2318    |
| 2     | 0.2726     | 0.2306    |
| 3     | 0.2722     | 0.2305    |

> Com poucas imagens, o modelo apresentou métricas baixas e estáveis, como esperado, servindo apenas para validar o pipeline.

---

### 🔹 Dataset Maior (600 imagens, 128×128)
| Época | Dice Score | IoU Score |
|-------|------------|-----------|
| 1     | 0.4660     | 0.3790    |
| 2     | 0.7217     | 0.6464    |
| 3     | 0.7717     | 0.6960    |

💡 **Interpretação:**
- **Crescimento rápido** das métricas indica que o modelo conseguiu aprender padrões relevantes já nas primeiras épocas.
- **IoU acima de 0.65** é considerado muito bom em segmentação de imagens reais.
- Como o aumento foi rápido, existe risco de **overfitting** se continuar treinando sem validação.

---

## 🚀 Como Rodar
1. Clone este repositório:

```
   git clone https://github.com/<usuario>/<repositorio>.git
   cd <repositorio>
```
   
Certifique-se de ter PyTorch instalado (GPU recomendado):

```
pip install torch torchvision
pip install imageio scikit-image numpy
```

Ajuste o caminho do dataset no script:
```
dataset = RockSegmentationDatasetMulti(
    root_dir="./dataset_rochas_mini",
    output_shape=(128, 128)
)
```

Rode o script:

```
python CapsNet_segmenter.py

```
## 📌 Próximos Passos
- Separar conjunto de treino/validação/teste.

- Salvar máscaras preditas para visualização.

- Testar com dataset completo e resoluções maiores.

- Comparar resultados com Baseline e CKAN.

- Ajustar profundidade e parâmetros do CapsNet.
