# 🧪 Experimento: CKAN Simple (Classificação Global)

Este experimento implementa uma pipeline de teste para classificação global de imagens de microtomografia de rochas. A arquitetura é inspirada na CKAN (Kolmogorov–Arnold Network), mas neste estágio inicial utiliza ReLU no lugar de camadas spline.

## 📦 Estrutura

- `rock_class_loader/`: contém o `RockSegmentationDatasetMulti` adaptado para retornar a classe dominante da máscara.
- `ckan_simple_experiment1/cli.py`: script principal com `typer` para treinamento.
- `mini_dataset.zip`: conjunto reduzido de 40 imagens para teste local.
- `pyproject.toml`: dependências e instalação com `uv` ou `pip install -e .`.

## 📈 Arquitetura do modelo
```
[Input 128x128x1]
→ CKANConv2D(1→16) + ReLU
→ MaxPool2D
→ CKANConv2D(16→32) + ReLU
→ MaxPool2D
→ CKANConv2D(32→64) + ReLU
→ AdaptiveAvgPool
→ Flatten → Linear(4096→128) → ReLU → Linear(128→3)
```

- A saída é a **classe dominante da imagem** (0: fundo, 1: rocha, 2: poro).

## 🚀 Execução

```bash
ckan-simple train --data-dir path/para/mini_dataset

```

## 📌 Próximos passos no projeto
Substituir ReLU por KANLayer real
Explorar camadas spline do pacote pykan nas ativações.

Evoluir de classificação para segmentação por pixel
Substituir as FCs finais por Conv2d(64, 3, kernel_size=1)
e treinar o modelo com máscaras completas [B, H, W].

Construir CKAN-U-Net
Arquitetura encoder-decoder com blocos CKANConv2D nas fases de compressão e expansão.

## 🔍 Observações
O pipeline atual está validado na CPU com dataset reduzido.

Ideal para testes rápidos de arquitetura e verificação de integração (dataloader, modelo, treino).
