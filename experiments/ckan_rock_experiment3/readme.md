# 🧪 Experimento CKAN com Convoluções KAN (CKANConv2DReal)

Este experimento investiga o uso de camadas convolucionais baseadas em KAN (`CKANConv2DReal`) para segmentação por pixel de imagens de microtomografia de rochas.

---

# Segmentação de Imagens de Microtomografia com CKAN (Convoluções 3×3 e Splines 2D)

Este projeto tem como objetivo segmentar imagens de microtomografia de rochas em três classes: **fundo**, **rocha** e **poro**, utilizando uma arquitetura baseada em **CKAN** (Kolmogorov–Arnold Networks) com **convoluções 3×3** e **splines 2D**.

---

## 📌 Objetivo

Implementar e avaliar uma arquitetura convolucional com ativações KAN 2D, capaz de segmentar imagens monocromáticas [128×128] por pixel, mantendo a resolução espacial e explorando o poder expressivo das splines aprendíveis.

---

## 🧠 Arquitetura do Modelo

```plaintext
Input: 128×128×1

→ CKANConv2D(1 → 16)        # Conv2D(3×3) + KANActivation(16)
→ Identity (sem pooling)

→ CKANConv2D(16 → 32)       # Conv2D(3×3) + KANActivation(32)
→ Identity (sem pooling)

→ CKANConv2D(32 → 64)       # Conv2D(3×3) + KANActivation(64)
→ Identity (sem pooling)

→ Conv2D(64 → 3)            # Camada final (logits por classe)

Output: 128×128×3
```

Cada bloco CKANConv2D aplica uma convolução com kernel 3×3 seguida de uma ativação não linear baseada em splines 2D aprendíveis.

## 📁 Estrutura

- `rock_seg/kan_conv.py`: Implementa convoluções KAN com `KANLayer` via `F.unfold`
- `rock_seg/model.py`: Arquitetura do modelo de segmentação com múltiplas camadas CKAN convolucionais
- `rock_seg/convolution.py`: Camada de abstração `CKANConv2DReal` que encapsula a lógica de convolução baseada em `KANLayer`
- `rock_seg/rock_dataset_multi.py`: Dataset com resize para `(128, 128)`, retornando imagem `(1, H, W)` e máscara `(H, W)`
- `main.py`: Script de treino, avaliação e visualização de segmentações por época

---


## 🧠 Modelo: CKANSegmentationModel

```python
Input: (B, 1, 128, 128)

Backbone:
  - CKANConv2DReal(1, 16)
  - CKANConv2DReal(16, 32)
  - CKANConv2DReal(32, 64)
  - Conv2d(64, 3, kernel_size=1)

Output: (B, 3, 128, 128)
```

Ativações não-lineares via KANLayer com splines aprendíveis (versão real, pykan==0.2.8)

Treinamento com CrossEntropyLoss + Adam

Visualização intermediária das segmentações por matplotlib

Status
Este experimento é uma evolução direta do Experimento 2, que utilizava convoluções 1×1 com splines 1D. Agora, adotamos filtros 3×3 com splines 2D, aumentando o poder de representação espacial do modelo.

## ⚠️ Observações
Este modelo ainda não está totalmente funcional em GPUs com memória limitada.

Durante os testes, observou-se erro de memória CUDA (out of memory).
Portanto, é necessário ajustar o modelo (por exemplo, reduzindo número de canais, lotes menores, ou otimizando a implementação) antes de treinos mais longos em GPU.

## 📌 Próximos Passos
Reduzir o custo computacional das camadas CKANConv2D

Introduzir pooling para reduzir dimensionalidade e profundidade da rede

Testar variantes tipo U-Net com ativações KAN

Separar conjuntos de treino, validação e teste
