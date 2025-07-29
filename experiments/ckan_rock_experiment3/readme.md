# 🧪 Experimento CKAN com Convoluções KAN (CKANConv2DReal)

Este experimento investiga o uso de camadas convolucionais baseadas em KAN (`CKANConv2DReal`) para segmentação por pixel de imagens de microtomografia de rochas.

---

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

## ✏️ Observação final
Este repositório representa uma primeira versão de um modelo de segmentação com convoluções KAN. Ainda exige ajustes estruturais e otimizações para ser treinado com sucesso em GPUs convencionais.
