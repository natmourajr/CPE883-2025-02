
# Modelo em construção

Este modelo ainda não está totalmente funcional em GPUs com memória limitada.

Durante os testes, observou-se erro de memória CUDA (out of memory).
Portanto, é necessário ajustar o modelo (por exemplo, reduzindo número de canais, lotes menores, ou otimizando a implementação) antes de treinos mais longos em GPU.

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

## 🗂️ Formato dos Dados
Input: imagens monocromáticas

Shape: [B, 1, 128, 128]

Máscaras (target): segmentações com rótulo por pixel

Shape: [B, 128, 128]

Valores inteiros: {0 = fundo, 1 = rocha, 2 = poro}

Saída esperada do modelo:

Shape: [B, 3, 128, 128]

Representa os logits por classe em cada pixel

### ⚙️ Treinamento
Loss function: nn.CrossEntropyLoss()

Otimizador: Adam

Dispositivo atual: CPU ou GPU (quando disponível)
