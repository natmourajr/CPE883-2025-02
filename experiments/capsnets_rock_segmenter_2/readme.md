
# SegCapsNet – Segmentação de Rochas

Implementação simples de uma rede CapsNet para segmentação de imagens de microtomografia de rochas.
---

## Arquitetura
1. **Conv2d inicial**: extração de características.
2. **PrimaryCaps**: gera mapas de cápsulas com função squash.
3. **Reorganização**: flatten das cápsulas para processamento 2D.
4. **ConvTranspose2d**: refino e ajuste da resolução.
5. **Conv2d final**: logits por classe.

## Configuração de Treino
- Resolução de entrada/saída: **128×128** (tons de cinza).
- Otimizador: **Adam** (lr=1e-3)
- Função de custo: **CrossEntropyLoss**
- Métricas: Dice Score, IoU.
- Batch size: 4.

## Resultados Preliminares
Com mini-dataset e 3 épocas (600 imagens):
```
Epoch | Dice | IoU
1 | 0.46 | 0.37
2 | 0.72 | 0.64
3 | 0.77 | 0.69
```
<img width="900" height="300" alt="sample_3" src="https://github.com/user-attachments/assets/5682513b-19bd-4e56-bb67-1fea55548b82" />


## Visualização de Predições
Após cada época, o script salva exemplos no diretório `results/` contendo:
- Imagem original
- Máscara real
- Máscara predita

> Obs.: Validação sem shuffle → sempre os mesmos cortes são salvos.
