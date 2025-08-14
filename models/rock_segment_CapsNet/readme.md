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
Com mini-dataset e 3 épocas:
```
Epoch | Dice | IoU
1 | 0.46 | 0.37
2 | 0.72 | 0.64
3 | 0.77 | 0.69
```
<img width="900" height="300" alt="sample_3" src="https://github.com/user-attachments/assets/7a815f2f-2691-4f3a-b59f-0d70276b2ec8" />


## Visualização de Predições
Após cada época, o script salva exemplos no diretório `results/` contendo:
