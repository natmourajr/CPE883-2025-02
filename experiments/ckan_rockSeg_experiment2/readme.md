# CKAN Rock Segmentation Experiment 2

Este projeto implementa um modelo de segmentação por pixel utilizando redes convolucionais combinadas com camadas de ativação KAN (Kolmogorov–Arnold Networks), aplicadas à segmentação de imagens de microtomografia de rochas em três classes: fundo, rocha e poro.


## 📐 Arquitetura do Modelo

A arquitetura é definida no arquivo [`model.py`](model.py) e combina convoluções `Conv2D` com ativações baseadas em KAN (splines 1D aprendíveis por canal).

### 🧠 Esquema da arquitetura

```

[Input: 128×128×1]
→ CKANConv2D(1 → 16)        # Conv2D(1→16) + KANActivation(16)
→ Identity (sem pooling)

→ CKANConv2D(16 → 32)       # Conv2D(16→32) + KANActivation(32)
→ Identity (sem pooling)

→ CKANConv2D(32 → 64)       # Conv2D(32→64) + KANActivation(64)
→ Identity (sem pooling)

→ Conv2D(64 → 3)            # Camada final para gerar mapa de classes

[Output: 128×128×3]         # Segmentação por pixel (3 classes)

```

🔹 Cada CKANConv2D consiste em uma convolução seguida de uma ativação não-linear KAN.

🔹 O modelo não usa pooling (por enquanto) para preservar a resolução espacial.

🔹 A saída é um tensor de tamanho [B, 3, 128, 128], onde cada pixel é classificado entre fundo, rocha ou poro.

###  📁 Estrutura do Projeto

```
ckan_rockSeg_experiment2/
├── main.py                    # Script principal 
├── rock_seg_loader/
│ └── rock_dataset_multi.py    # Dataloader que carrega múltiplas amostras com 3 classes
├── rock_seg_experiment2/
│ └── model.py                # Arquitetura CKAN para segmentação por pixel
├── mini_dataset.zip          # Subconjunto do dataset para teste
├── pyproject.toml            # Arquivo de configuração do projeto Python
└── README.md # Documentação do experimento

```

### 🧪 Resultado Esperado
Ao final do treinamento, o modelo deve aprender a segmentar corretamente os padrões das imagens de microtomografia, separando fundo, rocha e poro. A visualização por época ajuda a verificar se a segmentação melhora progressivamente.

## 🔭 Próximos Passos

- [ ] Verificar suporte a **KAN com splines 2D** (aplicação espacial)
- [ ] Substituir filtros convolucionais 1×1 por **filtros 3×3**
- [ ] **Plotar curva de erro (loss)** ao longo do treinamento
- [ ] Implementar métricas de validação: **IoU, Dice, Accuracy**
- [ ] Introduzir arquitetura **Encoder/Decoder** (como U-Net)
- [ ] Criar uma versão adaptada da **U-Net com CKAN**

Esses passos visam tornar o modelo mais robusto e preciso para tarefas reais de segmentação em imagens de microtomografia de rochas.

---
