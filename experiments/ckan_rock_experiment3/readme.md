### 🧪 Segmentação de Microtomografia com CKAN — Experimento 3
Este experimento aplica redes neurais convolucionais com ativações baseadas em splines (CKAN) para a segmentação de imagens de microtomografia de rochas, classificando cada pixel como fundo, rocha ou poro.



### 🎯 Objetivos
Utilizar camadas convolucionais CKAN para melhorar a expressividade do modelo com poucos parâmetros.

Realizar segmentação em um dataset real, com pré-processamento e visualização integrados.

Executar o experimento em ambiente com GPU local, mesmo com restrição de memória.

### 🧠 Arquitetura do Modelo
A arquitetura é composta por 3 camadas KAN_Convolution com kernel 3x3, seguidas por uma projeção para o número de classes:
```
CKAN_CONFIG = {
    "channels": [1, 4, 8, 16],         # input → ckan1 → ckan2 → ckan3
    "kernel_size": (3, 3),
    "stride": (1, 1),
    "padding": (1, 1),
    "dilation": (1, 1),
    "num_classes": 3,                 # fundo, rocha, poro
    "output_shape": (32, 32)
}
```

### 📁 Estrutura do Projeto

```
ckan_rockSeg/
├── main.py                       # Script principal com treino e visualização
├── rock_seg_loader/
│   └── rock_dataset_multi.py     # Dataloader para múltiplas amostras (com resize, normalização e máscara)
├── rock_seg/
│   ├── model.py                  # Rede CKAN principal
│   ├── kan_conv.py               # Camadas convolucionais KAN (splines 2D)
│   ├── convolution.py            # Operações auxiliares (unfold, pad, convolução)
│   └── config.py                 # Configuração do modelo (CKAN_CONFIG)
├── mini_dataset.zip              # Conjunto de dados reduzido para teste
├── pyproject.toml                # Configuração do ambiente com dependências
└── README.md                     # Documentação do projeto (este arquivo)

```

## 🧠 Arquitetura do Modelo

```plaintext
Input: 128×128×1

→ CKANConv2DReal(1 → 4)        # KAN_Convolutional_Layer(3×3) + KANActivation(16 splines por canal)
→ Identity (sem downsampling)

→ CKANConv2DReal(4 → 8)        # KAN_Convolutional_Layer(3×3) + KANActivation(32 splines por canal)
→ Identity (sem downsampling)

→ CKANConv2DReal(8 → 16)       # KAN_Convolutional_Layer(3×3) + KANActivation(64 splines por canal)
→ Identity (sem downsampling)

→ Conv2D(16 → 3)               # Camada final (logits por classe, sem ativação)

Output: 32×32×3
```

Cada bloco CKANConv2D aplica uma convolução com kernel 3×3 seguida de uma ativação não linear baseada em splines 2D aprendíveis.

### 🖥️ Ambiente de Execução
Python 3.10+

PyTorch ≥ 2.0

pykan==0.2.8

Executado em GPU local (NVIDIA, CUDA 12.0)

### 📊 Resultados Preliminares
Mesmo com poucas épocas de treino e baixa resolução, o modelo está evoluindo no aprendizado indicando mapeamentos coerentes entre poro, rocha e fundo, como no exemplo abaixo:

<img width="1200" height="400" alt="Resultado_preliminar_res-minima_32x32" src="https://github.com/user-attachments/assets/87ad2f73-d586-4f3b-98e6-33bc435aaece" />


<img width="640" height="480" alt="Loss" src="https://github.com/user-attachments/assets/0a40038f-e693-448d-94c6-84e4a913235f" />




### 🔄 Próximos Passos
Substituir a arquitetura base por uma U-Net com blocos CKAN.

Aumentar a resolução das imagens de entrada (de 32×32 para 128×128).

Adicionar métricas quantitativas (IoU, Dice Score).

Avaliar generalização com separação treino/validação/teste.

### 👩‍🔬 Autora
Projeto desenvolvido por Vivian de Carvalho Rodrigues, no contexto da disciplina Tópicos Especiais em Machine Learning (CPE883 - 2025/2), utilizando ferramentas de aprendizado profundo aplicadas a imagens de microtomografia.
