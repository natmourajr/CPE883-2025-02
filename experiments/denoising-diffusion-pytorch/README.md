# Denoising Diffusion MNIST - Experimento 

Este projeto implementa um experimento mínimo viável de Diffusion Models usando o dataset MNIST para geração de imagens, baseado no repositório [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch).

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/888c1d59-4afb-49be-89fa-a0a9911ba58f" />


## Estrutura do projeto

- `denoising_diffusion_pytorch/`: código principal da biblioteca Diffusion Models (clonado do repositório original).
- `mnist_images/`: imagens MNIST exportadas para treino (geradas pelo script).
- `results_mnist/`: imagens geradas e checkpoints do modelo durante o treino.
- `train_mnist_export.py`: script para baixar MNIST, exportar imagens e treinar o modelo.
- `requirements.txt`: lista de dependências para ambiente pip.
- `pyproject.toml`: configuração para Poetry (opcional).

```
denoising-diffusion-pytorch/       # pasta raíz do projeto (clone do repo)
│
├── denoising_diffusion_pytorch/   # código da biblioteca clonado do GitHub
│   ├── __init__.py
│   ├── denoising_diffusion_pytorch.py
│   └── ... (demais módulos)
│
├── mnist_images/                  # pasta criada no export MNIST (gerada em runtime)
│   └── 0_5.png
│   └── 1_0.png
│   └── ...
│
├── results_mnist/                 # resultados do treino gerados pelo Trainer
│
├── train_mnist_export.py          # script principal para baixar, exportar e treinar
├── requirements.txt               # dependências para pip
├── pyproject.toml                 # configuração do poetry (opcional)
└── README.md                     # explicação e instruções do projeto

```

### Pontos chave deste experimento
- 1. Download do MNIST	Exporta as 1000 primeiras imagens para uma pasta (./mnist_images)	✅
- 2. Modelo simples	Um Unet pequeno (dim=32, dim_mults=(1,2,4)) e imagens 32x32 em tons de cinza	✅
- 3. Diffusion simplificado	1000 timesteps, 100 de amostragem, objective='pred_v' (variação estável)	✅
- 4. Treinamento curto	1000 passos, batch de 32, salva a cada 100 steps	✅
- 5. FID desativado	Não tenta calcular métrica FID, ideal para CPU	✅
- 7. Estrutura clara e modular	Separou funções para organização	✅

### Resultados

Resultados apenas para validar que o modelo está aprendendo. Foi utilizado sub-amostram de imagens MNINST. E poucas iterações de treino.

<img width="172" height="172" alt="sample-4" src="https://github.com/user-attachments/assets/00e98ffe-695c-481e-9091-b6ff8e314b0e" />


### Próximos passos:

Com esse baseline funcionando, o projeto será migrado para segmentação de imagens de microtomografia de rochas, evoluindo a arquitetura e adaptando o pipeline.


## Como rodar

1. Crie um ambiente Conda com Python 3.8 ou superior:

```
conda create -n diffusion python=3.8 -y
conda activate diffusion

```

2. Instale as dependências:

```
pip install -r requirements.txt

```

3. Execute o treino:

```
python train_mnist_export.py

```

O script irá baixar o MNIST, exportar as imagens para ./mnist_images e iniciar o treino, salvando resultados em ./results_mnist.

---

### Máquina com GPU CUDA 12.0

Para usar GPU com CUDA 12.0, instale a versão do PyTorch compatível:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu120

```

Depois, reinstale as outras dependências conforme requirements.txt.



### Referências
- Denoising Diffusion Probabilistic Models:  https://arxiv.org/abs/2006.11239
- denoising-diffusion-pytorch - lucidrains: https://github.com/lucidrains/denoising-diffusion-pytorch
- PyTorch-FID: https://github.com/mseitzer/pytorch-fid
