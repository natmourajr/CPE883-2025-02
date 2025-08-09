# CPE883-2025-02
Repositório para ser utilizado para a disciplina do programa de engenharia elétrica da Coppe CPE883 Tópicos Especiais em Aprendizado De Máquina. Professor: Natanael Nunes de Moura Junior


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/natmourajr/CPE883-2025-02.svg?style=for-the-badge
[contributors-url]: https://github.com/natmourajr/CPE883-2025-02/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/natmourajr/CPE883-2025-02.svg?style=for-the-badge
[forks-url]: https://github.com/natmourajr/CPE883-2025-02/network/members
[stars-shield]: https://img.shields.io/github/stars/natmourajr/CPE883-2025-02.svg?style=for-the-badge
[stars-url]: https://github.com/natmourajr/CPE883-2025-02/stargazers
[issues-shield]: https://img.shields.io/github/issues/natmourajr/CPE883-2025-02.svg?style=for-the-badge
[issues-url]: https://github.com/natmourajr/CPE883-2025-02/issues
[license-shield]: https://img.shields.io/github/license/natmourajr/CPE883-2025-02.svg?style=for-the-badge
[license-url]: https://github.com/natmourajr/CPE883-2025-02/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: www.linkedin.com/in/natanael-moura-junior-425a3294


## Ementa do curso


### Pipeline de Carregamento de Dados (Dataloader)

Este projeto conta com um módulo de **dataloader** desenvolvido para fornecer dados de forma eficiente, escalável e reprodutível aos modelos de aprendizado de máquina nas etapas de treinamento, validação e teste.

#### 🔍 Objetivo

O objetivo do dataloader é automatizar e otimizar o processo de ingestão de dados, garantindo:
- Leitura eficiente de grandes volumes de dados
- Pré-processamento em tempo real
- Geração de lotes (batches) compatíveis com os frameworks de ML utilizados
- Controle sobre a aleatoriedade e reprodutibilidade dos experimentos
- Flexibilidade para diferentes formatos e tipos de dados (imagens, séries temporais, texto, etc.)

#### 🛠️ Processo de Desenvolvimento

O desenvolvimento do dataloader seguiu as seguintes etapas:

1. **Análise Estrutural dos Dados**  
   Diferentes formatos devem ser contemplados durante o desenvolvimento (ex.: `.csv`, `.json`, `.jpg`, `.npy`). Além disso, variáveis envolvidas e a necessidade de tratamento de dados ausentes, normalização e categorização de alvos devem ser avaliadas.
   Obs: neste ponto, tipicamente é interessante ter um conhecimento mais aprofundado dos dados e começar o desenvolvimento de uma EDA (Exploratory Data Analysis)

2. **Construção da Pipeline de Pré-processamento**  
   Definidas transformações a serem aplicadas, a construção de um pipeline (ex: [Scikit-Learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)), incluindo normalizações, codificações, conversão de tipos, recorte de imagens, padronização de tamanho, etc.

3. **Implementação Modular**  
   Utilizando APIs do framework principal (ex.: `torch.utils.data.Dataset`), o dataloader deve ser estruturado para permitir:
   - Carregamento preguiçoso (*lazy loading*)
   - Execução paralela via `num_workers` (paralelismo)
   - Amostragem estratificada (implementação de processo de Validação Cruzada)
   - Configuração flexível por meio de arquivos `.yaml` ou `.json`

4. **Testes e Verificações**  
   O desenvolvimento deve contemplar testes automatizados e verificações manuais de integridade, incluindo distribuição de classes, consistência de rótulos e validação do pipeline de transformação.

5. **Reprodutibilidade e Versionamento**  
   Integração com ferramentas como DVC (Data Version Control) e controle de semente (`random seed`) para garantir a reprodutibilidade dos experimentos.

#### 💻 Exemplo de Uso

```python
from dataloader import CustomDataset
from torch.utils.data import DataLoader

dataset = CustomDataset(
    data_dir="dados/imagens",
    transform=transformacoes_padrao
)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

```
#### 📁 Formatos de Dados Suportados
1. Imagens: .jpg, .png, .tiff
2. Dados tabulares: .csv, .xlsx, .parquet
3. Séries temporais: .npy, .hdf5
4. Dados anotados: .json, .xml

#### 🚧 Melhorias Futuras
1. Suporte a carregamento em tempo real via streaming
2. Integração com armazenamento em nuvem (AWS S3, Google Cloud, etc.)
3. Aumento de dados (data augmentation) com técnicas avançadas
4. Cache inteligente para acelerar a preparação dos lotes

### Modelos a serem estudados

A rápida evolução das técnicas de aprendizado de máquina tem impulsionado avanços significativos em diversas áreas da engenharia, ciência e tecnologia. No entanto, muitos dos modelos mais recentes e inovadores ainda não foram plenamente incorporados às disciplinas tradicionais dos programas de pós-graduação, o que limita a formação de profissionais e pesquisadores frente ao estado da arte. Com base nessa lacuna, esta disciplina do Programa de Engenharia Elétrica da Coppe/UFRJ, no âmbito da área de Inteligência Computacional, tem por objetivo apresentar, discutir criticamente e aprofundar o estudo de modelos emergentes de aprendizado de máquina que representam as novas fronteiras do conhecimento e da pesquisa aplicada.

Diferenciando-se das demais disciplinas do programa, este curso busca expor os alunos a abordagens contemporâneas que têm ganhado destaque na literatura científica internacional, mas que ainda são pouco exploradas no currículo regular. Entre os temas centrais estão os Kolmogorov-Arnold Networks (KAN), redes neurais baseadas em decomposição funcional e interpretabilidade; os Diffusion Probabilistic Models, que reformulam a geração de dados sintéticos por meio de processos estocásticos reversíveis; as Capsule Networks, que introduzem hierarquias estruturais para superar limitações de invariância em redes convolucionais tradicionais; os Attention Models, base conceitual para arquiteturas como os Transformers, com grande impacto em processamento de linguagem natural e visão computacional; e os Neural Operators (como os DeepONets), que representam uma nova classe de modelos capazes de aprender operadores em espaços funcionais, com aplicações promissoras em modelagem física e simulações científicas.

Além de fornecer uma base teórica sólida sobre os princípios matemáticos e computacionais que sustentam esses modelos, a disciplina incentivará a experimentação prática, análise crítica de publicações recentes e o desenvolvimento de projetos aplicados. Ao final do curso, espera-se que os alunos estejam capacitados a compreender o funcionamento interno desses modelos, avaliar suas vantagens e limitações, e aplicá-los de forma inovadora em suas respectivas áreas de pesquisa.

#### KAN: Kolmogorov-Arnold Networks
1. [KAN: Kolmogorov-Arnold Networks (artigo base)](https://arxiv.org/abs/2404.19756)
2. [A Survey on Kolmogorov-Arnold Network](https://arxiv.org/abs/2411.06078)
3. [Convolutional Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.13155)
4. [KAN-ODEs: Kolmogorov–Arnold network ordinary differential equations for learning dynamical systems and hidden physics](https://www.sciencedirect.com/science/article/pii/S0045782524006522)
5. [Kolmogorov-Arnold Graph Neural Networks](https://arxiv.org/abs/2406.18354)
6. [Kolmogorov-Arnold Networks (KANs) for Time Series Analysis](https://arxiv.org/abs/2405.08790)
7. [Kolmogorov-Arnold Networks are Radial Basis Function Networks](https://arxiv.org/abs/2405.06721)
8. [Kolmogorov-Arnold Transformer](https://arxiv.org/abs/2409.10594)
9. [seqKAN: Sequence processing with Kolmogorov-Arnold Networks](https://arxiv.org/abs/2502.14681)
10. [TKAN: Temporal Kolmogorov-Arnold Networks](https://arxiv.org/abs/2405.07344)

#### Diffusion Probabilistic Models
1. [Diffusion Probabilistic Models (artigo base)](https://arxiv.org/abs/2006.11239)

#### Capsule Networks
1. [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829v2)

#### Attention Models
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

#### Neural Operators
1. [DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193)

### Estrutura do repositório "main"
```
├── environment.yml
├── dataloaders
│       │   ├── tabular
│       │   ├── serial
│       │   ├── imagem

├── experiments
│       │   ├── tabular
                 ├── ...
                 ├── ...
│       │   ├── serial
                 ├── ...
                 ├── ... 
│       │   ├── imagem
                 ├── ... 
├── models
│       │   ├── KAN
                ├── ...
                ├── ...
│       │   ├── Diffusion Probabilistic Models
                ├── ...
                ├── ...
│       │   ├── Capsule Networks
                ├── ...
                ├── ...
│       │   ├── Attention Models
│       │   ├── Neural Operators
├── modules_processing
│   ├── ...
│   └── ...
├── utils
│   └── utils.py
└── setup.py
```

No diretório Experiments, deve ser armazenaodo o experimento com cada modelo. Portanto, na pasta "experiments/tabular/", devem ser adicionados os experimentos com modelos que processem dados tabulares.

A pasta "modules_processing/" contém definições de classes que criam tudo o que é usado para criar um experimento, ou seja, contém todas as classes utilitárias. Algumas delas são:
      modules_processing/feature\_mappers.py: Define as classes que extraem os atributos de um determinado experimento. A implementação utiliza o Torch para tornar a extração mais rápida ao utilizar muitos dados. Na versão atual, algumas estratégias já foram implementadas, por exemplo:


### Trabalho Final
