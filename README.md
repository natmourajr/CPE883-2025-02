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
   Definidas transformações a serem aplicadas, a construção de um pipeline (ex: [Scikit-Learn Pipeline](https://example.com)), incluindo normalizações, codificações, conversão de tipos, recorte de imagens, padronização de tamanho, etc.

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

### 💻 Exemplo de Uso

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

