# Experimento Template

Isso é um experimento padrão com as seguintes características:

- Depende de um _data loader_ cujo fonte está em `../dataloaders/TemplateLoader`. É instalado como uma dependência padrão de Python
- Publica um script chamado `template-loader`
- Pode ser usado com [uv](https://docs.astral.sh/uv/) ou pip e [pip-tools](https://pip-tools.readthedocs.io/en/latest/)
- Possui um modelo de Dockerfile com suporte a CUDA

## Como criar seu experimento

Crie um diretório para seu novo experimento e copie para lá o conteudo desse diretório molde. Em seguida edite os seguintes pontos:

- Edite o arquivo `pyproject.toml`:
    - Altere o `name` para o nome do experimento. É importante que esse nome seja um [identificador válido em Python](https://docs.python.org/3/reference/lexical_analysis.html#identifiers), de preferência seguindo a [convenção de nomes para módulos e pacotes](https://peps.python.org/pep-0008/#package-and-module-names)
    - Ajuste os campos `version`, `description` conforme necessário
    - Defina a versão do Python em `requires-python`. Essa informação será usada na construção da imagem docker (e também do seu virtualenv se você usar uv). Aqui, dê preferências a especificações que restrinjam uma versão _minor_ do python, tal como `==3.13.*`.
    - Ajuste a definição do script do projeto (ver seção sobre execução local abaixo).
- Escolha a ferramenta para gerenciar dpendências (veja abaixo para mais detalhes):
    - Caso escolha a **uv**, remova o arquivo `requirements.in`.
    - Caso escolha **pip-tools**, edite o `pyproject.toml` para remover a chave `dependencies ` da seção `project` e remova todas as seções que comecem por `tool.uv`. Remova também o arquivo `uv.lock`.
    - Ajuste suas dependências de acordo
- Renomeie o diretório `template_experiment` para o mesmo nome definido no campo `name` do `pyproject.toml`.
- Crie um README.md contendo uma breve descrição do seu experimento, as etapas e como executá-las.

## Declarando dependências

A forma mais moderna de declarar dependências no Python é através do arquivo `pyproject.toml`. No entanto, para dependencias editáveis, como no caso de um data loader ou modelo criado em outra pasta nesse mesmo repositório, ainda são declaradas de maneiras diferentes de acordo com a ferramenta usada para gerenciar os pacotes. Abaixo listarei para o caso de pip + pip-tools e também com o gerenciador uv.

### Usando uv

O [uv](https://docs.astral.sh/uv/) é um gerenciador de pacotes bastante moderno, compatível com a maior parte dos novos padrões do Python. Além disso, é muito rápido e eficiente em espaço. Ele mantém a lista de dependências na chave `project.dependencies` do arquivo `pyproject.toml` e, no caso de pacotes editáveis, na chave `tool.uv.sources.<pacote>`. Esse arquivo de configurações pode ser editato manualmente, mas o uv oferece uma forma mais simples através dos comandos a seguir.

Para remover o pacote `template-loader`:

```bash
uv remove template-loader
```

Para instalar um outro pacote local que, por exemplo, fique em `../../dataloaders/OutroLoader`:

```bash
uv add --editable ../../dataloaders/OutroLoader
```

Sempre que um comando add ou remove for executado, o uv automaticamente atualiza o arquivo `uv.lock` e sincroniza o virtualenv em `./.venv`. No entanto é sempre possível gerar um requirements.txt para manter compatibilidade com o pip clássico:

```bash
uv export --locked --no-hashes -o reqruirements.txt
```

Com esse arquivo gerado, caso você esteja criando um virtualenv do zero em algum lugar que não seja para desenvolvimento, é seguro simplesmente executar `pip install -r requirements.txt` para instalar as dependências certas.

O arquivo gerado terá a flag -e para instalação editável do pacote local.

### Usando pip e pip-tools

O [pip-tools](https://pip-tools.readthedocs.io/en/latest/) trabalha em conjunto com o pip, que é a ferramenta padrão do Python para instalação de pacotes. Apesar de ser uma opção menos moderna, tem compatibilidade garantida com qualquer instalação de Python. O pip-tools publica dois comandos a serem usados em conjunto:

- **pip-compile** - gera um arquivo requirements.txt a partir do arquivo requirements.in, ou do próprio pyproject.toml.
- **pip-sync** - Garante que o seu virtualenv está de acordo com o requirements.txt, instalando ou removendo pacotes conforme necessário.

Como o **pip-compile** não possui uma forma de especificar que uma dependência é editável através do `pyproject.toml`, só nos resta usar o requirements.in. Esse arquivo deve declarar as dependências diretas do projeto, podendo especificar versões. Um exemplo de conteúdo seria:

```pip-requirements
# Arquivo requirements.in

# Dependências diretas
torch
typer>=0.16.0

# Dependências editáveis
-e ../../dataloaders/OutroLoader

# Projeto atual, editável
-e .
```

Com esse arquivo ajustado, basta rodar o seguinte comando para gerar o requirements.txt:

```bash
pip-compile
```

Em seguida, aplique as modificações no seu virtual env com o seguinte comando:

```bash
pip-sync
```

Caso você esteja criando um virtualenv do zero, é seguro simplesmente executar `pip install -r requirements.txt`.

## Executando localmente

Como sugestão, foi criado o arquivo `cli.py` para ser o ponto de entrada para invocações por linha de comando. Diversas outras abordagens podem ser escolhidas e, independente da abordagem, o método de execução deve ser explicado no README do projeto.

Para declarar a existencia desse script foram inseridas as seguintes linhas no `pyproject.toml`:

```toml
[project.scripts]
template-experiment = "template_experiment.cli:app"
```

Essa descrição faz parte do padrão do python e será interpretada pelo `uv` ou pelo `pip install -e .` como definindo que um comando `template-experiment` deve ser criado e, quando invocado, executar `app()` dentro do módulo `template_experiment.cli`. Para invocar usando `uv`:

```bash
uv run template-experiment
```

Se ao invés de uv, você estiver usando pip normal:

```bash
pip install -e . # Isso provavelmente já foi feito pelo seu requirements.txt
template-experiment
```

Também como sugestão, o arquivo `cli.py` usa a biblioteca Typer para criação automática do código de interpretar parâmetros da linha de comando, e gerar automaticamente o help. Isso tem o intuito de facilitar a programação, e não isenta a necessidade de descrever a forma de execução deve no README.md

## Executando via docker

O exemplo inclui um arquivo `Dockerfile` que cria uma imagem baseada na imagem base de CUDA da Nvidia. Com isso, a imagem resultante está preparada para ter métodos que usem GPU. O comando `build_image.sh` facilita o processo de construção dessa imagem, e pode receber uma tag de versão como parâmetro.

O processo de construção dessa imagem precisou ser adaptado para as características desse repositório. Como o experimento atual depende de bibliotecas locais, a imagem precisa ter disponível todo o conteúdo do repositório. Sendo assim, o contexto de construção do projeto precisa ser a raíz do repositório (`../..` em relação ao diretório do experimento), e o Dockerfile precisa ter uma referência para o nome do diretório do experimento. O script `build_image.sh` garante que esses dois requisitos sejam atendidos.

O Dockerfile usa o `uv` para instalação do interpretador Python e dependências. No entanto, ele deve ser compatível mesmo se você não usar o uv para desenvolvimento local, mas gerar um requirements.txt válido conforme descrito acima.

Para execução, assumindo que foi gerada a imagem `templateexperiment:latest`, pode ser executada da seguinte forma interativamente, usando somente CPU:

```bash
docker run -it --rm templateexperiment:latest
```

Para executar com GPU em um sistema contendo o [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-the-nvidia-container-toolkit), basta invocar com:

```bash
docker run -it --rm --gpus all templateexperiment:latest
```
