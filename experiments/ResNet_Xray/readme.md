# Experimento: ResNet_Xray

Este diretório contém o script principal para executar o pipeline completo de treinamento e avaliação para a arquitetura **CKAN** no dataset de Tuberculose.

## ⚙️ Parâmetros da Arquitetura (config.yaml)

Este modelo é configurado dinamicamente a partir do arquivo `config.yaml` localizado na raiz do projeto. Os parâmetros específicos para esta arquitetura, encontrados sob a chave `architectures:`, são:

```yaml
# ===================================================================
# 1. PARÂMETROS DE DADOS E VALIDAÇÃO
# ===================================================================
dataset:
  # Caminho para a pasta que contém as imagens. 
  path: "data/images"
  
  # Semente aleatória para garantir que a divisão do K-Fold seja reprodutível.
  random_seed: 117

cross_validation:
  n_splits: 10

# ===================================================================
# 2. PARÂMETROS DE PRÉ-PROCESSAMENTO
# ===================================================================
preprocessing:
  image_size: 224 # ViT
  #image_size: 128 # Demais modelos

# ===================================================================
# 3. PARÂMETROS DE TREINAMENTO
# ===================================================================
training:
  optimizer: 'Adam'
  learning_rate: 0.001
  weight_decay: 1e-4
  batch_size: 4
  num_workers: 0
  
  epochs: 500
  early_stopping_patience: 25

# ===================================================================
# 4. ARQUITETURAS DOS MODELOS
# ===================================================================
architectures:

  CKAN:
    channels: [3, 8, 16] 
    kernel_size: [3, 3]
    padding: [1, 1]
    grid_size: 4         
    spline_order: 3

  CapsNet:
    # [Entrada, Saída Bloco 1, Saída Bloco 2]
    frontend_channels: [3, 32, 64]

    primary_caps_out_channels: 64
    primary_caps_dim: 8
    primary_caps_kernel_size: 5 

    digit_caps_dim: 16
    routings: 3
    lambda_reconstruction: 0.0005
```
(Nota: Para modelos como ResNet e ViT que não possuem uma entrada em `architectures:`, você pode omitir esta seção ou simplesmente declarar "Modelo padrão da torchvision, sem parâmetros customizados em config.yaml".)

## 🚀 Como Executar
Este script foi projetado para ser executado a partir do diretório raiz do projeto, para que todos os imports de módulos (`modules/`, `dataloaders/`, `models/`) funcionem corretamente.

1. Verifique a Configuração:

Antes de executar, confirme se os parâmetros da arquitetura (acima) e, principalmente, o `image_size` no `config.yaml` estão corretos para este modelo:

ViT: requer `image_size: 224`

Demais Modelos: usam `image_size: 128` (ou conforme sua configuração)

2. Execute o Script:

A partir do diretório raiz do projeto, execute o seguinte comando:

```Bash
python experiments/ResNet_Xray/run_experiment.py
```


## 🔬 O que este script faz?
O `run_experiment.py` automatiza todo o pipeline de avaliação robusta que definimos:

Carrega as configurações do `config.yaml`.

Separa um conjunto de teste final (Hold-Out) estratificado (por gênero e idade) do restante dos dados.

Executa uma Validação Cruzada de K-Folds (K=10) no restante dos dados (conjunto de Desenvolvimento).

Para cada fold:

Treina o modelo `ResNet_Xray`.

Usa `early_stopping_patience` para salvar o melhor checkpoint com base na perda de validação.

Avalia o melhor modelo do fold no conjunto de validação, calculando a AUC e encontrando o limiar ótimo (pelo Índice de Youden).

Avalia o melhor modelo do fold (com seu limiar ótimo) no conjunto de "Operação" (todos os dados de desenvolvimento).

Ao final dos K-folds, ele seleciona o "modelo campeão" (o modelo do fold com a maior AUC de "Operação").

Realiza uma avaliação final, única e imparcial deste modelo campeão no conjunto Hold-Out.

## 📊 Saídas (Resultados)
Todos os artefatos deste experimento serão salvos na pasta raiz `results/` em um diretório único com timestamp, seguindo o padrão:

`results/ResNet-18_Baseline/[YYYYMMDD_HHMMSS]/`

Este diretório conterá:

Subpastas para cada `fold_...` com logs e gráficos de perda.

A pasta `holdout_results/` com os gráficos ROC finais.

O modelo campeão salvo: `best_overall_model.pt`.

O resumo completo das métricas (com dados brutos dos folds): `summary_results.json.`