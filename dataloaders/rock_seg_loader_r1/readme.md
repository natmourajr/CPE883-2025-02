## Dataset Loader — RockSegmentationDatasetMulti

Este DataLoader foi desenvolvido para lidar com múltiplos conjuntos de imagens de microtomografia de rochas, cada um com suas respectivas máscaras de segmentação. 

## Principais Funcionalidades

- Suporte a múltiplas amostras: espera-se uma estrutura de diretórios onde cada conjunto de dados esteja separado (ex: DatasetA/, DatasetB/...), cada um contendo suas pastas inputs/ e masks/.

- Pré-processamento integrado:
-   Normalização das imagens para [0, 1]
-   Redimensionamento para um tamanho padrão configurável (ex: 512×512)
-   Compatível com múltiplos modelos reutilizando o mesmo dataset.

##### Rastreamento por nome de amostra: para cada par (imagem, máscara), o loader retorna também o nome da amostra original (amostra_nome + filename) — isso garante rastreabilidade das predições ao longo do processo de aprendizado e avaliação.

### Estrutura de diretórios esperada

```

root_dataset/
│
├── DatasetA/
│   ├── inputs/
│   │   ├── sampleA0001.tiff
│   │   └── ...
│   └── masks/
│       ├── maskA0001.tiff
│       └── ...
│
├── DatasetB/
│   ├── inputs/
│   └── masks/
│
└── ...

```

## Exemplo de retorno por amostra

Cada item retornado pelo __getitem__() tem a seguinte estrutura:

- image: Tensor  # shape: [1, H, W], float32, normalizada
- mask:  Tensor  # shape: [H, W], long, com classes inteiras
- filename: str  # exemplo: "DatasetA_sampleA0001.tiff"

Esse campo filename pode ser utilizado para:
- Nomear arquivos de saída durante a validação/visualização e teste
- Fazer rastreabilidade por origem da amostra
- Depuração e análise de desempenho por dataset


## 📚 Documentação do Projeto

- [x] Dataset Loader: `RockSegmentationDatasetMulti` — suporte multi-amostras com rastreabilidade
- [ ] Treinamento com CapsNet
- [ ] Treinamento com CKAN
- [ ]  Treinamento com DeepONet
- [ ] Métricas e Visualizações
