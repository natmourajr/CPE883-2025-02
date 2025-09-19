# Cognite Time Series Toolkit
O pacote `tsdiffusion_transformed` reúne ferramentas de segmentação, modelagem neuro-simbólica e difusão para séries temporais industriais. Os módulos foram desenhados para funcionar em sequência, mas cada componente permanece independente para uso em pipelines Python próprios.

## Dependências
- Python 3.10 ou superior
- PyTorch (com suporte opcional a CUDA)
- NumPy, pandas e SciPy
- scikit-learn

Instale as bibliotecas com `pip install -r requirements.txt` na raiz do projeto ou instale manualmente os pacotes listados acima.

## Estrutura do pacote
| Arquivo | Finalidade |
| --- | --- |
| `segmenter.py` | Segmentação não supervisionada de estados operacionais com HMM Gaussiano e extração automática de atributos multivariados. |
| `ode_jump.py` | Implementação base do encoder Jump-ODE com rotinas de treino, avaliação e utilidades para séries mascaradas. |
| `ode_jump_encoder.py` | Extensão transformer do Jump-ODE com blocos de self-attention, agendamentos de aprendizado e testes auxiliares. |
| `tsdiffusion.py` | Modelo de difusão latente que reutiliza o encoder Jump-ODE para denoising e imputação consistentes. |
| `lstm.py` | Baseline recorrente usada como comparação simples. |
| `ode_jump_encoder_test.py` | Testes unitários que exercitam cenários de treino e inferência. |

## Pré-processamento dos dados
As rotinas utilizam objetos `pandas.DataFrame` com:
- colunas numéricas a serem modeladas (`feature_cols`);
- colunas opcionais de atributos estáticos disponíveis em todos os instantes (`static_features_cols`);
- uma coluna temporal (`timestamp_col`) ou uso do índice como carimbo de tempo (`"index"`);
- um identificador de estado (`states_col`) para dividir janelas estratificadas durante o treino do Jump-ODE ou para validar segmentações.

Os métodos lidam com valores ausentes por meio de máscaras binárias; os dados podem conter `NaN`, `inf` ou gaps que serão preenchidos/ignorados conforme o fluxo.

## Fluxo sugerido
1. Gerar segmentos de estados com `UnsupervisedStateSegmenter`.
2. Treinar um encoder Jump-ODE (`ODEJump` ou `ODEJumpEncoder`) usando as janelas rotuladas.
3. Ajustar `TSDiffusion` para denoising/imputação e reutilizar as séries reconstruídas em loops adicionais.
4. Opcionalmente salvar e carregar cada componente com os utilitários de persistência embutidos.

### Segmentação de estados (`segmenter.py`)
O segmentador gera janelas, extrai atributos estatísticos/espectrais por canal, treina um HMM diagonal via EM e identifica automaticamente um estado anômalo.

#### Exemplo: ajustar e usar o segmentador
```python
import pandas as pd
from tsdiffusion_transformed.segmenter import UnsupervisedStateSegmenter, FeatureConfig

df = pd.read_csv("dados.csv").sort_values("timestamp")
feature_cols = [c for c in df.columns if c.startswith("sensor_")]
series = df[feature_cols].to_numpy()

seg = UnsupervisedStateSegmenter(
    k_states=6,
    window=30,
    step=10,
    device="cuda",
    feature_conf=FeatureConfig(acf_lags=15, bands=[(0.0, 0.1), (0.1, 0.3), (0.3, 0.5)])
)
seg.fit(series)

summary = seg.predict(series)
print(summary["states_per_window"])     # sequência discreta por janela
print(summary["anomaly_mask"])          # 1 onde o estado anômalo foi identificado
print(summary["slices"])                # blocos [inicio, fim, estado]
```

#### Exemplo: salvar e reaplicar um modelo treinado
```python
seg.save("segmenter.pkl")

# Em outra sessão
from tsdiffusion_transformed.segmenter import UnsupervisedStateSegmenter
loaded = UnsupervisedStateSegmenter.load("segmenter.pkl", map_location="cpu")
new_states = loaded.predict(novos_dados)["states"]
```

### Encoder Jump-ODE (`ode_jump.py` e `ode_jump_encoder.py`)
`ODEJump` combina máscaras de observação, integração contínua por ODE com saltos e uma cabeça de observabilidade. `ODEJumpEncoder` adiciona blocos transformer e duas taxas de aprendizado, tornando o modelo mais expressivo para tensões multivariadas.

#### Exemplo: treino supervisionado por janelas
```python
from tsdiffusion_transformed.ode_jump_encoder import ODEJumpEncoder

feature_cols = ["sensor_ax", "sensor_ay", "sensor_az"]
static_cols = ["compressor_id", "potencia_nominal"]
model = ODEJumpEncoder(
    in_channels=len(feature_cols),
    hidden_dim=256,
    static_dim=len(static_cols),
    n_layers=8,
    n_heads=8,
    lam=[0.5, 0.4, 0.1],      # pesos das perdas L1, difusão e máscara
)

metrics_stream = model.train_cognite(
    df=df,
    feature_cols=feature_cols,
    static_features_cols=static_cols,
    timestamp_col="timestamp",
    states_col="estado_segmentado",
    window_size=30,
    window_step=15,
    batch_size=128,
    epochs=60,
    validate=True,
    early_stopping=True,
)

for epoch, metrics in enumerate(metrics_stream, start=1):
    macro = metrics["macro_mse"]
    micro = metrics["micro_mse"]
    print(f"Época {epoch}: macro={macro:.6f} | micro={micro:.6f}")
```

#### Exemplo: avaliar e reutilizar checkpoints
```python
import numpy as np
from torch.utils.data import DataLoader

model.save("encoder_final.pt")

reloaded = ODEJumpEncoder.load(
    "encoder_final.pt",
    in_channels=len(feature_cols),
    hidden_dim=256,
    static_dim=len(static_cols),
    n_layers=8,
    n_heads=8,
    lam=[0.5, 0.4, 0.1],
)

### Difusão para denoising/imputação (`tsdiffusion.py`)
`TSDiffusion` reutiliza o encoder Jump-ODE para rodar passos DDPM no espaço latente, estimar ruído e reconstruir séries mascaradas com consistência em relação aos dados observados.

#### Exemplo: treino com difusão
```python
from tsdiffusion_transformed.tsdiffusion import TSDiffusion

diff = TSDiffusion(
    in_channels=len(feature_cols),
    hidden_dim=384,
    static_dim=len(static_cols),
    n_layers=12,
    n_heads=8,
    num_steps=500,
    lam=[0.3, 0.6, 0.1],
)

for epoch, metrics in enumerate(diff.train_cognite(
    df=df,
    feature_cols=feature_cols,
    static_features_cols=static_cols,
    timestamp_col="timestamp",
    states_col="estado_segmentado",
    window_size=30,
    window_step=15,
    batch_size=64,
    epochs=80,
    validate=False,
)):
    print(f"Época {epoch+1}: micro MSE = {metrics['micro_mse']:.6f}")
```

#### Exemplo: reconstruir DataFrames e imputar valores
```python
# Gera duas versões: sequência completa do processo reverso e blend com dados originais
reconstructed_df, denoised_df = diff.denoise_dataframe(
    df=df,
    feature_cols=feature_cols,
    timestamp_col="timestamp",
    static_features_cols=static_cols,
    window_size=30,
    window_step=15,
    replace_only_missing=True,
)