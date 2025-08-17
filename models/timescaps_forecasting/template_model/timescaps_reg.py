import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.utils import to_categorical

# Importa o CapsNet do arquivo baixado
from capsulenet_rnn_dim_change import CapsNet

SEQ_LEN = 360     # janela temporal (pode ajustar)
N_PRED_STEPS = 1  # quantos passos no futuro prever (ex: 1 hora ou 1 passo)
INST_PARAM = 16   # parâmetro do modelo, manter compatível

# Carregue seus dados reais aqui, exemplo genérico:
# x_data.shape = (num_samples, SEQ_LEN, 1)
# y_data.shape = (num_samples, N_PRED_STEPS, 1)  # saída regressiva

# Aqui só dados aleatórios para exemplo
num_samples = 1000
x_data = np.random.rand(num_samples, SEQ_LEN, 1).astype('float32')
y_data = np.random.rand(num_samples, N_PRED_STEPS, 1).astype('float32')

# Dividir treino/teste (exemplo 80/20)
split = int(num_samples * 0.8)
x_train, x_test = x_data[:split], x_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# --- Modificar a CapsNet para saída regressiva ---
# A função CapsNet original é para classificação.
# Para regressão, podemos pegar o modelo de avaliação e adicionar camada densa.

def CapsNet_regression(input_shape, inst_parameter):
    x = layers.Input(shape=input_shape)

    # Copia a parte convolucional + cápsulas (simplificado)
    l = layers.Conv1D(filters=64, kernel_size=7, strides=1, padding='same', activation='relu')(x)
    l = layers.BatchNormalization()(l)
    l = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(l)

    # Transformar para cápsulas (exemplo simples)
    l = layers.Reshape((-1, inst_parameter))(l)  # aqui inst_parameter como dim_capsule
    l = layers.Lambda(lambda z: z / (K.sqrt(K.sum(K.square(z), axis=-1, keepdims=True)) + 1e-7))(l)  # squash simples

    # Camada Capsule Layer - pode ser importada do seu capsulenet_rnn_dim_change.py se quiser
    # Aqui simplificamos: soma e ativação
    l = layers.GlobalAveragePooling1D()(l)

    # Saída regressiva - camada densa para prever demanda futura
    out = layers.Dense(N_PRED_STEPS, activation='linear')(l)

    model = models.Model(inputs=x, outputs=out)
    return model

# Instancia o modelo regressivo
model = CapsNet_regression(input_shape=(SEQ_LEN,1), inst_parameter=INST_PARAM)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treina o modelo
model.fit(x_train, y_train.reshape(-1, N_PRED_STEPS),
          batch_size=32,
          epochs=30,
          validation_data=(x_test, y_test.reshape(-1, N_PRED_STEPS)))

# Faz predições
y_pred = model.predict(x_test)

print("Previsão exemplo:", y_pred[:5])
