"""
Time series Capsnet for a classification problem based on the model TimeCaps[1].

Summary: Kan Classification of 3W pipeline data using P-MON-CKP and T-JUS-CKP as inputs.

Considerations:
    - 


version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] 
"""
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timm.layers import PatchEmbed
from typing import Union, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

sys.path.append("CPE883-2025-02/models/")
sys.path.append("CPE883-2025-02/dataloaders/3W/")
sys.path.append("/3W")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
from dataloaders.benchmark.collector import Collector3W

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from TimeCaps_master.capsulenet_rnn_dim_change import CapsNet  # Importa o modelo do arquivo local

# --- Parâmetros do problema ---
SEQ_LEN = 360          # comprimento da sequência temporal
N_FEATURES = 5         # número de features por timestep
N_CLASSES = 10         # número de classes para classificação
ROUTINGS = 3           # número de iterações de roteamento
DIM_CAPSULE = 16       # dimensão das cápsulas

# --- Preparar dados sintéticos para exemplo ---
x_train = np.random.rand(800, SEQ_LEN, N_FEATURES).astype('float32')
y_train = np.random.randint(0, N_CLASSES, size=(800,))
y_train_cat = to_categorical(y_train, N_CLASSES)

x_test = np.random.rand(200, SEQ_LEN, N_FEATURES).astype('float32')
y_test = np.random.randint(0, N_CLASSES, size=(200,))
y_test_cat = to_categorical(y_test, N_CLASSES)

# --- Instanciar o modelo CapsNet do arquivo importado ---
# Atenção: O CapsNet do repositório original provavelmente espera input shape (seq_len, 1)
# Se for multi-features, adapte a entrada (por exemplo use reshape ou ajuste no modelo)
input_shape = (SEQ_LEN, N_FEATURES)

train_model, eval_model, _ = CapsNet(input_shape, N_CLASSES, ROUTINGS, DIM_CAPSULE)

# --- Compilar o modelo ---
def margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    L = y_true * np.square(np.maximum(0., m_plus - y_pred)) + \
        lambda_val * (1 - y_true) * np.square(np.maximum(0., y_pred - m_minus))
    return np.mean(np.sum(L, axis=1))

# Usar a loss do CapsNet no Tensorflow/Keras (exemplo simples)
train_model.compile(
    optimizer=Adam(lr=0.001),
    loss=['categorical_crossentropy', 'mse'],
    loss_weights=[1., 0.392],
    metrics={'capsnet': 'accuracy'}
)

# --- Treinar o modelo ---
train_model.fit(
    [x_train, y_train_cat],
    [y_train_cat, x_train],
    batch_size=32,
    epochs=15,
    validation_data=([x_test, y_test_cat], [y_test_cat, x_test])
)

# --- Avaliar e testar ---
y_pred, x_recon = eval_model.predict(x_test)
print("Shape predição:", y_pred.shape)
print("Exemplo predição:", y_pred[:5])