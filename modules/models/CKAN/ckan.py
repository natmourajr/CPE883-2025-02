# modules/Models/ckan.py

import torch.nn as nn

#Busca lib
from .kan_lib.KANConv import KAN_Convolutional_Layer

class CKAN(nn.Module):
    def __init__(self, num_classes=2, device="cpu"): 
        super(CKAN, self).__init__()
        print(f"Inicializando modelo CKAN")
        # Extrai os parâmetros do dicionário de configuração
        arch_config = model_config['architecture']
        channels = arch_config['channels']
        kernel_size = tuple(arch_config['kernel_size']) # Converte a lista do YAML para tupla
        padding = tuple(arch_config['padding'])
        grid_size = arch_config['grid_size']
        spline_order = arch_config['spline_order']

        # Bloco 1:
        self.ckan1 = KAN_Convolutional_Layer(
            in_channels=channels[0], 
            out_channels=channels[1], 
            kernel_size=kernel_size, 
            padding=padding, 
            grid_size=grid_size, 
            spline_order=spline_order,
            device=device 
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Bloco 2: 
        self.ckan2 = KAN_Convolutional_Layer(
            in_channels=channels[1], 
            out_channels=channels[2], 
            kernel_size=kernel_size, 
            padding=padding, 
            grid_size=grid_size, 
            spline_order=spline_order,
            device=device 
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        final_channels = channels[2] # 32
        final_size = model_config['image_size'] // 4 # Divide por 2 a cada pooling
        self.fc1 = nn.Linear(final_channels * final_size * final_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x