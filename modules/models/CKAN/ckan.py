# modules/Models/ckan.py

import torch.nn as nn

#Busca lib
from .kan_lib.KANConv import KAN_Convolutional_Layer

class CKAN(nn.Module):
    def __init__(self, num_classes=2, device="cpu"): 
        super(CKAN, self).__init__()
        print(f"Inicializando modelo CKAN no dispositivo: {device}")

        grid_size = 5
        spline_order = 3

        # Bloco 1:
        self.ckan1 = KAN_Convolutional_Layer(
            in_channels=3, 
            out_channels=16, 
            kernel_size=(3,3), 
            padding=(1,1), 
            grid_size=grid_size, 
            spline_order=spline_order,
            device=device  # <-- passando o dispositivo
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Bloco 2: 
        self.ckan2 = KAN_Convolutional_Layer(
            in_channels=16, 
            out_channels=32, 
            kernel_size=(3,3), 
            padding=(1,1), 
            grid_size=grid_size, 
            spline_order=spline_order,
            device=device  # <-- passando o dispositivo
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x