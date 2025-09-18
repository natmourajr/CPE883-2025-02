"""
CNN Regressor

Summary:

Considerations:
    - 

Improvement Sugestions:
    - 
    

version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN Model (simple regression)
class CNNRegressor(nn.Module):
    def __init__(self, input_shape, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2,3))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2,3))
        self.dropout = nn.Dropout(0.2)

        # Calcula dinamicamente tamanho do flatten
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # shape: (1, C=1, H=6, W=9)
            dummy = torch.relu(self.conv1(dummy))
            dummy = torch.relu(self.conv2(dummy))
            n_features = dummy.numel()  # total de features após flatten

        self.fc = nn.Linear(n_features, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x