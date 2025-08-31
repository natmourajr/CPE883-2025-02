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


# CNN Model (simple regression)
class CNNRegressor(nn.Module):
    def __init__(self, input_shape, output_size=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B, 16, S, T)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 16, S/2, T/2)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, S/4, T/4)
            nn.Flatten()
        )
        sample_input = torch.zeros(1, *input_shape)
        out_features = self.cnn(sample_input).shape[1]
        self.fc = nn.Linear(out_features, output_size)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x)