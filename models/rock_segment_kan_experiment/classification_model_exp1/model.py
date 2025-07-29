# model.py
#[Entrada da imagem 512x512x1]
# CKANConv2D(1→16)       (Conv2d 1x1 + ReLU)
#MaxPool2D              (reduz para 256x256)
#CKANConv2D(16→32)      (Conv2d 1x1 + ReLU)
# MaxPool2D              (reduz para 128x128)
# CKANConv2D(32→64)      (Conv2d 1x1 + ReLU)
# AdaptiveAvgPool(8x8)   (reduz para 8x8, fixo)
# Flatten
# FC (Linear: 64*8*8 → 128)
# ReLU
# FC (Linear: 128 → 3 classes)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CKANConv2D(nn.Module):
    """
    Camada convolucional + ReLU (simulando CKAN de forma simplificada).
    Essa versão usa uma convolução 1x1 e aplica ReLU como substituto temporário do spline.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.kan = nn.ReLU()  # Substituto temporário para KANLayer

    def forward(self, x):
        x = self.conv(x)
        x = self.kan(x)
        return x


class SimpleCKANModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.ckan1 = CKANConv2D(1, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.ckan2 = CKANConv2D(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.ckan3 = CKANConv2D(32, 64)
        self.pool3 = nn.AdaptiveAvgPool2d((8, 8))  # Reduz tamanho para FC

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = self.pool3(self.ckan3(x))
        x = self.classifier(x)
        return x
