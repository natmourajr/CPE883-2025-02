# rock_seg_experiment2/model.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from kan import KAN
# from pykan.splines import coef2curve  # para visualização futura opcional


class KANActivation(nn.Module):
    """
    Camada KAN real: aplica uma spline 1D aprendível em cada canal.
    Usa o pacote `pykan` para aplicar splines de ativação.
    """
    def __init__(self, num_channels):
        super().__init__()
        # Cria uma KAN independente para cada canal
        self.kans = nn.ModuleList([KAN(width=[1, 1]) for _ in range(num_channels)])
        self.num_channels = num_channels

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x_split = torch.split(x, 1, dim=1)  # lista de [B,1,H,W] para cada canal

        out_channels = []
        for i, xi in enumerate(x_split):
            xi_flat = xi.reshape(B * H * W, 1)         # corrigido: .reshape() ao invés de .view()
            yi_flat = self.kans[i](xi_flat)           # aplica KAN individual
            yi = yi_flat.view(B, 1, H, W)             # esta view ainda é válida, tensor novo
            out_channels.append(yi)

        out = torch.cat(out_channels, dim=1)  # [B, C, H, W]
        return out

    # def plot_splines(self):
    #     for i, kan in enumerate(self.kans):
    #         curve = coef2curve(kan.coeffs[0])  # para ver curva do canal i
    #         plt.plot(*curve, label=f'Canal {i}')
    #     plt.title("Curvas spline aprendidas (KANActivation)")
    #     plt.xlabel("Entrada")
    #     plt.ylabel("Saída")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()


class CKANConv2D(nn.Module):
    """
    Camada convolucional seguida de ativação KAN real.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.kan = KANActivation(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.kan(x)
        return x


class CKANSegmentationModel(nn.Module):
    """
    Modelo de segmentação por pixel com arquitetura CNN simples + KAN.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.ckan1 = CKANConv2D(1, 16)
        self.pool1 = nn.Identity()   #nn.MaxPool2d(2)

        self.ckan2 = CKANConv2D(16, 32)
        self.pool2 = nn.Identity()    #nn.MaxPool2d(2)

        self.ckan3 = CKANConv2D(32, 64)
        self.pool3 =  nn.Identity()  #nn.AdaptiveAvgPool2d((64, 64))  # shape final fixo

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # saída: [B, 3, H, W]

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = self.pool3(self.ckan3(x))
        x = self.final_conv(x)
        return x  # [B, 3, H, W]
