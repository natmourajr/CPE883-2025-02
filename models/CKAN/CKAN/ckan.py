import torch.nn as nn
from .kan_lib.KANConv import KAN_Convolutional_Layer


class CKAN(nn.Module):
    def __init__(self, image_size, num_classes=10, device="cpu"):
        super(CKAN, self).__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.device = device

        # Extrai os parâmetros do dicionário de configuração
        channels = [3, 8, 16]
        kernel_size = (3, 3)
        padding = (1, 1)
        grid_size = 4
        spline_order = 3

        # Bloco 1:
        self.ckan1 = KAN_Convolutional_Layer(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size,
            padding=padding,
            grid_size=grid_size,
            spline_order=spline_order,
            device=device,
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
            device=device,
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        final_channels = channels[2]
        final_size = image_size // 4  # Divide por 2 a cada pooling
        self.fc1 = nn.Linear(final_channels * final_size * final_size, num_classes)

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
