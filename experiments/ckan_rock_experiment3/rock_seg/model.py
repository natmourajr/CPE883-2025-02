### rock_seg/model.py
import torch
import torch.nn as nn
from rock_seg.kan_conv import KAN_Convolutional_Layer   #from rock_seg.kan_conv import KAN_Convolutional_Layer

class CKANConv2DReal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.kan_conv = KAN_Convolutional_Layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
            dilation=(1,1),
            #base_activation=nn.SiLU
        )

    def forward(self, x):
        return self.kan_conv(x)

class CKANSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.ckan1 = CKANConv2DReal(1, 16)
        self.pool1 = nn.Identity()

        self.ckan2 = CKANConv2DReal(16, 32)
        self.pool2 = nn.Identity()

        self.ckan3 = CKANConv2DReal(32, 64)
        self.pool3 = nn.Identity()

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = self.pool3(self.ckan3(x))
        return self.final_conv(x)