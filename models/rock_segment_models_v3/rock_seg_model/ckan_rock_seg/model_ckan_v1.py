### rock_seg_model/kan_rock_seg/model_ckan_v1.py
import torch
import torch.nn as nn
from rock_seg_model.ckan_rock_seg.kan_conv import KAN_Convolutional_Layer

class CKANConv2DReal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.kan_conv = KAN_Convolutional_Layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        return self.kan_conv(x)

class CKANSegmentationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = config["channels"]
        ks = config["kernel_size"]
        st = config["stride"]
        pad = config["padding"]
        dil = config["dilation"]
        nc = config["num_classes"]

        self.ckan1 = CKANConv2DReal(ch[0], ch[1], ks, st, pad, dil)
        self.pool1 = nn.Identity()

        self.ckan2 = CKANConv2DReal(ch[1], ch[2], ks, st, pad, dil)
        self.pool2 = nn.Identity()

        self.ckan3 = CKANConv2DReal(ch[2], ch[3], ks, st, pad, dil)
        self.pool3 = nn.Identity()

        self.final_conv = nn.Conv2d(ch[3], nc, kernel_size=1)

    def forward(self, x):
        x = self.pool1(self.ckan1(x))
        x = self.pool2(self.ckan2(x))
        x = self.pool3(self.ckan3(x))
        return self.final_conv(x)