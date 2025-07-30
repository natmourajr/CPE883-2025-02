# rock_seg/kan_conv.py
import torch
import math
from rock_seg.convolution import multiple_convs_kan_conv2d
from kan import KANLayer  # Correto para pykan==0.2.8

class KAN_Convolution(torch.nn.Module):
    def __init__(
        self,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        #scale_base=1.0,
        #cale_spline=1.0,
    ):
        super().__init__()
        self.conv = KANLayer(
            in_dim=math.prod(kernel_size),
            out_dim=1,
            #scale_base=scale_base,
            #scale_spline=scale_spline,
        )

class KAN_Convolutional_Layer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        #scale_base=1.0,
        #scale_spline=1.0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

        self.convs = torch.nn.ModuleList([
            KAN_Convolution(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                #scale_base=scale_base,
                #scale_spline=scale_spline,
            )
            for _ in range(in_channels * out_channels)
        ])

    def forward(self, x):
        device = x.device
        return multiple_convs_kan_conv2d(
            x,
            self.convs,
            self.kernel_size[0],
            self.out_channels,
            self.stride,
            self.dilation,
            self.padding,
            device
        )
