import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KAN_Convolutional_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, grid_size=5, spline_order=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = grid_size
        self.spline_order = spline_order

        # CORREÇÃO: Guardar 'h' como um atributo da classe
        self.h = 2.0 / (grid_size - 1) if grid_size > 1 else 1.0
        
        # Weights
        self.base_weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
        self.spline_weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size, grid_size + spline_order))
        
        # Grid
        grid_tensor = (torch.arange(-grid_size, grid_size, step=1) * self.h).expand(in_channels, *self.kernel_size, -1)
        self.register_buffer("grid", grid_tensor)

    def b_splines(self, x):
        # x shape: (batch_size * num_patches, in_channels, k_h, k_w)
        x = x.unsqueeze(-1)  # shape becomes: (..., 1)
        
        # CORREÇÃO: Usar 'self.h' em vez de 'h'
        bases = ((x >= self.grid) & (x < self.grid + self.h)).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.grid) / k * bases + (self.grid + k * self.h - x) / k * bases.roll(-1, dims=-1)).to(x.dtype)
        return bases

    def forward(self, x):
        batch_size, _, H, W = x.shape
        padding_tuple = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding
        
        x_unfolded = F.unfold(x, self.kernel_size, stride=self.stride, padding=padding_tuple)
        num_patches = x_unfolded.shape[-1]
        
        x_unfolded = x_unfolded.view(batch_size, self.in_channels, self.kernel_size[0], self.kernel_size[1], num_patches)
        x_unfolded = x_unfolded.permute(0, 4, 1, 2, 3)

        base_output = torch.einsum('blchw,ochw->blo', x_unfolded, self.base_weight)

        x_for_spline = x_unfolded.reshape(batch_size * num_patches, self.in_channels, *self.kernel_size)
        spline_bases = self.b_splines(x_for_spline)
        
        spline_output = torch.einsum('nchwg,ochwg->no', spline_bases, self.spline_weight)
        spline_output = spline_output.view(batch_size, num_patches, self.out_channels)

        output = base_output + spline_output
        
        output_h = (H + 2 * padding_tuple[0] - self.kernel_size[0]) // self.stride + 1
        output_w = (W + 2 * padding_tuple[1] - self.kernel_size[1]) // self.stride + 1
        output = output.permute(0, 2, 1).view(batch_size, self.out_channels, output_h, output_w)

        return output