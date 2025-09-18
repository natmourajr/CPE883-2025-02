import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 1.0 / (grid_size - 1)
        grid = torch.arange(-spline_order, grid_size + spline_order) * h
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.randn(out_features, in_features))

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid) & (x < self.grid + 1.0)).float()
        for k in range(1, self.spline_order + 1):
            bases = ((x - self.grid) / k * bases + (self.grid + k + 1 - x) / k * bases.roll(-1, dims=-1))
        return bases

    def spline(self, x):
        return F.linear(self.b_splines(x).view(x.size(0), -1), self.spline_weight.view(self.out_features, -1))

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        spline_output = self.spline(x)
        return base_output + spline_output