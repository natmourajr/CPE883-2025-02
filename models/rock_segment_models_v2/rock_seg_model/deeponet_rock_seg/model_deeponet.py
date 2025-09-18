import torch
import torch.nn as nn
import torch.nn.functional as F
from rock_seg_model.deeponet_rock_seg.deeponet_config import DEEPONET_CONFIG

class DeepONetConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Branch net — processa a entrada
        layers_branch = []
        in_ch = config["img_channels"]
        for out_ch, k, s in zip(config["branch_conv_channels"], config["branch_kernel_sizes"], config["branch_strides"]):
            layers_branch += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k//2), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.branch_net = nn.Sequential(*layers_branch)

        # Trunk net — também pode processar informação da entrada ou outra fonte
        layers_trunk = []
        in_ch = config["img_channels"]
        for out_ch, k, s in zip(config["trunk_conv_channels"], config["trunk_kernel_sizes"], config["trunk_strides"]):
            layers_trunk += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k//2), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.trunk_net = nn.Sequential(*layers_trunk)

        # Combinação via operador (produto escalar)
        hidden = config["hidden_dim"]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * config["output_shape"][0] * config["output_shape"][1], hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, config["num_classes"] * config["output_shape"][0] * config["output_shape"][1])
        )
        self.config = config

    def forward(self, x):
        b = self.branch_net(x)
        t = self.trunk_net(x)
        # junção simples — produto escalar por canal
        h = (b * t).flatten(1)
        out = self.fc(h)
        B, C, H, W = b.size(0), self.config["num_classes"], *self.config["output_shape"]
        return out.view(B, C, H, W)
