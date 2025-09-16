import torch
import torch.nn as nn
import torch.nn.functional as F
from rock_seg_model.caps_rock_seg.caps_config import CAPS_CONFIG


# ==== Blocos CapsNet ====
class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, caps_channels, caps_dim, kernel_size, stride):
        super().__init__()
        self.caps_dim = caps_dim
        self.caps_channels = caps_channels
        self.conv = nn.Conv2d(in_channels, caps_channels * caps_dim, kernel_size, stride)

    def forward(self, x):
        batch = x.size(0)
        out = self.conv(x)
        out = out.view(batch, self.caps_channels, self.caps_dim, out.size(2), out.size(3))
        out = self.squash(out)
        return out

    def squash(self, s):
        norm = torch.norm(s, dim=2, keepdim=True)
        return (norm**2 / (1 + norm**2)) * (s / (norm + 1e-8))

class SegCapsNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = nn.Conv2d(
            config["img_channels"], config["conv1_channels"],
            kernel_size=config["conv1_kernel"],
            stride=config["conv1_stride"],
            padding=config["conv1_padding"]
        )

        self.primary_caps = PrimaryCaps(
            in_channels=config["conv1_channels"],
            caps_channels=config["primary_caps_channels"],
            caps_dim=config["primary_caps_dim"],
            kernel_size=1,
            stride=1
        )

        self.deconv = nn.ConvTranspose2d(
            config["primary_caps_channels"] * config["primary_caps_dim"],
            config["conv1_channels"], kernel_size=1, stride=1
        )

        self.final = nn.Conv2d(
            config["conv1_channels"], config["num_classes"],
            kernel_size=1
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        caps = self.primary_caps(x)
        caps = caps.view(x.size(0), -1, caps.size(3), caps.size(4))  # [B, C*dim, H, W]
        x = F.relu(self.deconv(caps))
        out = self.final(x)
        return out


