# rock_seg_model/cnn_rock_seg/model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from rock_seg_model.cnn_rock_seg.cnn_config import CNN_CONFIG

class CNNSegmentationModel(nn.Module):
    def __init__(self, config=CNN_CONFIG):
        super().__init__()
        ch = config["channels"]
        nc = config["num_classes"]

        # 3 camadas convolucionais padrão
        self.conv1 = nn.Conv2d(config["img_channels"], ch[0],
                               kernel_size=config["kernel_size"],
                               stride=config["stride"],
                               padding=config["padding"])
        self.conv2 = nn.Conv2d(ch[0], ch[1],
                               kernel_size=config["kernel_size"],
                               stride=config["stride"],
                               padding=config["padding"])
        self.conv3 = nn.Conv2d(ch[1], ch[2],
                               kernel_size=config["kernel_size"],
                               stride=config["stride"],
                               padding=config["padding"])

        # camada final para classificação por pixel
        self.final_conv = nn.Conv2d(ch[2], nc, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        out = self.final_conv(x)
        return out
