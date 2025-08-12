import torch
from torch import nn
import torch.nn.functional as F

from .capsnet.capsulelayers import squash, DenseCapsule, PrimaryCapsule

class CapsNet(nn.Module):
    def __init__(self, model_config, num_classes=2, device="cpu"):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes

        # Armazena o image_size para usá-lo mais tarde no forward pass
        self.image_size = model_config['image_size']

        # Camada 1: CNN Convencional
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        
        # Camada 2: Cápsulas Primárias
        self.primary_caps = PrimaryCapsule(in_channels=256, out_channels=256, dim_caps=8, kernel_size=9, stride=2)

        # Camada 3: Cápsulas de Classe (DigitCaps)
        conv_output_size = self._get_conv_output_size(self.image_size)
        self.digit_caps = DenseCapsule(
            in_num_caps=conv_output_size,
            in_dim_caps=8, 
            out_num_caps=num_classes, 
            out_dim_caps=16, 
            routings=3
        )

        # Decoder para a reconstrução
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.image_size * self.image_size),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            x = F.relu(self.conv1(dummy_input))
            x = self.primary_caps(x)
            return x.size(1)

    def forward(self, x, y_true_one_hot=None):
        digit_caps_output = self.digit_caps(self.primary_caps(F.relu(self.conv1(x))))
        
        y_pred = digit_caps_output.norm(dim=-1)

        if self.training:
            # Se não passar y_true_one_hot durante o treino, usa a predição para o decoder
            if y_true_one_hot is None:
                _, max_length_indices = y_pred.max(dim=1)
                y_true_one_hot = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
        else:
             _, max_length_indices = y_pred.max(dim=1)
             y_true_one_hot = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
            
        reconstruction_input = (digit_caps_output * y_true_one_hot[:, :, None]).view(digit_caps_output.size(0), -1)
        reconstruction = self.decoder(reconstruction_input)
        

        # Redimensiona a reconstrução usando o image_size que salvo, e não o shape de x
        reconstruction = reconstruction.view(-1, 3, self.image_size, self.image_size)

        return y_pred, reconstruction