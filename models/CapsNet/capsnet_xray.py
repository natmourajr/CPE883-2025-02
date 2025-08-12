import torch
from torch import nn
import torch.nn.functional as F

from .capsnet.capsulelayers import squash, DenseCapsule, PrimaryCapsule

class CapsNet(nn.Module):
    def __init__(self, model_config, num_classes=2, device="cpu"):
        super(CapsNet, self).__init__()
        self.num_classes = num_classes
        image_size = model_config['image_size']

        # Camada 1: Uma CNN convencional para extração de características
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=9, stride=1)
        
        # Camada 2: Cápsulas Primárias
        self.primary_caps = PrimaryCapsule(
            in_channels=256, 
            out_channels=256, 
            dim_caps=8, 
            kernel_size=9, 
            stride=2
        )

        # Camada 3: Cápsulas de Classe (DigitCaps)
        # Para descobrir o in_num_caps, precisamos calcular o shape da saída da camada anterior.
        # Faremos isso com um "dry run"
        conv_output_size = self._get_conv_output_size(image_size)
        self.digit_caps = DenseCapsule(
            in_num_caps=conv_output_size, # <-- Número calculado dinamicamente!
            in_dim_caps=8, 
            out_num_caps=num_classes, 
            out_dim_caps=16, 
            routings=3
        )

        # Decoder para a regularização via reconstrução
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * image_size * image_size),
            nn.Sigmoid()
        )

    def _get_conv_output_size(self, image_size):
        # Cria um tensor de teste para passar pelas camadas convolucionais
        # e descobrir o shape da saída dinamicamente.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            x = F.relu(self.conv1(dummy_input))
            x = self.primary_caps(x) # Shape: [1, num_caps, dim_caps]
            return x.size(1) # Retorna o num_caps

    def forward(self, x, y_true_one_hot=None):
        # Passa pelas camadas convolucional e de cápsulas primárias
        x = F.relu(self.conv1(x))
        x = self.primary_caps(x)
        
        # Passa pelas cápsulas de classe
        digit_caps_output = self.digit_caps(x) # Shape: [batch, num_classes, 16]

        # Calcula o comprimento dos vetores de saída para obter a predição
        y_pred = digit_caps_output.norm(dim=-1) # Shape: [batch, num_classes]

        # Lógica de Reconstrução
        # Se os rótulos forem fornecidos (durante o treino), usa-os para mascarar.
        # Senão (durante a validação/teste), usa a predição com maior probabilidade.
        if y_true_one_hot is None:
            _, max_length_indices = y_pred.max(dim=1)
            y_true_one_hot = torch.eye(self.num_classes).to(x.device).index_select(dim=0, index=max_length_indices)
            
        reconstruction = self.decoder((digit_caps_output * y_true_one_hot[:, :, None]).view(x.size(0), -1))
        reconstruction = reconstruction.view(-1, 3, x.shape[2], x.shape[3]) # Redimensiona para o formato da imagem

        return y_pred, reconstruction