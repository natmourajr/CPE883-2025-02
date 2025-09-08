# models/ckan.py

import torch.nn as nn
from torchvision import models

class ViTBaseline(nn.Module):
    """
    Classe para o Vision Transformer (ViT-B/16) com fine-tuning.
    """
    def __init__(self, model_config, num_classes=2, device="cpu"):
        super(ViTBaseline, self).__init__()

        # 1. Carrega o modelo ViT-Base com patches 16x16, pré-treinado no ImageNet-1K
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        # 2. "Congela" todos os pesos da rede para não treinar o corpo principal.
        for param in self.vit.parameters():
            param.requires_grad = False

        # 3. Substitui a camada final (a "cabeça" classificadora).
        # No ViT da torchvision, a camada se chama 'heads'.
        num_ftrs = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_ftrs, num_classes)

        # Apenas os parâmetros desta nova camada de classificação serão treinados.
        print("Modelo ViT-B/16 Baseline Inicializado! Camadas congeladas e classificador substituído.")

    def forward(self, x):
        return self.vit(x)
