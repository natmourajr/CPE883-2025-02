# experiments/ResNet_Baseline/run_experiment.py

import yaml
import torch
import torch.nn as nn
from torchvision import models
import sys
import os

from modules.Evaluation.evaluator import run_kfold_evaluation

# ============================================================
# CLASSE DO MODELO BASELINE (ResNet-18 com Fine-Tuning)
# ============================================================
class ResNetBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetBaseline, self).__init__()

        # 1. Carrega o modelo ResNet-18 pré-treinado no ImageNet
        # A API 'weights' carrega modelos
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 2. "Congela" todos os pesos da rede. Não treina as camadas que já
        # aprenderam a extrair características no ImageNet.
        for param in self.resnet.parameters():
            param.requires_grad = False

        # 3. Substitui a camada final (o "classificador").
        # A camada original da ResNet-18 (chamada 'fc') tem 1000 saídas (classes do ImageNet).
        # Precisa de uma nova camada com apenas 2 saídas (Normal vs. Tuberculose).
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        # Apenas os parâmetros desta nova camada serão treinados.
        print("Modelo ResNet-18 Baseline Inicializado! Camadas congeladas e classificador substituído.")

    def forward(self, x):
        # Apenas passa os dados pelo modelo modificado
        return self.resnet(x)
