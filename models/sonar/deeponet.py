import torch
import torch.nn as nn
import functools
import typing

from models.base_model import BaseModel

class DeepONet(BaseModel):
    """
    Implementação da Deep Operator Network (DeepONet) para classificação.
    Esta versão é flexível e aceita qualquer módulo PyTorch como Branch Net.
    """
    def __init__(self,
                 branch_net: torch.nn.Module,
                 trunk_net: torch.nn.Module,
                 class_head: torch.nn.Module,
                 use_bias: bool = True):
        super().__init__()

        self.branch_net = branch_net

        self.trunk_net = trunk_net
        self.use_bias = use_bias
        
        self.class_head = class_head
        
        if use_bias:
            self.bias = torch.nn.Parameter(torch.randn(1))


    def forward(self, data: torch.Tensor, coords: torch.Tensor, embeddings=False) -> torch.Tensor:
        branch_output = self.branch_net(data)
        
        # if (embeddings):
        #     return branch_output
        
        trunk_output = self.trunk_net(coords)
        
        logits = torch.matmul(branch_output, trunk_output.t())

        if self.use_bias:
            logits = logits + self.bias

        if (embeddings):
            return logits
        
        y_pred = self.class_head(logits)
        return y_pred