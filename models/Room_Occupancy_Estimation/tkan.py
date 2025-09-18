import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Dataset personalizado para PyTorch
class RoomOccupancyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Bloco Linear + Não Linear
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.non_linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x):
        linear_out = self.linear(x)
        non_linear_out = self.non_linear(x)
        return linear_out + non_linear_out


# Arquitetura principal
class KANTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            nn.Sequential(
                KANLinear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.gelu(self.input_layer(x))

        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual

        return self.output_layer(x)
