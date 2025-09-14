import torch
import torch.nn as nn


class TKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(TKANLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.reshape(batch_size * seq_len, input_dim)
        x = self.linear(x)
        x = self.activation(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x
# Modelo TKAN em PyTorch. O original está no Keras
class TKANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TKANModel, self).__init__()
        self.tkan1 = TKANLayer(input_dim, hidden_dim)
        self.tkan2 = TKANLayer(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        x = self.tkan1(x)
        x = self.tkan2(x)
        x = x[:, -1, :]    # [batch, hidden] -> pega o último time step
        x = self.fc(x)
        return x