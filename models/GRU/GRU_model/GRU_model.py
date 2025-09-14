import torch
import torch.nn as nn

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]        # pegar a saída do último passo da sequência
        out = self.fc(out)         # transformar para output_size
        return out