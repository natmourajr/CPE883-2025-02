# models/s_deeponet_lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definição da Arquitetura da S-DeepONet (LSTM)

class BranchNet_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(BranchNet_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM para processar a série temporal
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x.shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)

        # Pegamos a saída do último passo de tempo
        output = lstm_out[:, -1, :]
        return self.out(output)


class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(TrunkNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, output_dim)

    def forward(self, y):
        return self.out(self.trunk(y))


class S_DeepONet_LSTM(nn.Module):
    def __init__(self, branch_input_dim, branch_hidden_dim, trunk_input_dim, trunk_hidden_layers, output_dim, num_layers_lstm, dropout=0.3):
        super(S_DeepONet_LSTM, self).__init__()

        self.branch_net = BranchNet_LSTM(branch_input_dim, branch_hidden_dim, num_layers_lstm, dropout)
        self.trunk_net = TrunkNet(trunk_input_dim, trunk_hidden_layers, branch_hidden_dim)

        self.final_fc = nn.Linear(branch_hidden_dim, output_dim)

    def forward(self, u, y):
        # u: Dados da janela de série temporal
        # y: Coordenadas de domínio (o tempo)

        branch_output = self.branch_net(u)
        trunk_output = self.trunk_net(y)

        # Combinação: multiplicamos a saída da branch com a da trunk
        # e somamos para obter um vetor final
        output = branch_output * trunk_output

        # Camada final de classificação
        output = self.final_fc(output)

        return output