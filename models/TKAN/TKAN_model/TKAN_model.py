import torch
import torch.nn as nn


# class TKANLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, use_bias=True):
#         super(TKANLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim, bias=use_bias)
#         self.activation = nn.ReLU()

#     def forward(self, x):
#         batch_size, seq_len, input_dim = x.size()
#         x = x.reshape(batch_size * seq_len, input_dim)
#         x = self.linear(x)
#         x = self.activation(x)
#         x = x.reshape(batch_size, seq_len, -1)
#         return x
# # Modelo TKAN em PyTorch. O original está no Keras
# class TKANModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(TKANModel, self).__init__()
#         self.tkan1 = TKANLayer(input_dim, hidden_dim)
#         self.tkan2 = TKANLayer(hidden_dim, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # x: (batch_size, seq_len, 1)
#         x = self.tkan1(x)
#         x = self.tkan2(x)
#         x = x[:, -1, :]    # [batch, hidden] -> pega o último time step
#         x = self.fc(x)
#         return xS

# TKAN Puro Real
# class TKANLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, Q=4):
#         """
#         input_dim: número de features / tamanho da janela
#         hidden_dim: neurônios em cada φ e Φ
#         Q: número de funções Φ
#         """
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.Q = Q

#         # φ_{q,p}: pequenas redes univariadas para cada dimensão p e cada q
#         self.phi = nn.ModuleList([
#             nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(1, hidden_dim),
#                     nn.ReLU()
#                 ) for _ in range(input_dim)
#             ]) for _ in range(Q)
#         ])

#         # Φ_q: combinação das somas das φ
#         self.Phi = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             ) for _ in range(Q)
#         ])

#     def forward(self, x):
#         """
#         x: [batch, seq_len, input_dim]
#         """
#         batch_size, seq_len, input_dim = x.size()
#         assert input_dim == self.input_dim, "Input dim mismatch"

#         out = 0
#         for q in range(self.Q):
#             sum_phi = 0
#             for p in range(input_dim):
#                 # φ_{q,p} aplicado à dimensão p
#                 phi_out = self.phi[q][p](x[:, :, p:p+1])  # [batch, seq_len, hidden_dim]
#                 sum_phi = sum_phi + phi_out
#             # Φ_q aplicado à soma
#             Phi_out = self.Phi[q](sum_phi)  # [batch, seq_len, hidden_dim]
#             out = out + Phi_out  # soma sobre q
#         return out  # [batch, seq_len, hidden_dim]

# class TKANModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, Q=4):
#         super().__init__()
#         self.tkan_layer = TKANLayer(input_dim, hidden_dim, Q)
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         """
#         x: [batch, seq_len, input_dim]
#         """
#         tkan_out = self.tkan_layer(x)      # [batch, seq_len, hidden_dim]
#         last_step = tkan_out[:, -1, :]     # pegar último timestep
#         output = self.fc(last_step)        # [batch, output_dim]
#         return output


# TKAN + LSTM
class TKANLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, Q=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.Q = Q

        # φ_{q,p}: pequenas redes univariadas
        self.phi = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU()
                ) for _ in range(input_dim)
            ]) for _ in range(Q)
        ])

        # Φ_q: combinação das somas das φ
        self.Phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(Q)
        ])

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        out = 0
        for q in range(self.Q):
            sum_phi = 0
            for p in range(input_dim):
                phi_out = self.phi[q][p](x[:, :, p:p+1])  # [batch, seq_len, hidden_dim]
                sum_phi = sum_phi + phi_out
            Phi_out = self.Phi[q](sum_phi)  # [batch, seq_len, hidden_dim]
            out = out + Phi_out
        return out  # [batch, seq_len, hidden_dim]

# -----------------------------------
# Modelo híbrido TKAN + LSTM
# -----------------------------------
class TKANModel(nn.Module):
    def __init__(self, input_dim, tkan_hidden_dim, lstm_hidden_dim, output_dim, Q=4,
                 lstm_layers=1, dropout=0.2):
        super().__init__()
        self.tkan = TKANLayer(input_dim, tkan_hidden_dim, Q)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden_dim,
                            num_layers=lstm_layers, batch_first=True, dropout=dropout)
        # Concat TKAN embedding + LSTM hidden
        self.fc = nn.Linear(tkan_hidden_dim + lstm_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        tkan_out = self.tkan(x)          # [batch, seq_len, tkan_hidden_dim]
        lstm_out, _ = self.lstm(x)       # [batch, seq_len, lstm_hidden_dim]
        # Pega último timestep de cada
        tkan_last = tkan_out[:, -1, :]
        lstm_last = lstm_out[:, -1, :]
        combined = torch.cat([tkan_last, lstm_last], dim=1)
        combined = self.dropout(combined)
        out = self.fc(combined)
        return out