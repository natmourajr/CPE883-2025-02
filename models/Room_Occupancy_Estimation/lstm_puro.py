import torch.nn as nn

class Pure_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(Pure_LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        output = lstm_out[:, -1, :]  # saída do último passo de tempo
        return self.fc(output)
