import torch
import torch.nn as nn

class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)  # [batch, hidden_dim]

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)  # [batch, hidden_dim]

class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim=1):
        super().__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # output_dim = passos futuros

    def forward(self, branch_input, trunk_input):
        branch_out = self.branch_net(branch_input)   # [batch, hidden_dim]
        trunk_out = self.trunk_net(trunk_input)      # [batch, hidden_dim]

        combined = branch_out * trunk_out             # [batch, hidden_dim]
        out = self.output_layer(combined)             # [batch, output_dim]
        return out  # retorna [batch]