"""
Time series Capsnet for a regression problem based on the model Y. Kim et al.

Summary: Kan Classification of 3W pipeline data using P-MON-CKP and T-JUS-CKP as inputs.

Considerations:
    - 


version: 0.0.1
date: 13/07/2025

copyright Copyright (c) 2025

References:
[1] 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, num_capsules, capsule_dim, kernel_size, stride):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Convolution layer outputs (num_capsules * capsule_dim) channels
        self.conv = nn.Conv2d(
            in_channels, 
            num_capsules * capsule_dim, 
            kernel_size=kernel_size, 
            stride=stride
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Shape after conv: (B, num_capsules*capsule_dim, H, W)
        out = self.conv(x)
        
        # Flatten spatial dims H and W
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        # (B, num_capsules, capsule_dim, H*W)
        
        # Transpose to get capsules as vectors: (B, num_capsules * H*W, capsule_dim)
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, -1, self.capsule_dim)
        
        # Squash function as in CapsNet paper
        out = self.squash(out)
        return out
    
    def squash(self, x, epsilon=1e-7):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + epsilon)


class DigitCapsules(nn.Module):
    def __init__(self, num_primary_caps, primary_dim, num_output_caps, output_dim, routing_iters=3):
        super().__init__()
        self.num_primary_caps = num_primary_caps
        self.primary_dim = primary_dim
        self.num_output_caps = num_output_caps
        self.output_dim = output_dim
        self.routing_iters = routing_iters
        
        # Weight matrix transforms each primary capsule to each output capsule
        self.W = nn.Parameter(
            0.01 * torch.randn(1, num_primary_caps, num_output_caps, output_dim, primary_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_primary_caps, primary_dim)
        batch_size = x.size(0)
        
        # Expand input for multiplication: (B, num_primary_caps, 1, primary_dim, 1)
        x = x.unsqueeze(2).unsqueeze(-1)
        
        # Tile W to batch size: (B, num_primary_caps, num_output_caps, output_dim, primary_dim)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        
        # Compute "u_hat": predicted output capsules
        # Matrix multiplication over last dim: (B, num_primary_caps, num_output_caps, output_dim, 1)
        u_hat = torch.matmul(W, x)
        u_hat = u_hat.squeeze(-1)  # (B, num_primary_caps, num_output_caps, output_dim)
        
        # Routing logits (initially zeros)
        b_ij = torch.zeros(batch_size, self.num_primary_caps, self.num_output_caps, device=x.device)
        
        for iter in range(self.routing_iters):
            # Softmax over output capsules dimension
            c_ij = F.softmax(b_ij, dim=2)  # (B, num_primary_caps, num_output_caps)
            c_ij = c_ij.unsqueeze(-1)       # (B, num_primary_caps, num_output_caps, 1)
            
            # Weighted sum over primary capsules
            s_j = (c_ij * u_hat).sum(dim=1)  # (B, num_output_caps, output_dim)
            
            # Squash output capsules
            v_j = self.squash(s_j)
            
            if iter < self.routing_iters - 1:
                # Update b_ij based on agreement
                # Agreement: dot product between u_hat and v_j
                v_j_expanded = v_j.unsqueeze(1)  # (B, 1, num_output_caps, output_dim)
                agreement = (u_hat * v_j_expanded).sum(dim=-1)  # (B, num_primary_caps, num_output_caps)
                b_ij = b_ij + agreement
        
        return v_j  # (B, num_output_caps, output_dim)
    
    def squash(self, x, epsilon=1e-7):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + epsilon)


class CapsNetRegressor(nn.Module):
    def __init__(self, input_shape, predict_steps=10):
        super().__init__()

        # 1st convolution: 1 input channel → 32 output channels, kernel 3x3, padding 1 (preserves H,W)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # 2nd convolution: 32 input channels → 32 output channels, kernel 3x3, padding 1
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Primary capsules: from 32 input channels, create 8 capsule types, each capsule 8D,
        # kernel size 3x3, stride=2 (reduces spatial size)
        self.primary_caps = PrimaryCapsules(32, num_capsules=8, capsule_dim=8, kernel_size=3, stride=2)

        # Use dummy input to compute number of primary capsules dynamically from shape
        dummy = torch.zeros(1, *input_shape)
        pcaps_out = self.primary_caps(self.conv2(self.conv1(dummy)))
        num_primary_caps = pcaps_out.size(1)  # number of capsules = batch x capsule count x dim

        # DigitCaps (here called TrafficCaps): routing capsules
        # from primary capsules to output capsule(s)
        self.traffic_caps = DigitCapsules(
            num_primary_caps=num_primary_caps,
            primary_dim=8,
            num_output_caps=1,  # only one output capsule (regression output)
            output_dim=16       # capsule output dimension
        )

        # Decoder for regression: maps 16D capsule vector → predict_steps outputs
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, predict_steps)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.primary_caps(x)
        x = self.traffic_caps(x)  # shape: (batch_size, 1, 16)
        x = x.squeeze(1)          # shape: (batch_size, 16)
        return self.decoder(x)    # final output: (batch_size, predict_steps)