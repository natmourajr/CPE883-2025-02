# Source: https://arxiv.org/pdf/2406.06470
# https://github.com/WillHua127/GraphKAN-Graph-Kolmogorov-Arnold-Networks/blob/main/utils.py

import time
import torch
import torch.nn.functional as F
from cora_dataloaders import CoraDataset

# Initialize dataset (point to directory containing cora.cites and cora.content)
dataset = CoraDataset(
    root_dir='/home/eduardo/doc/CPE883-2025-02/datasets/cora')

# Get the single graph
data = dataset[0]

# Print dataset info
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of features: {dataset.num_node_features}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Training nodes: {data.train_mask.sum().item()}')
print(f'Validation nodes: {data.val_mask.sum().item()}')
print(f'Test nodes: {data.test_mask.sum().item()}')

time.sleep(5000)
# Initialize model
model = GKAN(in_feat, hidden_feat, out_feat, grid_feat)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    loss = train()
    if epoch % 20 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# Final evaluation
train_acc, val_acc, test_acc = test()
print(
    f'Final Results: Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
