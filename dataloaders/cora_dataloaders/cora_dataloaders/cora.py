import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder


class CoraDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Load CORA dataset from cora.cites and cora.content files.

        Args:
            root_dir (str): Directory containing cora.cites and cora.content
            transform (callable, optional): Optional transform to be applied
        """
        super(CoraDataset, self).__init__(transform=transform)
        self.root_dir = root_dir
        self.data = self.load_data()

    def load_data(self):
        # Load node features and labels from cora.content
        content_path = os.path.join(self.root_dir, 'cora.content')
        paper_ids = []
        features = []
        labels = []

        with open(content_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                paper_ids.append(items[0])
                features.append([int(x) for x in items[1:-1]])
                labels.append(items[-1])

        # Convert to numpy arrays
        features = np.array(features, dtype=np.float32)

        # Encode labels as integers
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        labels = torch.from_numpy(labels).long()

        # Create paper_id to index mapping
        paper_to_idx = {paper_id: idx for idx,
                        paper_id in enumerate(paper_ids)}

        # Load edge indices from cora.cites
        cites_path = os.path.join(self.root_dir, 'cora.cites')
        edge_indices = []

        with open(cites_path, 'r') as f:
            for line in f:
                cited, citing = line.strip().split('\t')
                if cited in paper_to_idx and citing in paper_to_idx:
                    edge_indices.append(
                        [paper_to_idx[citing], paper_to_idx[cited]])

        # Convert to edge_index tensor
        edge_index = torch.tensor(
            edge_indices, dtype=torch.long).t().contiguous()

        # Create train/val/test masks (standard split: 140/500/1000)
        num_nodes = len(paper_ids)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Standard CORA split (first 140 train, next 500 val, last 1000 test)
        train_mask[:140] = True
        val_mask[140:640] = True
        test_mask[640:] = True

        # Create PyG Data object
        data = Data(
            x=torch.from_numpy(features),
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )

        return data

    def len(self):
        return 1  # Single graph dataset

    def get(self, idx):
        return self.data

    @property
    def num_node_features(self):
        return self.data.num_node_features

    @property
    def num_classes(self):
        return int(self.data.y.max().item()) + 1
