"""
Collect the benchmark datasets

Datasets:

- W3: https://github.com/ricardovvargas/3w_dataset.git

Considerations:
    - 


version: 0.0.1
date: 02/07/2025

copyright Copyright (c) 2025

References:
[1]

"""

import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset

class Collector(Dataset):
    def __init__(self, root_dir, class_names=None, transform=None):
        """
        Args:
            root_dir (str): Root directory containing class subfolders with CSV files
            class_names (list[str], optional): List of class folder names in desired order.
                If None, will use sorted folder names.
            transform (callable, optional): Optional transform applied on data tensor.
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get class folder names if not provided
        if class_names is None:
            class_names = sorted([
                d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ])
        self.class_names = class_names
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        # List all (file_path, label) tuples
        self.samples = []
        for cls_name in class_names:
            cls_folder = os.path.join(root_dir, cls_name)
            csv_files = glob.glob(os.path.join(cls_folder, '*.csv'))
            for f in csv_files:
                self.samples.append((f, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        df = pd.read_csv(file_path)

        # Convert dataframe to float tensor: shape [timesteps, features]
        data = torch.tensor(df.values, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)

        return data, label


# Usage example:

train_root = 'data/train'  # adjust path to your extracted 3W train folder
test_root = 'data/test'    # adjust path to your extracted 3W test folder

train_dataset = Collector(train_root)
test_dataset = Collector(test_root)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Example iteration:
for x_batch, y_batch in train_loader:
    print(f'Data batch shape: {x_batch.shape}')  # [batch, timesteps, features]
    print(f'Label batch shape: {y_batch.shape}') # [batch]
    break