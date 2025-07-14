from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd


class HeartDataset(Dataset):
    def __init__(self, directory: Path):
        """
        Args:
            directory (Path): directory with a .csv for Heart Disease dataset.
        """

        # Load the dataset
        self.data = pd.read_csv(directory)

        self.features = self.data.iloc[:, :-1].values
        self.targets = self.data.iloc[:, -1].values

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Args:
            idx (int): index of sample to return

        Returns:
            tuple: (features, target) where features is a numpy array
            and target is a scalar
        """
        features = self.features[idx]
        target = self.targets[idx]

        return features, target
