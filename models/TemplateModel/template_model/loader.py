from torch.utils.data import Dataset

# According to PyTorch, a custom Dataset class must implement three
# methods: __init__, __len__, and __getitem__.


class TemplateDataset(Dataset):
    """Dataset class for the Template Loader"""

    def __init__(self,):
        self.dummy_data = [1, 2, 3, 4, 5]

    def __len__(self):
        return len(self.dummy_data)

    def __getitem__(self, idx):
        return self.dummy_data[idx]