import os
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Callable, Optional, Union
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T


class CIFAR10Dataset(Dataset):
    base_folder = "cifar-10-batches-py"
    train_list = [
        ("data_batch_1", None),
        ("data_batch_2", None),
        ("data_batch_3", None),
        ("data_batch_4", None),
        ("data_batch_5", None),
    ]
    test_list = [
        ("test_batch", None),
    ]
    meta = {"filename": "batches.meta", "key": "label_names", "md5": None}

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = str(root)
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = train

        self.data: Any = []
        self.targets = []
        for file_name, _ in self.train_list if self.is_train else self.test_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.float32) / 255.0  # Convert to float and scale

        # Convert to torch tensor and HWC to CHW
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Compose default transforms: normalization and random augmentations
        default_transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(0.2),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
            ]
        )

        # Apply user transform if provided, else use default
        if self.transform:
            img = self.transform(img)
        else:
            img = default_transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        target = torch.tensor(target)
        return img, target

    def __len__(self):
        return len(self.data)
