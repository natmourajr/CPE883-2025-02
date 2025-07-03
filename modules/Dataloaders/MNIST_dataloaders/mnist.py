from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch


class MNISTDataset(Dataset):
    def __init__(self, img_dir, annotation_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images
            annotation_file (string): Path to the annotation file
            transform (callable, optional): Optional transform to be applied
        """
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.samples = []
        with open(annotation_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
