from torch.utils.data import DataLoader
from modules.Dataloaders.MNIST_dataloaders import mnist
import os


data_dir = 'datasets/mnist_data/'
datasets = {
    'train': mnist.MNISTDataset(
        img_dir=os.path.join(data_dir, 'images/train'),
        annotation_file=os.path.join(data_dir, 'train_annotations.txt'),
    ),
    'val': mnist.MNISTDataset(
        img_dir=os.path.join(data_dir, 'images/val'),
        annotation_file=os.path.join(data_dir, 'val_annotations.txt'),
    ),
    'test': mnist.MNISTDataset(
        img_dir=os.path.join(data_dir, 'images/test'),
        annotation_file=os.path.join(data_dir, 'test_annotations.txt'),
    )
}


batch_size = 64
dataloaders = {
    'train': DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    ),
    'val': DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    ),
    'test': DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
}

if __name__ == '__main__':
    inputs, labels = next(iter(dataloaders['train']))

    print(f"Batch shape: {inputs.shape}")
    print(f"Labels: {labels[:10]}")

    print(f"Training set size: {len(datasets['train'])}")
    print(f"Validation set size: {len(datasets['val'])}")
    print(f"Test set size: {len(datasets['test'])}")
