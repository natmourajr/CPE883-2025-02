# dataloader_cifar10.py

import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10_dataloaders(
    data_dir="./data",
    batch_size=64,
    num_workers=2,
    download=True,
    augment=True
):
    """
    Retorna DataLoaders de treino e teste para o dataset CIFAR-10.

    Args:
        data_dir (str): Caminho para salvar/carregar os dados.
        batch_size (int): Tamanho do lote.
        num_workers (int): Número de workers para carregamento paralelo.
        download (bool): Baixar automaticamente o dataset se não existir.
        augment (bool): Se True, aplica aumentação de dados no treino.

    Returns:
        train_loader, test_loader (DataLoader, DataLoader)
    """

    # Normalização padrão do CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Dataset de treino e teste
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_cifar10_dataloaders()

    # Exemplo: iterar sobre um batch
    images, labels = next(iter(train_loader))
    print(f"Batch de imagens: {images.shape}")
    print(f"Batch de rótulos: {labels.shape}")
