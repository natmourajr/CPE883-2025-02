from cifar import CIFAR10Loader

if __name__ == "__main__":
    # Set the path to the extracted CIFAR-10 directory
    root = "../../../datasets/cifar10"
    # Instantiate loader for training set
    train_loader = CIFAR10Loader(root=root, train=True)
    print(f"Train set size: {len(train_loader)}")
    img, label = train_loader[0]
    print(f"First train image shape: {img.shape}, label: {label}")

    # Instantiate loader for test set
    test_loader = CIFAR10Loader(root=root, train=False)
    print(f"Test set size: {len(test_loader)}")
    img, label = test_loader[0]
    print(f"First test image shape: {img.shape}, label: {label}")
