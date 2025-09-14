from cifar import CIFAR100Dataset

if __name__ == "__main__":
    data_root = "datasets/cifar100"

    # Instantiate the dataset
    dataset = CIFAR100Dataset(root=data_root, train=True, superclass=True)

    # Print number of classes and class names
    print(f"Number of classes: {len(dataset.classes)}")
    print("Class names:")
    for idx, name in enumerate(dataset.classes):
        print(f"{idx}: {name}")

    # Print a few sample targets
    print("\nSample targets from the dataset:")
    for i in range(10):
        img, target = dataset[i]
        print(
            f"Sample {i}: class index={target.item()}, class name={dataset.classes[target.item()]}"
        )
