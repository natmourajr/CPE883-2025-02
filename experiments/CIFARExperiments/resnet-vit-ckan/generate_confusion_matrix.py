import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cifar_dataloaders import CIFAR10Dataset, CIFAR100Dataset
from runner import create_model, load_cifar, parse_args
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    args = parse_args()
    # Load dataset
    _, test_set, n_classes = load_cifar(
        cifar_type=args.cifar_type,
        class_type=args.class_type,
        path=args.data_dir,
        model_type=args.model,
    )
    # Load model
    model = create_model(args.model, n_classes=n_classes)
    assert args.weights is not None, "Please provide --weights path to a checkpoint."
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # DataLoader
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=8
    )

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=False,  # No numbers
        cmap="crest",
        # linewidths=0.5,  # Thicker grid lines
        # linecolor="gray",
        square=True,  # Square cells
        cbar_kws={"shrink": 0.8, "label": "Count"},
    )
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    plt.title(f"Confusion Matrix - {args.model}", fontsize=16)
    plt.tight_layout()
    os.makedirs(f"result/{args.model}", exist_ok=True)
    plt.savefig(f"result/{args.model}/confusion_matrix_test.png", dpi=200)
    plt.close()
    print(f"Confusion matrix saved to result/{args.model}/confusion_matrix_test.png")
