import torch
import copy
import torch.nn as nn
import torch.optim as optim
from cifar_dataloaders import CIFAR10Dataset
from CKAN import CKAN
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import random
import csv
from time import time
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights


import torchvision.transforms as transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
NUM_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Training")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument(
        "--data_dir",
        help="Directory of data.",
    )
    parser.add_argument(
        "--seed",
        default=11,
        type=int,
        help="Random seed for initialization",
    )
    parser.add_argument(
        "--model",
        default="resnet",
        type=str,
        choices=["resnet", "vit"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="The path of the saved weights. Should be specified when testing",
    )
    parser.add_argument(
        "-f",
        "--folds",
        default=5,
        type=int,
        help="The number of folds for cross-validation",
    )

    args = parser.parse_args()
    return args


def load_cifar10(path="../../../datasets/cifar10", model_type="vit"):
    # kwargs = {"num_workers": 1, "pin_memory": True}

    if model_type == "vit":
        transform = transforms.Resize(384)
    else:
        transform = None

    train_set = CIFAR10Dataset(root=path, train=True, transform=transform)
    test_set = CIFAR10Dataset(root=path, train=False, transform=transform)
    return train_set, test_set


def create_model(model_type, n_classes=10):
    if model_type == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model = model.to(DEVICE)

    if model_type == "vit":
        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        model = vit_b_16(weights=weights)
        model.heads = nn.Sequential(nn.Linear(model.heads.head.in_features, n_classes))
        # Freeze all layers
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last encoder layer and the head
        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in model.heads.parameters():
            param.requires_grad = True

        model = model.to(DEVICE)

    if model_type == "ckan":
        model = CKAN(32, n_classes)
        model = model.to(DEVICE)

    return model


def create_criterion_and_optimizer(model, lr):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Add scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return criterion, optimizer, scheduler


def train_one_epoch(
    model, criterion, optimizer, train_loader, current_epoch, total_epochs
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {current_epoch + 1}/{total_epochs}")
    ):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc


def evaluate_one_epoch(model, criterion, val_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_preds), np.array(all_labels)


def plot_curves(train_losses, train_accs, val_losses, val_accs, fold, model):
    plt.figure(figsize=(12, 4))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Over Epochs FOLD_{fold}")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Over Epochs FOLD_{fold}")
    plt.legend()

    os.makedirs("result", exist_ok=True)
    plt.savefig(f"result/{model}/curves_fold_{fold}.png")


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def main():
    # load args
    args = parse_args()
    set_seeds(args.seed)

    train_set, test_set = load_cifar10(args.data_dir, model_type=args.model)
    targets = np.array(train_set.targets)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_accuracies, fold_loss = [], []
    patience = 5

    # Prepare CSV logging for fold metrics
    csv_path = f"result/{args.model}/fold_metrics.csv"
    os.makedirs("result", exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["fold", "best_val_acc", "best_val_loss", "train_time_s"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        fold_start_time = time()
        # Reset optimizer and scheduler for each fold
        model = create_model(args.model)
        criterion, optimizer, scheduler = create_criterion_and_optimizer(model, args.lr)
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(train_set, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=10
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=10
        )
        best_val_acc = float("-inf")
        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        # Training loop per fold
        for epoch in range(args.epochs):
            model.train()

            train_loss, train_acc = train_one_epoch(
                model, criterion, optimizer, train_loader, epoch, args.epochs
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            val_loss, val_acc = evaluate_one_epoch(model, criterion, val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f"==> fold {fold}, epoch {epoch:02d}: loss={train_loss:.5f}, "
                f"val_loss={val_loss:.5f}, val_acc={val_acc:.4f}"
            )

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"best val_acc increased to {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(
                    f"No improvement in validation loss."
                    f" Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Save only the best model for this fold
        os.makedirs("checkpoints", exist_ok=True)
        print(
            f"Saving best model for fold {fold} (best_val_acc={best_val_acc:.4f}) ..."
        )
        if best_state_dict is not None:
            torch.save(
                best_state_dict,
                f"checkpoints/fold_{fold}.pkl",
            )
        else:
            print(f"Warning: No best_state_dict found for fold {fold}!")

        # Save curves
        plot_curves(train_losses, train_accs, val_losses, val_accs, fold, args.model)
        plt.close("all")

        print(
            f"Fold {fold}: Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}"
        )

        # Save fold metrics
        fold_accuracies.append(best_val_acc)
        fold_loss.append(best_val_loss)
        fold_time = time() - fold_start_time
        # Append metrics to CSV
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=["fold", "best_val_acc", "best_val_loss", "train_time_s"],
            )
            writer.writerow(
                {
                    "fold": fold,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "train_time_s": fold_time,
                }
            )

    print("Training complete, starting evaluation. ")

    # load result/fold_metrics.csv
    with open(f"result/{args.model}/fold_metrics.csv", "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fold_accuracies.append(float(row["best_val_acc"]))
            fold_loss.append(float(row["best_val_loss"]))

    # Check best fold and run test
    best_fold = np.argmax(fold_accuracies)
    print(
        f"Best Fold: {best_fold}, Best Val Acc: {fold_accuracies[best_fold]:.4f}, Best Val Loss: {fold_loss[best_fold]:.4f}"
    )
    # load best fold and evaluate over test_set
    model = create_model(args.model)
    model.load_state_dict(torch.load(f"checkpoints/fold_{best_fold}.pkl"))
    criterion, _, _ = create_criterion_and_optimizer(model, args.lr)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=10
    )
    test_loss, test_acc, test_preds, test_labels = evaluate_one_epoch(
        model, criterion, test_loader
    )
    print(f"Test Acc: {test_acc:.4f} +- {np.std(fold_accuracies):.4f}")
    print(f"Test Loss: {test_loss:.4f} +- {np.std(fold_loss):.4f}")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Test Set")
    plt.savefig(f"result/{args.model}/confusion_matrix_test.png")
    plt.close()

    # Save test results to a new csv.
    with open(f"result/{args.model}/test_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "test_acc",
                "test_loss",
                "test_acc_std",
                "test_loss_std",
            ],
        )
        writer.writerow(
            {
                "test_acc": test_acc,
                "test_loss": test_loss,
                "test_acc_std": np.std(fold_accuracies),
                "test_loss_std": np.std(fold_loss),
            }
        )


if __name__ == "__main__":
    main()
