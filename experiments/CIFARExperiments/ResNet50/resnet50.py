import torch
import copy
import torch.nn as nn
import torch.optim as optim
from cifar_dataloaders import CIFAR10Dataset
from torchvision import models
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import random
import csv
from time import time
from torchvision.models import resnet50, ResNet50_Weights


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet50 Training")
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


def load_cifar10(path="../../../datasets/cifar10", batch_size=100):
    # kwargs = {"num_workers": 1, "pin_memory": True}
    train_set = CIFAR10Dataset(root=path, train=True)
    test_set = CIFAR10Dataset(root=path, train=False)
    return train_set, test_set


def create_model():
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)
    # print(model)
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
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def plot_curves(train_losses, train_accs, val_losses, val_accs, fold):
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
    plt.savefig(f"result/curves_fold_{fold}.png")


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

    train_set, test_set = load_cifar10(args.data_dir, batch_size=args.batch_size)
    targets = np.array(train_set.targets)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    # classes = train_set.classes
    fold_accuracies, fold_loss = [], []
    patience = 5

    # Prepare CSV logging for fold metrics
    csv_path = "result/fold_metrics.csv"
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
        model = create_model()
        criterion, optimizer, scheduler = create_criterion_and_optimizer(model, args.lr)
        train_subset = Subset(train_set, train_idx)
        val_subset = Subset(train_set, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
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
        plot_curves(train_losses, train_accs, val_losses, val_accs, fold)
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

    # Check best fold and run test
    best_fold = np.argmax(fold_accuracies)
    print(
        f"Best Fold: {best_fold}, Best Val Acc: {fold_accuracies[best_fold]:.4f}, Best Val Loss: {fold_loss[best_fold]:.4f}"
    )
    # load best fold and evaluate over test_set
    model.load_state_dict(
        torch.load(args.save_dir + f"checkpoints/fold_{best_fold}.pkl")
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    test_loss, test_acc = evaluate_one_epoch(model, criterion, test_loader)
    print(f"Test Acc: {test_acc:.4f} +- {np.std(fold_accuracies):.4f}")
    print(f"Test Loss: {test_loss:.4f} +- {np.std(fold_loss):.4f}")

    # Save test results to a new csv.
    # Add +- std as new column
    with open("results/test_results.csv", "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "fold",
                "test_acc",
                "test_loss",
                "test_acc_std",
                "test_loss_std",
            ],
        )
        writer.writerow(
            {
                "fold": best_fold,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "test_acc_std": np.std(fold_accuracies),
                "test_loss_std": np.std(fold_loss),
            }
        )


if __name__ == "__main__":
    main()
