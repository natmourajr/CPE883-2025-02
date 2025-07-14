import argparse
from pathlib import Path
import torch
from kan import *
from sklearn.metrics import accuracy_score
# from heart_dataloaders import HeartDataset
from sklearn.model_selection import KFold
import numpy as np
from heart_dataloaders import HeartDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train KAN model on Heart Disease dataset")
    parser.add_argument("--data_path",
                        type=str,
                        default="/home/eduardo/doc/CPE883-2025-02/datasets/heart_disease/full/heart_disease.csv",
                        help="Path to heart disease dataset CSV file")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="Batch size for training")
    parser.add_argument("--grid_size",
                        type=int,
                        default=1,
                        help="Number of grid points for KAN")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Order of B-splines for KAN")
    parser.add_argument("--steps",
                        type=int,
                        default=2,
                        help="Number of training steps")
    parser.add_argument("--folds",
                        type=int,
                        default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    return parser.parse_args()


def prepare_kan_data(loader, device):
    """Prepare data in the format expected by KAN"""
    features = []
    labels = []
    for batch_features, batch_labels in loader:
        features.append(batch_features)
        labels.append(batch_labels)

    features = torch.cat(features).float().to(device)
    labels = torch.cat(labels).float().unsqueeze(1).to(device)
    return features, labels


def normalize(tensor):
    """Normalize input features"""
    return (tensor - tensor.mean(dim=0)) / (tensor.std(dim=0) + 1e-8)


def train_acc(model, dataset):
    preds = torch.sigmoid(model(dataset['train_input'])) > 0.5
    labels = dataset['train_label'].squeeze()

    score = accuracy_score(labels.cpu(), preds.cpu())

    return torch.tensor(score)


def test_acc(model, dataset):
    preds = torch.sigmoid(model(dataset['test_input'])) > 0.5
    labels = dataset['test_label'].squeeze()

    score = accuracy_score(labels.cpu(), preds.cpu())

    return torch.tensor(score)


def create_dataset_from_split(all_features,
                              all_labels,
                              train_idx,
                              test_idx,
                              device):

    # Split data
    train_features, val_features = all_features[train_idx], all_features[test_idx]
    train_labels, val_labels = all_labels[train_idx], all_labels[test_idx]

    # Convert to tensors
    train_input = torch.from_numpy(train_features).float().to(device)
    train_label = torch.from_numpy(
        train_labels).float().unsqueeze(1).to(device)
    val_input = torch.from_numpy(val_features).float().to(device)
    val_label = torch.from_numpy(
        val_labels).float().unsqueeze(1).to(device)

    dataset = {
        'train_input': normalize(train_input),
        'test_input': normalize(val_input),
        'train_label': train_label,
        'test_label': val_label,
    }

    shape = train_input.shape[1]
    return dataset, shape


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load
    dataset_path = Path(args.data_path)
    heart_dataset = HeartDataset(directory=dataset_path)

    # Split
    all_features = np.stack([x[0] for x in heart_dataset])
    all_labels = np.stack([x[1] for x in heart_dataset])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    best_accuracy = 0.0

    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_features)):
        print(f"\n=== Fold {fold + 1} ===")
        dataset, shape = create_dataset_from_split(all_features,
                                                   all_labels,
                                                   train_idx,
                                                   test_idx,
                                                   device)

        model = KAN(
            width=[shape, 1],
            grid=args.grid_size,
            k=args.k,
            device=device,
        )

        # Metric functions
        def current_train_acc():
            return train_acc(model, dataset)

        def current_val_acc():
            return test_acc(model, dataset)

        # Train the model
        results = model.fit(
            dataset,
            opt="LBFGS",
            steps=args.steps,
            metrics=(current_train_acc, current_val_acc),
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            lr=0.01,
            lamb=0.001
        )

        # Store fold results
        fold_accuracy = results['current_val_acc'][-1]
        fold_results.append(fold_accuracy)

        print(f"Fold {fold + 1} test accuracy: {fold_accuracy:.4f}")

        # Symbolic regression
        print("\nPerforming symbolic regression...")
        lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log',
               'sqrt', 'tanh', 'sin', 'tan', 'abs']
        model.prune()
        model.auto_symbolic(lib=lib)
        formula = model.symbolic_formula()[0][0]
        print(f"Symbolic formula: {ex_round(formula, 4)}")

        # Track best model
        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy

    # Cross-validation results
    print("\n=== Cross-validation Results ===")
    print(
        f"Average test accuracy: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == "__main__":
    main()
