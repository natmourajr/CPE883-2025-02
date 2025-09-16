
import os, json, argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torchvision
import torchvision.transforms as T
import torchvision.models as models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2430, 0.2610)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, test_tf


def load_cifar10(data_dir):
    train_tf, test_tf = get_transforms()
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=test_tf)
    return train_set, test_set


def build_model(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def get_optimizer(params, name, lr):
    if name == "Adam":
        return optim.Adam(params, lr=lr)
    if name == "RMSprop":
        return optim.RMSprop(params, lr=lr)
    if name == "SGD":
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Optimizer {name} not supported")


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def save_confusion_matrix(model, loader, class_names, save_dir, prefix="resnet18_final_holdout"):
    """
    Gera e salva a matriz de confusão (PNG e CSV) a partir de um loader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"confusion_matrix_{prefix}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(class_names[i] + "," + ",".join(map(str, row.tolist())) + "\n")

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Matriz de confusão - {prefix}")
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"confusion_matrix_{prefix}.png")
    plt.savefig(png_path)
    plt.close(fig)
    print(f"[Confusion Matrix] Salva: {png_path} e {csv_path}")


def train_single(model, train_loader, val_loader, args, fold_idx=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model.parameters(), args.optimizer_name, args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience, patience_counter = 10, 0

    disable_tqdm = getattr(args, "quiet", False)

    scaler = GradScaler()

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm):
        model.train()
        running = 0.0
        # ***Passo de treino usado no Colab (comentar no VS Code)
        # for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=disable_tqdm):
        #     x, y = x.to(DEVICE), y.to(DEVICE)
        #     optimizer.zero_grad()
        #     logits = model(x)
        #     loss = criterion(logits, y)
        #     loss.backward()
        #     optimizer.step()
        #     running += loss.item() * x.size(0)

        # ***Passo de treino usado no VS Code (Treino mixed precisio, AMP) - comentar se usar no Colab
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=disable_tqdm):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type="cuda"):                          # FP16 forward
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()             # backward em FP16
            scaler.step(optimizer)                    # atualiza pesos
            scaler.update()
            running += loss.item() * x.size(0)
        
        scheduler.step()
        train_loss = running / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if not disable_tqdm:
            print(f"==> fold {fold_idx}, epoch {epoch:02d}: loss={train_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"resnet18_best_fold{fold_idx}.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if not disable_tqdm:
                    print("Early stopping.")
                break

    return best_val_acc, best_val_loss, model, val_loader


def run_kfold(dataset, args, k=5, shuffle=True):
    targets = torch.tensor(dataset.targets)
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=11)

    fold_accs, fold_losses = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(torch.zeros(len(targets)), targets)):
        train_subset = Subset(dataset, tr_idx.tolist())
        val_subset   = Subset(dataset, va_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = build_model(num_classes=10)
        best_acc, best_loss, trained_model, vloader = train_single(model, train_loader, val_loader, args, fold_idx=fold)
        fold_accs.append(best_acc); fold_losses.append(best_loss)
        print(f"Fold {fold}: best val acc={best_acc:.4f}, loss={best_loss:.5f}")

        # salvar CM do fold
        try:
            class_names = getattr(dataset, 'classes', [str(i) for i in range(10)])
            save_confusion_matrix(trained_model, vloader, class_names, args.save_dir, prefix=f"resnet18_kfold_fold{fold}")
        except Exception as e:
            print(f"[CM] Aviso: não foi possível salvar matriz de confusão do fold {fold}: {e}")

    mean_acc = float(sum(fold_accs) / len(fold_accs))
    mean_loss = float(sum(fold_losses) / len(fold_losses))
    print(f"K-Fold mean acc: {mean_acc:.4f}")
    return mean_acc, mean_loss


def run_single_training(dataset, args, val_split=0.1, shuffle=True):
    targets = torch.tensor(dataset.targets)
    if val_split and val_split > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=getattr(args, "final_shuffle_seed", 42))
        (tr_idx, va_idx), = sss.split(torch.zeros(len(targets)), targets)
        train_subset = Subset(dataset, tr_idx.tolist())
        val_subset   = Subset(dataset, va_idx.tolist())
    else:
        tr_idx = torch.arange(len(targets)).tolist()
        train_subset = Subset(dataset, tr_idx)
        val_subset   = Subset(dataset, tr_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(num_classes=10)
    best_acc, best_loss, trained_model, vloader = train_single(model, train_loader, val_loader, args, fold_idx=0)

    # salvar métricas finais
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "final_metrics_resnet18.json"), "w") as f:
        json.dump({"val_acc": best_acc, "val_loss": best_loss}, f, indent=2)

    # salvar CM do holdout
    try:
        class_names = getattr(dataset, 'classes', [str(i) for i in range(10)])
        save_confusion_matrix(trained_model, vloader, class_names, args.save_dir, prefix="resnet18_final_holdout")
    except Exception as e:
        print(f"[CM] Aviso: não foi possível salvar matriz de confusão do holdout: {e}")

    return best_acc, best_loss


class OptunaTuner:
    def __init__(self, dataset, base_args, k_folds=2):
        self.dataset = dataset
        self.base_args = deepcopy(base_args)
        self.k_folds = int(k_folds)

    def objective(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        args = deepcopy(self.base_args)
        if getattr(args, "epochs_optuna", None):
            args.epochs = int(args.epochs_optuna)
        if getattr(args, "batch_size_optuna", None):
            args.batch_size = int(args.batch_size_optuna)
        args.optimizer_name = optimizer_name
        args.lr = float(lr)

        mean_acc, _ = run_kfold(self.dataset, args, k=self.k_folds, shuffle=True)
        return mean_acc

    def optimize(self, n_trials=20, timeout=None):
        import optuna
        study = optuna.create_study(direction="maximize", study_name="resnet18-cifar10")
        study.optimize(self.objective, n_trials=int(n_trials),
                       timeout=None if timeout is None else int(timeout))

        print("\n===== RESULTADO OPTUNA (ResNet18) =====")
        print(f"Melhor acurácia média (K-Fold): {study.best_value:.4f}")
        print("Melhores hiperparâmetros:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        os.makedirs(self.base_args.save_dir, exist_ok=True)
        best_path = os.path.join(self.base_args.save_dir, "best_params_resnet18.json")
        with open(best_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        print(f"Best params salvos em {best_path}")

        return study


def main(argv=None):
    parser = argparse.ArgumentParser(description="ResNet18 baseline CIFAR-10 (com Optuna e K-Fold).")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_dir", default="./result_resnet")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--epochs_optuna", default=None, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--batch_size_optuna", default=None, type=int)
    parser.add_argument("--optimizer_name", default="SGD", choices=["Adam","RMSprop","SGD"])
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--lr_decay", default=0.9, type=float)
    parser.add_argument("--quiet", action="store_true", help="Silencia barras do tqdm.")

    parser.add_argument("--use_optuna", action="store_true")
    parser.add_argument("--n_trials", default=20, type=int)
    parser.add_argument("--k_folds_optuna", default=2, type=int)
    parser.add_argument("--timeout_optuna", default=None, type=int)
    parser.add_argument("--final_no_kfold", action="store_true")
    parser.add_argument("--final_val_split", default=0.1, type=float)
    parser.add_argument("--final_shuffle_seed", default=42, type=int)

    args = parser.parse_args(argv)
    os.makedirs(args.save_dir, exist_ok=True)

    print("CUDA:", torch.cuda.is_available())
    train_set, test_set = load_cifar10(args.data_dir)

    if args.use_optuna:
        tuner = OptunaTuner(train_set, args, k_folds=args.k_folds_optuna)
        study = tuner.optimize(n_trials=args.n_trials, timeout=args.timeout_optuna)

        best_opt = study.best_params.get("optimizer_name", args.optimizer_name)
        best_lr  = float(study.best_params.get("lr", args.lr))
        print("\nTreino final com melhores hiperparâmetros...")
        args.optimizer_name, args.lr = best_opt, best_lr

    if args.final_no_kfold:
        acc, loss = run_single_training(train_set, args, val_split=args.final_val_split, shuffle=True)
        print(f"[FINAL ÚNICO] Val Acc: {acc}")
    else:
        mean_acc, mean_loss = run_kfold(train_set, args, k=5, shuffle=True)
        with open(os.path.join(args.save_dir, "final_metrics_resnet18.json"), "w") as f:
            json.dump({"kfold_mean_acc": mean_acc, "kfold_mean_loss": mean_loss}, f, indent=2)

    print("Concluído.")


if __name__ == "__main__":
    main()
