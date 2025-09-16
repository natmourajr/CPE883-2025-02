
import torch, platform
import optuna
import json, os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from cifar_dataloaders.cifar import CIFAR10Dataset
from models.capsnet.capsnet_3 import CapsuleNet, run_kfold, test, show_reconstruction, train

print("cuda disponível?", torch.cuda.is_available())

def load_cifar10(path="/content/drive/MyDrive/Doutorado/Doutorado/CPE883-2025-02/Antonio_Alberto/data/cifar10", batch_size=100):
    train_set = CIFAR10Dataset(root=path, train=True)
    test_set = CIFAR10Dataset(root=path, train=False)
    return train_set, test_set


def save_confusion_matrix(model, loader, class_names, save_dir, prefix="final_holdout"):
    """
    Gera e salva a matriz de confusão (PNG e CSV) a partir de um loader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, _ = model(x)
            preds = logits.argmax(1).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    # salvar CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"confusion_matrix_{prefix}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, row in enumerate(cm):
            f.write(class_names[i] + "," + ",".join(map(str, row.tolist())) + "\n")

    # salvar PNG
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    plt.title(f"Matriz de confusão - {prefix}")
    plt.tight_layout()
    png_path = os.path.join(save_dir, f"confusion_matrix_{prefix}.png")
    plt.savefig(png_path)
    plt.close(fig)
    print(f"[Confusion Matrix] Salva: {png_path} e {csv_path}")


class OptunaTuner:
    """Executa busca de hiperparâmetros (optimizer_name e lr) com Optuna."""
    def __init__(self, dataset, base_args, k_folds=2):
        from copy import deepcopy
        self.dataset = dataset
        self.base_args = deepcopy(base_args)
        self.k_folds = int(k_folds)

    def _build_model(self):
        model = CapsuleNet(input_size=[3, 32, 32], classes=10, routings=self.base_args.routings)
        model.cuda()
        return model

    def objective(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        from copy import deepcopy
        args = deepcopy(self.base_args)
        if getattr(args, "epochs_optuna", None):
            args.epochs = int(args.epochs_optuna)
        if getattr(args, "batch_size_optuna", None):
            args.batch_size = int(args.batch_size_optuna)

        args.optimizer_name = optimizer_name
        args.lr = float(lr)

        model = self._build_model()
        mean_acc = run_kfold(self.dataset, model, args, k=self.k_folds, shuffle=True)
        return float(mean_acc)

    def optimize(self, n_trials=20, timeout=None):
        study = optuna.create_study(direction="maximize", study_name="capsnet-cifar10")
        study.optimize(self.objective, n_trials=int(n_trials), timeout=None if timeout is None else int(timeout))

        print("===============================================================")
        print("\n===== RESULTADO OPTUNA =====")
        print(f"Melhor acurácia média (K-Fold): {study.best_value:.4f}")
        print("Melhores hiperparâmetros:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        best_path = os.path.join(self.base_args.save_dir, "best_params.json")
        with open(best_path, "w") as f:
            json.dump(study.best_params, f, indent=2)
        print(f"Best params salvos em {best_path}")
        return study


def run_single_training(dataset, model, args, val_split=0.1, shuffle=True):
    """
    Treina o modelo UMA vez (sem K-Fold).
    Se val_split > 0, faz holdout estratificado para validação; caso contrário, treina sem validação explícita.
    Retorna (best_val_acc, best_val_loss) quando há validação; se não houver, retorna (None, None).
    """
    import numpy as np
    import torch
    from torch.utils.data import Subset, DataLoader

    targets = np.array(dataset.targets)
    if val_split and float(val_split) > 0.0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=getattr(args, "final_shuffle_seed", 42))
        (train_idx, val_idx), = sss.split(np.zeros(len(targets)), targets)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
        best_val_acc, best_val_loss = train(model, train_loader, val_loader, args, fold_n=0, classes=dataset.classes)
        print(f"[Final único] Val Acc: {best_val_acc:.4f} | Val Loss: {best_val_loss:.5f}")
        # salvar matriz de confusão no holdout
        save_confusion_matrix(model, val_loader, dataset.classes, args.save_dir, prefix="final_holdout")
        return float(best_val_acc), float(best_val_loss)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
        _acc, _loss = train(model, loader, loader, args, fold_n=0, classes=dataset.classes)
        print("[Final único] Treino sem split de validação concluído.")
        return None, None


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Capsule Network on CIFAR10.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--epochs_optuna", default=None, type=int,
                        help="Se definido, número de épocas por trial do Optuna (sobrepõe --epochs durante a busca).")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--batch_size_optuna", default=None, type=int,
                        help="Se definido, usa este batch_size somente nos trials do Optuna.")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--lr_decay", default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument("--lam_recon", default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument("-r", "--routings", default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument("--shift_pixels", default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument("--data_dir", default="./data",
                        help="Directory of data. If no data, use '--download' flag to download it")
    parser.add_argument("--download", action="store_true", help="Download the required data.")
    parser.add_argument("--save_dir", default="./result")
    parser.add_argument("-t","--testing", action="store_true", help="Test the trained model on testing dataset")
    parser.add_argument("-w","--weights", default=None, help="The path of the saved weights. Should be specified when testing")

    parser.add_argument("--optimizer_name", default="Adam", type=str,
                        choices=["Adam", "RMSprop", "SGD"],
                        help="Otimizador a usar (quando não estiver usando Optuna).")
    parser.add_argument("--use_optuna", action="store_true",
                        help="Se setado, roda Optuna para otimizar hiperparâmetros.")
    parser.add_argument("--n_trials", default=20, type=int,
                        help="Número de trials do Optuna.")
    parser.add_argument("--k_folds_optuna", default=5, type=int,
                        help="Nº de folds K-Fold durante a busca do Optuna (default=2).")
    parser.add_argument("--k_folds_final", default=2, type=int,
                        help="Nº de folds K-Fold para o treino final (default=3).")
    parser.add_argument("--timeout_optuna", default=None, type=int,
                        help="Tempo máximo TOTAL (em segundos) para a busca do Optuna.")
    parser.add_argument("--final_no_kfold", action="store_true",
                        help="Se setado, a validação final NÃO usa K-Fold; treino único com holdout.")
    parser.add_argument("--final_val_split", default=0.2, type=float,
                        help="Proporção de validação no treino único final (padrão=0.2). Use 0 para treinar sem validação.")
    parser.add_argument("--final_shuffle_seed", default=42, type=int,
                        help="Seed do split estratificado para o treino único final.")
    parser.add_argument("--final_only", action="store_true",
                        help="Ignora qualquer busca do Optuna e roda apenas o treino final com os hiperparâmetros fornecidos.")
    parser.add_argument("--use_amp", action="store_true",
                    help="Ativa mixed precision (AMP) no treino para economizar VRAM.")


    args = parser.parse_args()
    print(args)
    
    if args.use_amp:
        print("[AMP] Mixed precision solicitado (--use_amp). O loop de treino usará autocast/GradScaler.")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_set, test_set = load_cifar10(args.data_dir, batch_size=args.batch_size)

    model = CapsuleNet(input_size=[3, 32, 32], classes=10, routings=args.routings)
    model.cuda()
    print(model)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights))

    if not args.testing:
        if args.final_only:
            print("Modo --final_only: pulando Optuna e executando apenas o treino final.")
            if args.final_no_kfold:
                run_single_training(train_set, model, args, val_split=args.final_val_split, shuffle=True)
            else:
                final_mean_acc = run_kfold(train_set, model, args, k=args.k_folds_final, shuffle=True)
                print(f"Acurácia média final ({args.k_folds_final}-Fold): {final_mean_acc:.4f}")
                print("Observação: use --final_no_kfold para gerar matriz de confusão direta do holdout.")
        elif args.use_optuna:
            tuner = OptunaTuner(train_set, args, k_folds=args.k_folds_optuna)
            study = tuner.optimize(n_trials=args.n_trials, timeout=args.timeout_optuna)

            best_optimizer = study.best_params.get("optimizer_name", args.optimizer_name)
            best_lr = study.best_params.get("lr", args.lr)
            print("===========================================================")
            print("\nTreinando modelo final com os melhores hiperparâmetros...")
            print("-------------------------------------final experiment with best parameters-----------------------------")
            args.optimizer_name = best_optimizer
            args.lr = float(best_lr)
            model = CapsuleNet(input_size=[3, 32, 32], classes=10, routings=args.routings)
            model.cuda()
            print("Melhores hiperparâmetros:")
            for k, v in study.best_params.items():
                print(f"  {k}: {v}")

            if args.final_no_kfold:
                run_single_training(train_set, model, args, val_split=args.final_val_split, shuffle=True)
            else:
                final_mean_acc = run_kfold(train_set, model, args, k=args.k_folds_final, shuffle=True)
                print(f"Acurácia média final ({args.k_folds_final}-Fold): {final_mean_acc:.4f}")
                print("Observação: use --final_no_kfold para gerar matriz de confusão direta do holdout.")
        else:
            final_mean_acc = run_kfold(train_set, model, args, k=args.k_folds_final, shuffle=True)
            print(f"Acurácia média final ({args.k_folds_final}-Fold): {final_mean_acc:.4f}")
            print("Observação: use --final_no_kfold para gerar matriz de confusão direta do holdout.")
    else:
        if args.weights is None:
            print("No weights are provided. Will test using random initialized weights.")
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False
        )
        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print(f"test acc = {test_acc:.4f}, test loss = {test_loss:.5f}")
        show_reconstruction(model, test_loader, 50, args)
