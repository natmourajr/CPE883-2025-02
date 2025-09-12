"""
Pytorch implementation of CapsNet in paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       Launch `python CapsNet.py -h` for usage help

Result:
    Validation accuracy > 99.6% after 50 epochs.
    Speed: About 73s/epoch on a single GTX1070 GPU card and 43s/epoch on a GTX1080Ti GPU.

Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from .capsulelayers import DenseCapsule, PrimaryCapsule
import numpy as np
import csv
import math
from PIL import Image
from time import time
import matplotlib.pyplot as plt
import os
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CapsuleNet(nn.Module):
    """
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(
            input_size[0], input_size[1], kernel_size=3, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=input_size[1],
            out_channels=input_size[1] * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(
            input_size[1] * 2, 64, 8, kernel_size=5, stride=2, padding=0
        )

        in_num_caps = self._get_primary_caps_output_size()

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(
            in_num_caps=in_num_caps,  # 32 channels, 8x8 spatial size after convs
            in_dim_caps=8,
            out_num_caps=classes,
            out_dim_caps=16,
            routings=routings,
        )

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16 * classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def _get_primary_caps_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            # Passa pelo novo frontend sem pooling
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = self.primarycaps(x)
            return x.size(1)

    def forward(self, x, y=None):
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:
            index = length.max(dim=1)[1]
            y = torch.zeros(length.size(), device=device)
            y.scatter_(1, index.view(-1, 1), 1.0)
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = (
        y_true * torch.clamp(0.9 - y_pred, min=0.0) ** 2
        + 0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.0) ** 2
    )
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


def show_reconstruction(model, test_loader, n_images, args):
    model.eval()
    for x, _ in test_loader:
        x = x[: min(n_images, x.size(0))].to(DEVICE)
        with torch.no_grad():
            _, x_recon = model(x)
        data = np.concatenate([x.cpu().numpy(), x_recon.cpu().numpy()])
        img = combine_images(np.transpose(data, [0, 2, 3, 1]))
        image = img * 255
        Image.fromarray(image.astype(np.uint8)).save(
            args.save_dir + "/real_and_recon.png"
        )
        print()
        print("Reconstructed images are saved to %s/real_and_recon.png" % args.save_dir)
        print("-" * 70)
        break


def plot_sample_predictions(
    model, data_loader, class_names, save_dir, epoch, fold_n, device="cuda"
):
    os.makedirs(save_dir + "/predictions", exist_ok=True)
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
        outputs, _ = model(images)
        preds = outputs.max(1)[1]

    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()

    plt.figure(figsize=(12, 6))
    for i in range(min(16, len(images))):  # Show up to 16 images
        plt.subplot(2, 8, i + 1)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.247, 0.243, 0.261])
        img = images[i].transpose(1, 2, 0)
        img = img * std + mean  # Unnormalize
        img = img.clip(0, 1)
        plt.imshow(img)
        plt.title(f"T:{class_names[labels[i]]}\nP:{class_names[preds[i]]}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "predictions", f"epoch_{epoch}_fold{fold_n}.png")
    )
    plt.close()


def test(model, test_loader, args):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    n_classes = model.classes if hasattr(model, "classes") else 10
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_onehot = torch.zeros(y.size(0), n_classes, device=DEVICE).scatter_(
            1, y.view(-1, 1), 1.0
        )
        with torch.no_grad():
            y_pred, x_recon = model(x, y_onehot)
        test_loss += caps_loss(
            y_onehot, y_pred, x, x_recon, args.lam_recon
        ).item() * x.size(0)
        y_pred_label = y_pred.max(1)[1]
        correct += y_pred_label.eq(y).cpu().sum()
        all_preds.extend(y_pred_label.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    # Confusion matrix
    try:
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=False,  # No numbers
            cmap="crest",
            square=True,  # Square cells
            cbar_kws={"shrink": 0.8, "label": "Count"},
        )
        plt.xlabel("Predicted", fontsize=14)
        plt.ylabel("True", fontsize=14)
        plt.title("Confusion Matrix - capsnet", fontsize=16)
        plt.tight_layout()
        os.makedirs("result/capsnet", exist_ok=True)
        plt.savefig("result/capsnet/confusion_matrix_test.png", dpi=200)
        plt.close()
        print("Confusion matrix saved to result/capsnet/confusion_matrix_test.png")

    except Exception as e:
        print(f"Could not save confusion matrix: {e}")

    return test_loss, correct / len(test_loader.dataset)


def run_kfold(
    dataset,
    args,
    k=5,
    input_size=[3, 32, 32],
    n_classes=100,
    routings=3,
    shuffle=True,
):
    targets = np.array(dataset.targets)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=11)
    classes = dataset.classes
    print(f"Classes: {classes}, Number of classes: {n_classes}")
    fold_accuracies = []
    fold_loss = []
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(targets)), targets)
    ):
        model = CapsuleNet(
            input_size=input_size,
            classes=n_classes,
            routings=routings,
        )
        model.to(DEVICE)
        print(model)
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=shuffle, num_workers=8
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=8
        )
        best_val_acc, best_val_loss = train(
            model,
            train_loader,
            val_loader,
            args,
            fold,
            classes,
            n_classes,
        )
        fold_accuracies.append(best_val_acc)
        fold_loss.append(best_val_loss)
        print(
            f"Fold {fold}: Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}"
        )

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"K-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    mean_loss = np.mean(fold_loss)
    std_loss = np.std(fold_loss)
    print(f"K-Fold Loss: {mean_loss:.4f} ± {std_loss:.4f}")

    return


def train(model, train_loader, val_loader, args, fold_n, classes, n_classes):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print("Begin Training" + "-" * 70)

    logfile = open(args.save_dir + "/log.csv", "w")
    logwriter = csv.DictWriter(
        logfile,
        fieldnames=["epoch", "loss", "val_loss", "val_acc", "fold", "train_time"],
    )
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.0
    losses = []
    val_losses = []
    val_accs = []
    patience = 10
    patience_counter = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.epochs)):
        model.train()  # set to training mode
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        ):  # batch training
            x, y = x.to(DEVICE), y.to(DEVICE)
            y = torch.zeros(y.size(0), n_classes, device=DEVICE).scatter_(
                1, y.view(-1, 1), 1.0
            )

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`

        # compute validation loss and acc
        val_loss, val_acc = test(model, val_loader, args)
        epoch_loss = training_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_time = time() - ti
        logwriter.writerow(
            dict(
                epoch=epoch,
                loss=epoch_loss,
                val_loss=val_loss,
                val_acc=val_acc,
                fold=fold_n,
                train_time=train_time,
            )
        )
        logfile.flush()
        print(
            f"==> fold {fold_n}, epoch {epoch:02d}: loss={epoch_loss:.5f}, val_loss={val_loss:.5f}, val_acc={val_acc:.4f}, time={int(time() - ti)}s"
        )

        # Plot and save loss curve after every epoch
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(losses) + 1), losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("CapsNet Training/Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.save_dir + f"/loss_curve_fold{fold_n}.png")
        plt.close()

        # Plot and save accuracy curve after every epoch
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(val_accs) + 1),
            val_accs,
            label="Val Accuracy",
            color="tab:orange",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("CapsNet Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.save_dir + f"/accuracy_curve_fold{fold_n}.png")
        plt.close()

        plot_sample_predictions(
            model,
            val_loader,
            class_names=classes,
            save_dir=args.save_dir,
            epoch=epoch,
            fold_n=fold_n,
        )

        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), args.save_dir + f"/epoch{epoch}_fold{fold_n}.pkl"
            )
            print(f"best val_acc increased to {best_val_acc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{patience}"
            )
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    logfile.close()
    torch.save(model.state_dict(), args.save_dir + f"/trained_model_{fold_n}.pkl")
    print("Trained model saved to '%s/trained_model_%d.pkl'" % (args.save_dir, fold_n))
    print("Total time = %ds" % (time() - t0))
    print("End Training" + "-" * 70)

    return best_val_acc, best_val_loss


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros(
        (height * shape[0], width * shape[1]), dtype=generated_images.dtype
    )
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0] : (i + 1) * shape[0], j * shape[1] : (j + 1) * shape[1]] = (
            img[:, :, 0]
        )
    return image
