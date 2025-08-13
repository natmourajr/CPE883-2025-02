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
from matplotlib import pyplot as plt
import csv
import math
from PIL import Image


class CapsuleNet(nn.Module):
    """
    A Capsule Network on CIFAR-10.
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
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=9, stride=1, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(
            256, 256, 8, kernel_size=9, stride=2, padding=0
        )

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(
            in_num_caps=32 * 8 * 8,  # 32 channels, 8x8 spatial size after convs
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

    def forward(self, x, y=None):
        device = x.device
        x = self.relu(self.conv1(x))
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
        x = x[: min(n_images, x.size(0))].to(next(model.parameters()).device)
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
        # plt.imshow(
        #     plt.imread(
        #         args.save_dir + "/real_and_recon.png",
        #     )
        # )
        # plt.show()
        break


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    device = next(model.parameters()).device
    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10, device=device).scatter_(
            1, y.to(device).view(-1, 1), 1.0
        )
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).item() * x.size(0)
        y_pred = y_pred.max(1)[1]
        y_true = y.max(1)[1]
        correct += y_pred.eq(y_true).cpu().sum()

    test_loss /= len(test_loader.dataset)
    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print("Begin Training" + "-" * 70)
    from time import time
    import csv

    logfile = open(args.save_dir + "/log.csv", "w")
    logwriter = csv.DictWriter(
        logfile, fieldnames=["epoch", "loss", "val_loss", "val_acc"]
    )
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y) in enumerate(train_loader):  # batch training
            device = next(model.parameters()).device
            y = torch.zeros(y.size(0), 10, device=device).scatter_(
                1, y.view(-1, 1), 1.0
            )
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.item() * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        logwriter.writerow(
            dict(
                epoch=epoch,
                loss=training_loss / len(train_loader.dataset),
                val_loss=val_loss,
                val_acc=val_acc,
            )
        )
        print(
            "==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
            % (
                epoch,
                training_loss / len(train_loader.dataset),
                val_loss,
                val_acc,
                time() - ti,
            )
        )
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + "/epoch%d.pkl" % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + "/trained_model.pkl")
    print("Trained model saved to '%s/trained_model.h5'" % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print("End Training" + "-" * 70)
    return model


def plot_log(filename, show=True):
    # load data
    keys = []
    values = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if keys == []:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                continue

            for _, value in row.items():
                values.append(float(value))

        values = np.reshape(values, newshape=(-1, len(keys)))

    fig = plt.figure(figsize=(4, 6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    epoch_axis = 0
    for i, key in enumerate(keys):
        if key == "epoch":
            epoch_axis = i
            values[:, epoch_axis] += 1
            break
    for i, key in enumerate(keys):
        if key.find("loss") >= 0:  # loss
            print(values[:, i])
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.title("Training loss")

    fig.add_subplot(212)
    for i, key in enumerate(keys):
        if key.find("acc") >= 0:  # acc
            plt.plot(values[:, epoch_axis], values[:, i], label=key)
    plt.legend()
    plt.grid()
    plt.title("Accuracy")

    # fig.savefig('result/log.png')
    if show:
        plt.show()


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
