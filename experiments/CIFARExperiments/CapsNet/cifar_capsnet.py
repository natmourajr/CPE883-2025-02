import torch
from cifar_dataloaders import CIFAR10Dataset, CIFAR100Dataset
from capsnet import CapsuleNet, run_kfold, test, show_reconstruction
import torchvision.transforms as transforms


def load_cifar(cifar_type="10", path="../../../datasets/cifar10", model_type="vit"):
    n_classes = 10 if cifar_type == "10" else 100

    if model_type == "vit":
        transform = transforms.Resize(384)
    else:
        transform = None
    if cifar_type == "10":
        train_set = CIFAR10Dataset(root=path, train=True, transform=transform)
        test_set = CIFAR10Dataset(root=path, train=False, transform=transform)
    if cifar_type == "100":
        train_set = CIFAR100Dataset(root=path, train=True, transform=transform)
        test_set = CIFAR100Dataset(root=path, train=False, transform=transform)
    else:
        raise ValueError("cifar_type must be '10' or '100'")

    return train_set, test_set, n_classes


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network.")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument(
        "--lr_decay",
        default=0.9,
        type=float,
        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs",
    )
    parser.add_argument(
        "--lam_recon",
        default=0.0005 * 784,
        type=float,
        help="The coefficient for the loss of decoder",
    )
    parser.add_argument(
        "-r",
        "--routings",
        default=3,
        type=int,
        help="Number of iterations used in routing algorithm. should > 0",
    )  # num_routing should > 0
    parser.add_argument(
        "--shift_pixels",
        default=2,
        type=int,
        help="Number of pixels to shift at most in each direction.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory of data. If no data, use '--download' flag to download it",
    )
    parser.add_argument(
        "--download", action="store_true", help="Download the required data."
    )
    parser.add_argument("--save_dir", default="./result")
    parser.add_argument(
        "-t",
        "--testing",
        action="store_true",
        help="Test the trained model on testing dataset",
    )
    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        help="The path of the saved weights. Should be specified when testing",
    )
    parser.add_argument(
        "--cifar_type",
        default="10",
        type=str,
        choices=["10", "100"],
        help="CIFAR dataset to use",
    )

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_set, test_set, n_classes = load_cifar(
        cifar_type=args.cifar_type, path=args.data_dir, model_type="capsnet"
    )

    model = CapsuleNet(
        input_size=[3, 32, 32],
        classes=n_classes,
        routings=3,
    )
    model.cuda()
    print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))
    if not args.testing:
        run_kfold(
            train_set,
            args,
            input_size=[3, 32, 32],
            n_classes=n_classes,
            routings=3,
        )
    else:  # testing
        if args.weights is None:
            print(
                "No weights are provided. Will test using random initialized weights."
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=8
        )
        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print(f"test acc = {test_acc:.4f}, test loss = {test_loss:.5f}")
        show_reconstruction(model, test_loader, 50, args)
