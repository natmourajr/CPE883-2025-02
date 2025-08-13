import torch
from cifar_dataloaders import CIFAR10Dataset
from capsnet import CapsuleNet, train, test, show_reconstruction


def load_cifar10(path="../../../datasets/cifar10", batch_size=100):
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_set = CIFAR10Dataset(root=path, train=True)
    test_set = CIFAR10Dataset(root=path, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, **kwargs
    )
    return train_loader, test_loader


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
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
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_loader, test_loader = load_cifar10(args.data_dir, batch_size=args.batch_size)

    # define model
    model = CapsuleNet(input_size=[3, 32, 32], classes=10, routings=3)
    model.cuda()
    print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))
    if not args.testing:
        train(model, train_loader, test_loader, args)
    else:  # testing
        if args.weights is None:
            print(
                "No weights are provided. Will test using random initialized weights."
            )
        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print("test acc = %.4f, test loss = %.5f" % (test_acc, test_loss))
        show_reconstruction(model, test_loader, 50, args)
