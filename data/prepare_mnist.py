#!/usr/bin/env python3
import os
import gzip
import numpy as np
from PIL import Image

# Configuration
INPUT_DIR = "../datasets/mnist_data"
IMG_OUTPUT_DIR = "../datasets/mnist_data/images"
ANN_OUTPUT_DIR = "../datasets/mnist_data/"
RANDOM_SEED = 50


def extract_images(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # Skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28, 28)  # Reshape to (num_images, 28, 28)


def extract_labels(filename):
    with gzip.open(filename, 'rb') as f:
        f.read(8)  # Skip header
        return np.frombuffer(f.read(), dtype=np.uint8)


def save_images(images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image)
        img.save(os.path.join(output_dir, f"{i:05d}.jpg"), quality=95)


def save_annotations(labels, output_path):
    with open(output_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i:05d}.jpg {label}\n")


def main():
    print("Processing MNIST dataset...")

    os.makedirs(os.path.join(IMG_OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(IMG_OUTPUT_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(IMG_OUTPUT_DIR, "test"), exist_ok=True)

    train_images = extract_images(os.path.join(
        INPUT_DIR, "train-images.gz"))
    train_labels = extract_labels(os.path.join(
        INPUT_DIR, "train-labels.gz"))

    test_images = extract_images(os.path.join(
        INPUT_DIR, "test-images.gz"))
    test_labels = extract_labels(os.path.join(
        INPUT_DIR, "test-labels.gz"))

    print("Saving training set...")
    save_images(train_images, train_labels,
                os.path.join(IMG_OUTPUT_DIR, "train"))
    save_annotations(train_labels, os.path.join(
        ANN_OUTPUT_DIR, "train_annotations.txt"))

    print("Saving test set...")
    save_images(test_images, test_labels, os.path.join(IMG_OUTPUT_DIR, "test"))
    save_annotations(test_labels, os.path.join(
        ANN_OUTPUT_DIR, "test_annotations.txt"))


if __name__ == "__main__":
    main()
