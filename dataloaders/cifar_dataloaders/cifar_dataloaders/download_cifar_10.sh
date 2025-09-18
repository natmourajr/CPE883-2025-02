#!/bin/bash

# Download CORA from UCSC and extract into a structured format
TARGET_DIR="../../../datasets/cifar10"
mkdir -p "$TARGET_DIR"

echo "Downloading CIFAR-10..."
wget -q --show-progress https://www-cs-toronto-edu.translate.goog/~kriz/cifar-10-python.tar.gz -P "$TARGET_DIR"

echo "Extracting CIFAR-10..."
tar -xzf "$TARGET_DIR/cifar-10-python.tar.gz" -C "$TARGET_DIR"

echo "Cleaning up..."
rm "$TARGET_DIR/cifar-10-python.tar.gz"

echo "CIFAR-10 dataset ready in $TARGET_DIR"