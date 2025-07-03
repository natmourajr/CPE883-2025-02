#!/bin/bash

URLS=(
    "https://azureopendatastorage.blob.core.windows.net/mnist/train-images-idx3-ubyte.gz"
    "https://azureopendatastorage.blob.core.windows.net/mnist/train-labels-idx1-ubyte.gz"
    "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-images-idx3-ubyte.gz"
    "https://azureopendatastorage.blob.core.windows.net/mnist/t10k-labels-idx1-ubyte.gz"
)

OUTPUT_NAMES=(
    "train-images.gz"
    "train-labels.gz"
    "test-images.gz"
    "test-labels.gz"
)

DATA_DIR="../datasets/mnist_data"
mkdir -p "$DATA_DIR"

for i in "${!URLS[@]}"; do
    url="${URLS[$i]}"
    output_file="$DATA_DIR/${OUTPUT_NAMES[$i]}"
    
    echo "downloading $url..."
    if ! wget -q --show-progress -O "$output_file" "$url"; then
        echo "Failed to download $url" >&2
        continue
    fi
    
    echo "done -> ${OUTPUT_NAMES[$i]}"
    echo
done

echo "running prepare_data.py..."
python3 prepare_mnist.py

echo "dataset saved at: $DATA_DIR/"
