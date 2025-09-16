#!/bin/bash

# Run CapsNet testing on CIFAR-10
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPSNET_SCRIPT="$SCRIPT_DIR/cifar_capsnet.py"
DATA_DIR="/content/drive/MyDrive/Doutorado/Doutorado/CPE883-2025-02/Antonio_Alberto/data/cifar10"
SAVE_DIR="$SCRIPT_DIR/result"
WEIGHTS="$SAVE_DIR/trained_model_4.pkl"  # Change if you want to test a different weights file

# Default arguments
BATCH_SIZE=100
LR=0.001
LR_DECAY=0.9
LAM_RECON=1.536  # 0.0005*3072 = 3*32*32 for CIFAR-10
ROUTINGS=3

# Create result directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# Run testing
python3 "$CAPSNET_SCRIPT" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --lr_decay $LR_DECAY \
    --lam_recon $LAM_RECON \
    --routings $ROUTINGS \
    --data_dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --testing \
    --weights "$WEIGHTS"
