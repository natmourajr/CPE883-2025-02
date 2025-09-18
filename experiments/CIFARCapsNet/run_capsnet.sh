#!/bin/bash

# Run CapsNet training/testing on CIFAR-10
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CAPSNET_SCRIPT="$SCRIPT_DIR/cifar_capsnet_2.py"
# DATA_DIR="G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\Antonio_Alberto/data/cifar10"
# SAVE_DIR="$SCRIPT_DIR/result"

python cifar_capsnet_2.py \
  --data_dir G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\Antonio_Alberto/data/cifar10 \
  --save_dir G:\Meu Drive\Doutorado\Doutorado\CPE883-2025-02\Antonio_Alberto\experiments\CIFARCapsNet\result \
  --epochs 50 \
  --epochs_optuna 50 \
  --batch_size 128 \
  --batch_size_optuna 192 \
  --use_optuna \
  --n_trials 20 \
  --k_folds_optuna 5 \
  --timeout_optuna 18000
