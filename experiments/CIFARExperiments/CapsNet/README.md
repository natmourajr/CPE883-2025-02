# CapsNet for CIFAR-10/100 Experiments

This README provides a detailed guide for running Capsule Network (CapsNet) experiments on CIFAR-10 and CIFAR-100 datasets, including both 100 fine classes and 20 superclasses.

## Table of Contents
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Arguments and Options](#arguments-and-options)
- [Results and Outputs](#results-and-outputs)
- [Troubleshooting](#troubleshooting)

---

## Requirements

Install dependencies (Python 3.8+ recommended):

```bash
uv pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```

The dataset and model class (local) can be installed with: 

```bash
uv pip install -e dataloaders/cifar_dataloaders
uv pip install -e models/CapsNet
```

## Directory Structure

```
experiments/CIFARExperiments/CapsNet/
├── cifar_capsnet.py         # Main runner script
├── run_capsnet.sh           # Bash script for training
├── run_capsnet_test.sh      # Bash script for testing
├── result/                  # Output directory for logs, models, plots
├── README.md                # This file
└── ...
```

## Dataset Preparation

Download CIFAR-10 or CIFAR-100 datasets and place them in the appropriate directory (default: `../../../datasets/cifar10` or `../../../datasets/cifar100`).

To download both datasets, please refer to scripts located in the _dataloaders/cifar_dataloaders/_ directory.

If you want to use the 20 superclasses of CIFAR-100, the dataset loader supports this via the `superclass` argument.

## Training

You can train CapsNet using the provided shell script or directly via Python:

### Using the Shell Script

```bash
./run_capsnet.sh
```

This will run training on CIFAR-100 (default) with 100 classes. You can edit the script to change dataset, epochs, batch size, etc.

### Using Python Directly

```bash
python cifar_capsnet.py \
	--epochs 50 \
	--batch_size 100 \
	--lr 0.001 \
	--lr_decay 0.9 \
	--lam_recon 1.536 \
	--routings 3 \
	--data_dir ../../../datasets/cifar100 \
	--save_dir ./result \
	--cifar_type 100 \
	--class_type class
```

#### To use CIFAR-100 superclasses (20 classes):

```bash
python cifar_capsnet.py \
	--epochs 50 \
	--batch_size 100 \
	--lr 0.001 \
	--lr_decay 0.9 \
	--lam_recon 1.536 \
	--routings 3 \
	--data_dir ../../../datasets/cifar100 \
	--save_dir ./result \
	--cifar_type 100 \
	--class_type superclass
```

#### To use CIFAR-10:

```bash
python cifar_capsnet.py \
	--epochs 50 \
	--batch_size 100 \
	--lr 0.001 \
	--lr_decay 0.9 \
	--lam_recon 0.0005 \
	--routings 3 \
	--data_dir ../../../datasets/cifar10 \
	--save_dir ./result \
	--cifar_type 10
```

## Testing and Evaluation

To test a trained model and generate reconstructions and a confusion matrix:

### Using the Shell Script

Edit `run_capsnet_test.sh` to set the correct weights path and options, then run:

```bash
./run_capsnet_test.sh
```

### Using Python Directly

```bash
python cifar_capsnet.py \
	--batch_size 100 \
	--lr 0.001 \
	--lr_decay 0.9 \
	--lam_recon 1.536 \
	--routings 3 \
	--data_dir ../../../datasets/cifar100 \
	--save_dir ./result \
	--cifar_type 100 \
	--class_type superclass \
	--testing true \
	--weights /path/to/trained_model_0.pkl
```

This will print test accuracy/loss and save a confusion matrix and reconstructions in the result directory.

## Arguments and Options

| Argument         | Description                                      | Default         |
|------------------|--------------------------------------------------|-----------------|
| --epochs         | Number of training epochs                        | 50              |
| --batch_size     | Batch size                                       | 100             |
| --lr             | Learning rate                                    | 0.001           |
| --lr_decay       | Learning rate decay per epoch                     | 0.9             |
| --lam_recon      | Decoder loss coefficient (e.g. 1.536 for CIFAR)   | 1.536           |
| --routings       | Routing iterations in CapsNet                     | 3               |
| --data_dir       | Path to dataset directory                         | ./data          |
| --save_dir       | Directory to save results                         | ./result        |
| --cifar_type     | '10' or '100'                                    | 10              |
| --class_type     | 'class' (100) or 'superclass' (20, CIFAR-100)     | class           |
| --testing        | Test mode (true/None)                             | None            |
| --weights        | Path to model weights for testing                 | None            |

## Results and Outputs

- Training logs: `result/log.csv` (per-epoch loss, accuracy, time)
- Model checkpoints: `result/epoch*_fold*.pkl`, `result/trained_model_*.pkl`
- Plots: Loss/accuracy curves, confusion matrix, sample predictions
- Reconstructions: `result/real_and_recon.png`