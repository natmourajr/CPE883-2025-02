# ResNet, ViT, and CKAN Experiments on CIFAR-10/100

This README provides a comprehensive guide for running experiments with ResNet-50, Vision Transformer (ViT), and CKAN models on CIFAR-10 and CIFAR-100 datasets, including both 100 fine classes and 20 superclasses.

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
pip install torch torchvision numpy matplotlib seaborn scikit-learn tqdm
```

The dataset and model class (local) can be installed with: 

```bash
uv pip install -e dataloaders/cifar_dataloaders
uv pip install -e models/CKAN
```

## Directory Structure

```
experiments/CIFARExperiments/resnet-vit-ckan/
├── runner.py                  # Main runner script
├── runner.sh                  # Bash script for training/testing
├── test.sh                    # Bash script for test/confusion matrix
├── generate_confusion_matrix.py # Script to generate confusion matrix
├── result/                    # Output directory for logs, models, plots
├── checkpoints/               # Directory for model checkpoints
├── README.md                  # This file
└── ...
```

## Dataset Preparation

Download CIFAR-10 or CIFAR-100 datasets and place them in the appropriate directory (default: `CPE883-2025-02/datasets/cifar10/` or `cifar100/`).

If you want to use the 20 superclasses of CIFAR-100, the dataset loader supports this via the `superclass` argument.

## Training

You can train models using the provided shell script or directly via Python:

### Using the Shell Script

```bash
./runner.sh --model resnet --cifar_type 100 --class_type class
```

This will run training on CIFAR-100 (100 classes) with ResNet-50. You can change `--model` to `vit` or `ckan`, and adjust other arguments as needed.

#### Example for ViT on CIFAR-100 superclasses (20 classes):

```bash
./runner.sh --model vit --cifar_type 100 --class_type superclass
```

#### Example for CKAN on CIFAR-10:

```bash
./runner.sh --model ckan --cifar_type 10
```

### Using Python Directly

```bash
python runner.py \
	--epochs 100 \
	--batch_size 100 \
	--lr 0.005 \
	--folds 5 \
	--data_dir /path/to/cifar100 \
	--model vit \
	--cifar_type 100 \
	--class_type superclass
```

## Testing and Evaluation

To test a trained model and generate a confusion matrix:

### Using the Shell Script

Edit `test.sh` to set the correct weights path and options, then run:

```bash
./test.sh --model vit --cifar_type 100 --class_type superclass --weights /path/to/fold_0.pkl
```

### Using Python Directly

```bash
python generate_confusion_matrix.py \
	--batch_size 100 \
	--data_dir /path/to/cifar100 \
	--model vit \
	--cifar_type 100 \
	--class_type superclass \
	--weights /path/to/fold_0.pkl
```

This will print test accuracy/loss and save a confusion matrix in the result directory.

## Arguments and Options

| Argument         | Description                                      | Default         |
|------------------|--------------------------------------------------|-----------------|
| --epochs         | Number of training epochs                        | 100             |
| --batch_size     | Batch size                                       | 100             |
| --lr             | Learning rate                                    | 0.005           |
| --folds          | Number of cross-validation folds                  | 5               |
| --data_dir       | Path to dataset directory                         | (required)      |
| --model          | Model to use: resnet, vit, ckan                   | resnet          |
| --cifar_type     | '10' or '100'                                    | 10              |
| --class_type     | 'class' (100) or 'superclass' (20, CIFAR-100)     | class           |
| --weights        | Path to model weights for testing                 | None            |
| --test           | Set to 1 for test mode, 0 for training           | 0               |

## Results and Outputs

- Training logs: `result/<model>/fold_metrics.csv` (per-fold metrics)
- Model checkpoints: `checkpoints/<model>/fold_*.pkl`
- Plots: Loss/accuracy curves, confusion matrix
- Test results: `result/<model>/test_results.csv`

## Model Details

- **ResNet-50**: Standard ResNet-50 with final layer adapted for the number of classes.
- **ViT**: Vision Transformer (ViT-B/16) with head adapted for the number of classes. By default, only the last encoder layer and head are trainable.
- **CKAN**: Custom KAN-based convolutional model (see `models/CKAN/CKAN/ckan.py`).