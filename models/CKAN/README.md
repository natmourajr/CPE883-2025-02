# CKAN Python Package

This package provides an implementation of the CKAN (KAN-based Convolutional Neural Network) model in PyTorch, designed for image classification tasks such as CIFAR-10/100.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reproducible Python package management.

### 1. Install uv (if not already installed)

```bash
pip install uv
# or
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Install CKAN and dependencies

Navigate to the CKAN directory and run:

```bash
cd /path/to/models/CKAN
uv pip install -e .
```

This will install all dependencies listed in `pyproject.toml` and make the package available for import.

## Usage

You can use the CKAN model in your own scripts or notebooks. Example usage:

```python
import torch
from CKAN.ckan import CKAN

# Example: create a CKAN model for 32x32 images and 10 classes
model = CKAN(image_size=32, num_classes=10, device="cuda")
model.eval()

# Dummy input
x = torch.randn(2, 3, 32, 32).to("cuda")
logits = model(x)
print("Output shape:", logits.shape)
```

## Features
- KAN-based convolutional layers
- Simple, modular architecture
- Ready for integration in research and production pipelines

## Development

To install in editable mode for development:

```bash
uv pip install -e .
```