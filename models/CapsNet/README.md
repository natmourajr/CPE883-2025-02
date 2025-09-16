# CapsNet Python Package

This package provides a flexible and configurable implementation of Capsule Networks (CapsNet) in PyTorch, including dynamic shape calculation and support for both classic and strided architectures.

## Installation

We recommend using [uv](https://github.com/astral-sh/uv) for fast and reproducible Python package management.

### 1. Install uv (if not already installed)

```bash
pip install uv
# or
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Install CapsNet and dependencies

Navigate to the CapsNet directory and run:

```bash
cd /path/to/models/CapsNet
uv pip install -e .
```

This will install all dependencies listed in `pyproject.toml` and make the package available for import.

## Usage

You can use the CapsNet model in your own scripts or notebooks. Example usage:

```python
import torch
from capsnet.capsnet_xray import CapsNet, CapsNetStrided

# Example config dictionary (adapt as needed)
model_config = {
	'preprocessing': {'image_size': 32},
	'architectures': {
		'CapsNet': {
			'frontend_channels': [3, 64, 128],
			'primary_caps_out_channels': 32,
			'primary_caps_dim': 8,
			'primary_caps_kernel_size': 9,
			'digit_caps_dim': 16,
			'routings': 3
		}
	}
}

model = CapsNet(model_config, num_classes=10, device="cuda")
model.eval()

# Dummy input
x = torch.randn(2, 3, 32, 32).to("cuda")
y_pred, recon = model(x)
print("Output shape:", y_pred.shape)
print("Reconstruction shape:", recon.shape)
```

## Features
- Dynamic shape calculation for primary capsules
- Configurable frontend and capsule layers
- Classic and strided architectures
- Ready for integration in research and production pipelines

## Development

To install in editable mode for development:

```bash
uv pip install -e .
```