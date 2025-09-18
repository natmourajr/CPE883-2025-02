import torch

class BaseModel(torch.nn.Module):
    """ A base class for PyTorch models."""
    def __init__(self):
        super().__init__()
        self.name = type(self).__name__

    def __str__(self) -> str:
        return self.name

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError