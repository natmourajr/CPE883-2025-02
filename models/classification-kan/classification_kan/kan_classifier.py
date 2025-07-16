from kan import KAN
import torch
import torch.nn as nn
import lightning as L
from typing import Tuple


class KANClassifier(L.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kan = KAN(
            *args,
            **kwargs
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.kan(x)

    def predict_probabilities(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def training_step(self,
                      batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: torch.Tensor):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: torch.Tensor):
        # this is the validation loop
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
