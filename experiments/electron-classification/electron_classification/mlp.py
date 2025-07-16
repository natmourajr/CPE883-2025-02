import torch
import torch.nn as nn
import lightning as L
from typing import Tuple, List
from typer import Typer
import mlflow
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import tensor_dataset_from_df
from .misc import list_by_pattern, N_RINGS


class MLP(L.LightningModule):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

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


def run_fold(
    train_idx: List[int],
    val_idx: List[int],
    data: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str],
    batch_size: int
):
    train_dataset = tensor_dataset_from_df(
        data,
        feature_cols=feature_cols,
        label_cols=label_cols,
        idx=train_idx
    )
    val_dataset = tensor_dataset_from_df(
        data,
        feature_cols=feature_cols,
        label_cols=label_cols,
        idx=val_idx
    )
    datamodule = L.LightningDataModule.from_datasets(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
    )

    model = MLP(input_dim=N_RINGS)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator='cpu',
        devices=1
    )
    trainer.fit(model, datamodule=datamodule)


app = Typer(
    name='mlp',
    help='Utility for training MLP models on electron classification data.'
)


@app.command(
    help='Train an MLP model on electron classification data.'
)
def train(
    dataset_paths: List[Path],
    batch_size: int = 32,
    tracking_uri: str | None = None,
    experiment_name: str = 'electron-classification',
    seed: int = 42,
    feature_cols: List[str] = [f'ring_{i}' for i in range(N_RINGS)],
    label_cols: List[str] = ['label']
):
    tags = {
        'model': 'MLP'
    }
    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()

    dataset_files = list_by_pattern(dataset_paths, '*.parquet')
    load_cols = feature_cols + label_cols
    data = pd.read_parquet([str(f) for f in dataset_files],
                           engine='pyarrow',
                           columns=load_cols)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    with mlflow.start_run(run_name='MLP KFold Training', tags=tags):
        for fold, (train_idx, val_idx) in enumerate(cv.split(data, data['label'])):
            nested_run_tags = tags.copy()
            nested_run_tags['fold'] = fold
            with mlflow.start_run(run_name=f'MLP Fold {fold}',
                                  nested=True,
                                  tags=nested_run_tags):
                run_fold(
                    train_idx=train_idx,
                    val_idx=val_idx,
                    data=data,
                    feature_cols=feature_cols,
                    label_cols=label_cols,
                    batch_size=batch_size,
                )
