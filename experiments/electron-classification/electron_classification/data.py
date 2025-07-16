import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from typing import List
import torch


def tensor_dataset_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str] = ['label'],
    feature_type: str = 'float32',
    label_type: str = 'int64',
    idx: List[int] | None = None
) -> Dataset:
    if idx is not None:
        features = df.loc[idx, feature_cols].values.astype(feature_type)
        labels = df.loc[idx, label_cols].values.astype(label_type)
    else:
        features = df[feature_cols].values.astype(feature_type)
        labels = df[label_cols].values.astype(label_type)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    return dataset
