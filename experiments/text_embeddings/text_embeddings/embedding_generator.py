"""
Perform Embedding extraction from data loader to HDF5 file.
"""
from datasets import Dataset, concatenate_datasets
import datetime
import h5py
from hf_dataset_loader import HFDataset
import numpy as np

from .embeddings import EmbeddingExtractor


def _extract_embeddings(
    dataset: Dataset,
    extractor: EmbeddingExtractor,
    batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_samples = len(dataset)
    embedding_size = extractor.embedding_size

    embeddings = np.empty((n_samples, embedding_size), dtype=np.float32)
    texts = np.empty(n_samples, dtype=object)
    labels = np.empty(n_samples, dtype=int)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_docs = dataset[start_idx:end_idx]

        batch_embeddings = extractor.get_embeddings_batch(batch_docs['text'])

        embeddings[start_idx:end_idx, :] = batch_embeddings.numpy()
        texts[start_idx:end_idx] = batch_docs['text']
        labels[start_idx:end_idx] = batch_docs['label']

    return embeddings, texts, labels


def extract_embeddings_to_hdf5(
    loader: HFDataset,
    extractor: EmbeddingExtractor,
    model_name: str,
    output_path: str,
    batch_size: int = 32,
):
    all_datasets = list(loader.dataset.values())
    concatenated = concatenate_datasets(all_datasets)

    embeddings, texts, labels = _extract_embeddings(
        dataset=concatenated,
        extractor=extractor,
        batch_size=batch_size,
    )

    with h5py.File(output_path, 'w') as h5f:

        h5f.attrs['created_at'] = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        h5f.attrs['model_name'] = model_name
        h5f.attrs['dataset_name'] = loader.dataset_name
        h5f.attrs['dataset_config'] = loader.dataset_config
        h5f.attrs['embedding_size'] = extractor.embedding_size
        h5f.attrs['num_samples'] = len(concatenated)

        h5f.create_dataset('embeddings', data=embeddings, compression="gzip", compression_opts=9)
        h5f.create_dataset('texts', data=texts.tolist(), compression="gzip", compression_opts=9)
        h5f.create_dataset('labels', data=labels, compression="gzip", compression_opts=9)

