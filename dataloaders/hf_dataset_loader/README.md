# HF Dataset Loader

A Data Loader providing a unified interface for loading, preprocessing, and tokenizing datasets from the Huggingface Hub.

## Purpose

The `HFDataset` base class provides a standardized wrapper for Huggingface datasets with the following capabilities:

- **Unified Loading**: Consistent interface for loading datasets from Huggingface Hub
- **Preprocessing Pipeline**: Extensible preprocessing system for dataset-specific transformations
- **Tokenization Support**: Built-in tokenization with customizable formatting and parameters
- **Caching**: Efficient dataset caching to avoid repeated downloads
- **Lazy Loading**: Datasets are loaded only when accessed

## Core Features

- Load datasets with configurable parameters (name, config, text field)
- Apply custom preprocessing logic through inheritance
- Tokenize datasets with any Huggingface tokenizer
- Handle batched processing for efficiency
- Support for custom text formatting functions

## Available Dataset Implementations

### WikiTextDataset
- **Dataset**: `Salesforce/wikitext` (wikitext-103-raw-v1)
- **Preprocessing**: Filters out empty text entries

### TwitterFinancialNewsTopicDataset
- **Dataset**: `zeroshot/twitter-financial-news-topic`
- **Preprocessing**: Removes URLs and filters empty text entries

## Usage Example

```python
from hf_dataset_loader import WikiTextDataset
from transformers import AutoTokenizer

# Load dataset
dataset = WikiTextDataset(cache_dir="./cache")

# Get preprocessed dataset
ds = dataset.dataset

# Tokenize with a specific tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_ds = dataset.get_tokenized(
    tokenizer=tokenizer,
    max_length=512,
    truncation=True
)
```
