# NLPDataloader
A Dataloader used to load sequences of text for NLP tasks.

## Supported Tokenizations
* char: A simple character level decoder based on all unique characters for a dataset
* gpt2: Uses the tokenizer library tiktoken from OpenAI
    - Kind of supported, has code to load it but did not implement masking for BERT.

## Supported Datasets
* TinyShakespeare: A simple dataset containing many plays written by shakespeare.

## Parameters
retokenize: Can load the dataset directly or for faster initialization, can load tokenized data directly from file.
tokenizer: How to tokenize the dataset.
dataset_name: Which data set to load.
block_size: Length of sequence to sample.
batch_size: How many samples to load at one time.
epochs: Number of times to go through the entire dataset.

## Example
```python
# Parameters
epochs = 10
tokenizer = 'char'
block_size = 32
dataset_name = 'shakespeare'
split_ratio = {"train": 0.9, "val": 0.1, "test": 0.0}
retokenize=True
batch_size = 128

# DataLoading
nlp_dataloader = NLPDataLoader(tokenizer, block_size, dataset_name, split_ratio, retokenize)

train_loader = DataLoader(
    nlp_dataloader.dataset['train'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device_type == 'cuda')
)
val_loader = DataLoader(
    nlp_dataloader.dataset['val'],
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device_type == 'cuda')
)

# Loads a random sequence
for i in range(epochs):
    for x, y in train_loader:
        x = x.to(device, non_blocking=(device.type == 'cuda'))

```