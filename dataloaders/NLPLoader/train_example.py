import torch
from torch.utils.data import DataLoader
from nlp_loader.loader import NLPDataLoader


# Device selection
if torch.cuda.is_available():
    device_type = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device_type = 'mps'
else:
    device_type = 'cpu'

device = torch.device(device_type)
print(f"Using device: {device}")


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
train_dataset = nlp_dataloader.dataset['train']
val_dataset = nlp_dataloader.dataset['val']

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device_type == 'cuda')
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device_type == 'cuda')
)

# 
for i in range(epochs):
    for x, y in train_loader:
        x = x.to(device, non_blocking=(device.type == 'cuda'))