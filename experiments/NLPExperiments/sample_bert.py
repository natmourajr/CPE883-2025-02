"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from models import GPTConfig, BERT

from nlp_loader.loader import NLPDataLoader
from torch.utils.data import DataLoader


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # TODO revert to cuda
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Dataloader parameters
# TODO I don't like that some of these values live in 2 places like block_size
tokenizer = 'char'
dataset_name = 'shakespeare'
split_ratio = {"train": 0.9, "val": 0.1, "test": 0.0}
retokenize=True
block_size = 128
batch_size = 1
model_name = 'bert'

nlp_dataloader = NLPDataLoader(tokenizer, block_size, out_dir, model_name, dataset_name, split_ratio, False)
#out_dir = nlp_dataloader.get_out_dir()

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = BERT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = BERT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join(nlp_dataloader.data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['t2i'], meta['i2t']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, Exiting")
    exit()


val_dataset = nlp_dataloader.dataset['val']

data_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=(device_type == 'cuda')
)

data_iter = iter(data_loader)
num_samples = 5
total_correct = 0
total_predictions = 0
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            x, y = next(data_iter)
            original = y.clone()  # y contains the original tokens (with -100 for non-masked)
            
            # Create mask for positions that were actually masked (where y != -100)
            mask = (y != -100)  # This identifies the 15% of tokens that were masked
            
            predicted = model.predict_mask(x)
            
            predicted = predicted.to(device)
            mask = mask.to(device)
            original = original.to(device)
            
            # Calculate accuracy for this batch - only on masked positions
            correct_predictions = (predicted[mask] == original[mask]).sum().item()
            num_predictions = mask.sum().item()
            
            total_correct += correct_predictions
            total_predictions += num_predictions
            
            # Create a version with predictions filled in
            x_with_predictions = x.clone()
            x_with_predictions[mask] = predicted[mask]
            
            # For display, create the original unmasked version
            original_unmasked = x.clone()
            original_unmasked[mask] = original[mask]  # Replace mask tokens with original tokens

            print(f"Sample {k+1}:")
            print(f"Correct predictions: {correct_predictions}/{num_predictions} ({correct_predictions/num_predictions*100:.1f}%)")
            print("Original (unmasked):")
            print(decode(original_unmasked[0].tolist()))
            print("Input (with masks):")
            print(decode(x[0].tolist()))
            print("Predicted:")
            print(decode(x_with_predictions[0].tolist()))
            
            print('---------------')

# Print overall accuracy
overall_accuracy = total_correct / total_predictions * 100 if total_predictions > 0 else 0
print(f"\nOverall Accuracy: {total_correct}/{total_predictions} ({overall_accuracy:.1f}%)")
