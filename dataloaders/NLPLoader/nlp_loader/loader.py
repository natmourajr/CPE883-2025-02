import numpy as np
import os
import pickle
import tiktoken
from torch.utils.data import Dataset
import torch
import requests

class NLPDataLoader:
    def __init__(self, tokenizer, block_size, out_dir, dataloader = "gpt", dataset_name='shakespeare', split_ratio={"train":0.9,"val":0.1,"test":0.0}, retokenize=True):
        # TODO maybe I should allow the user to specify the data_dir
        self.data_dir = out_dir
        self.dataset = {}
        self.mask = 65 # TODO get this dynamically
        
        self.download_dataset(dataset_name)

         # Retokenize dataset
        if retokenize:
            print(f"Retokenizing dataset {dataset_name} with tokenizer {tokenizer}")
            self.tokenize_file(f'{dataset_name}.txt', tokenizer, split_ratio)
        self.load_decoders()

        for split in split_ratio.keys():
            if split_ratio[split] > 0.0:
                if dataloader == "gpt":
                    self.dataset[split] = GPTDataset(
                        split,
                        block_size,
                        self.data_dir,
                    )
                elif dataloader == "bert":
                    self.dataset[split] = BERTDataset(
                    split,
                    block_size,
                    self.data_dir,
                    self.mask
                )
                elif dataloader == "diffusion":
                    self.dataset[split] = DiffusionDataset(
                    split,
                    block_size,
                    self.data_dir,
                    self.mask
                )
                else:
                    assert False, f"Unknown dataloader type: {dataloader}"

    def download_dataset(self, dataset_name):
        dataset_path = os.path.join(self.data_dir, f'{dataset_name}.txt')
        if not os.path.exists(dataset_path):
            print(f"Downloading dataset {dataset_name}...")
            if dataset_name == 'shakespeare':
                data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            elif dataset_name == 'wikitext2':
                raise NotImplementedError("Dataset 'wikitext2' is not implemented yet.")
            else:
                raise ValueError(f"Dataset {dataset_name} is not recognized or supported.")
            
            with open(dataset_path, 'w') as f:
                f.write(requests.get(data_url).text)
        else:
            print(f"Dataset {dataset_name} already exists at {dataset_path}.")

    def load_decoders(self):
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        assert os.path.exists(meta_path), f"Meta file not found at {meta_path}. Try retokenizing the dataset."

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.tokenizer = meta["encoder_name"]

        if self.tokenizer == 'char':
            id_to_token = meta['i2t']
            token_to_id = meta['t2i']
            self.vocab_size = meta['vocab_size']
            print(f"found vocab_size = {self.vocab_size} (inside {meta_path})")

            def encode(s):
                return [token_to_id[c] for c in s]

            def decode(ids):
                return ''.join([id_to_token[i] for i in ids])
            
            self.encode = encode
            self.decode = decode

            
        elif self.tokenizer == 'gpt2':
            print("Using GPT-2 tokenizer")
            enc = tiktoken.get_encoding(self.tokenizer)
            id_to_token = lambda idx: enc.decode([idx])
            token_to_id = lambda s: enc.encode(s)[0]

            # Used to convert between strings and tokens 
            self.encode = enc.encode
            self.decode = enc.decode
            
        # Testing functions to convert between single tokens and ids
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id

    # Tokenize data and save to file with metadata
    def tokenize_file(self, input_file_path, tokenizer, split_ratio={"train":0.9,"val":0.1,"test":0.0}):
        meta = {"encoder_name": tokenizer}

        with open(f'{self.data_dir}/{input_file_path}', 'r') as f:
            data = f.read()

        if tokenizer == 'char':
            special_tokens = ['[MASK]'] 
            token_set = sorted(list(set(data))) + special_tokens
            vocab_size = len(token_set)

            token_to_id = { ch:i for i,ch in enumerate(token_set) }
            id_to_token = { i:ch for i,ch in enumerate(token_set) }

            def encode(s):
                return [token_to_id[c] for c in s] 
            tokenized_data = encode(data)

            meta['vocab_size'] =  vocab_size
            meta['i2t'] = id_to_token
            meta['t2i'] = token_to_id

            self.mask = token_to_id['[MASK]']
            
        # TODO - this is deprecated at the moment, not sure if I need it but if I want to use then I need to add a [MASK] token.
        elif tokenizer == 'gpt2':
            enc = tiktoken.get_encoding(tokenizer)
            tokenized_data = enc.encode(data, disallowed_special=())
            
        self.create_data_files(tokenized_data, self.data_dir, split_ratio, meta)

    # Create data files from tokenized data and save as bins
    def create_data_files(self, tokenized_data, data_dir, split_ratio, meta):

        data_length = len(tokenized_data)

        # Split data
        train_end = int(data_length * split_ratio['train'])
        val_end   = train_end + int(data_length * split_ratio['val'])

        train_split = tokenized_data[:train_end]
        val_split   = tokenized_data[train_end:val_end]
        test_split  = tokenized_data[val_end:] if split_ratio['test'] > 0.0 else []

        # Save splits as .bin files
        np.array(train_split, dtype=np.uint16).tofile(os.path.join(self.data_dir, 'train.bin'))
        np.array(val_split, dtype=np.uint16).tofile(os.path.join(self.data_dir, 'val.bin'))
        if split_ratio['test'] > 0.0:
            np.array(test_split, dtype=np.uint16).tofile(os.path.join(self.data_dir, 'test.bin'))
            
        # Save tokenizer metadata
        with open(os.path.join(self.data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)


class GPTDataset(Dataset):
    def __init__(self, split, block_size, token_dir):
        self.block_size = block_size

        self.data = np.memmap(
            os.path.join(token_dir, f"{split}.bin"),
            dtype=np.uint16,
            mode='r'
        )

        self.length = len(self.data) - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

class BERTDataset(Dataset):
    def __init__(self, split, block_size, token_dir, mask):
        self.block_size = block_size
        self.mask = mask

        self.data = np.memmap(
            os.path.join(token_dir, f"{split}.bin"),
            dtype=np.uint16,
            mode='r'
        )

        self.length = len(self.data) - block_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size]
        x = torch.from_numpy(chunk.astype(np.int64))
        y = x.clone()

        mask = torch.rand(x.shape) < 0.15
        y[~mask] = -100         # only calculate loss on masked tokens
        x[mask] = self.mask 
        #y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

class DiffusionDataset(Dataset):
    def __init__(self, split, block_size, token_dir, mask):
        self.block_size = block_size
        self.mask = mask
        self.eps = 1e-3

        self.data = np.memmap(
            os.path.join(token_dir, f"{split}.bin"),
            dtype=np.uint16,
            mode='r'
        )

        self.length = len(self.data) - block_size

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load a chunk of tokens
        chunk = self.data[idx : idx + self.block_size]
        x = torch.from_numpy(chunk.astype(np.int64))  # (block_size,)

        # === Diffusion-style stochastic masking ===
        t = torch.rand(1)  # one value per example
        p_mask = (1 - self.eps) * t + self.eps  # scalar ∈ [eps, 1]
        p_mask = p_mask.item()  # convert to Python float

        # Apply the same p_mask across the sequence
        mask = torch.rand(x.shape) < p_mask

        y = x.clone()
        y[~mask] = -100 
        x[mask] = self.mask

        return x, y












   