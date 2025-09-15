"""
Train and use a BPE tokenizer
"""
from itertools import chain
from typing import Generator

import datasets
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def _gen_texts(*datasets) -> Generator[str, None, None]:
    for row in chain(*datasets):
        text = row['text'].strip()
        if text:
            yield text


def train_tokenizer(*datasets, vocab_size: int=50_000) -> Tokenizer:

    tokenizer = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=['[UNK]', '[PAD]', '[MASK]'],
        vocab_size=vocab_size,
        end_of_word_suffix=' '
    )

    tokenizer.train_from_iterator(
        _gen_texts(*datasets),
        trainer
    )

    return tokenizer


def train_from_hf_dataset(name: str, config: str):

    splits = datasets.get_dataset_split_names(name, config)

    dss = [datasets.load_dataset(name, config, split=split) for split in splits]

    return train_tokenizer(*dss)

