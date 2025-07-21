# NLP Experiments
This will contain the experiments for all of the developed NLP models. Running Group KAN requires a GPU because of its dependency on rational_kat_cu. Currently just the Dockerfile exists here and the training code is with the model code (need to migrate here since I reused another repo heavily)

## Current NLP Options
* BERT
* GPT
* KANBERT
* KANGPT

# nano{GPT|BERT}
This directory contains a GPT and BERT heavily based on [nanoGPT](https://github.com/karpathy/nanoGPT). Also has functionality for learning the activation using the package [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu). To use KAN with these models, a GPU must be available and if using docker, run with `--gpus=all`.



## rational_kat_cu
Implements KAT Group which is required for using KANBERT. Not sure if there is a required python version for this but I used 3.10


## Configuration
config/train_shakespeare_char.py
* Use this file to set configuration values such as hyperparameters.
* The nanoGPT repository has examples of finetuning and using a GPT tokenizer. The code may need to be updated to allow for finetuning.

The main things to adjust here:
* device (i.e, cpu, cuda)
* batch_size
* block_size
* n_layers
* n_heads
* n_embd
* max_iters

## Using
Should set cwd to nanoGPT or nanoBERT
### Train
```python
python train.py config/train_shakespeare_char.py
```
This will download the dataset, tokenize and train the model. Can set `retokenize=False` in the config file to use the last tokenized dataset (Might help speed up testing on larger datasets)

### Sample
```python
python sample.py --out_dir=out-shakespeare-char
```
For GPT this will generate several samples. For BERT this will print the original sequence and the sequence with BERTs predictions.
* Need to change the device for sampling in sample.py



## BERT Implementation Details (changes from nanoGPT)
* Uses full attention instead of casual.
    - The masks are added to the token sequence but maybe it should be added inside of the attention calculations
* Masks tokens to create labels instead of shifting by 1.
    - Does a masking of 15% randomly. This worked better in my tests than the original paper that does masking for 15% of tokens of those, 80% are random, 10% are incorrect and 10% are original token. (TEST)
    * Just my idea but might work better if this percent wasnt fixed and it changed during training. (TEST)

## Experiment Notes
* Originally used rational_kat_cu for KAN transformers but this isn't really KAN
* Need to test out efficient KAN some more. Initial experiments were quite slow still.

## Loader
These experiments use the NLPLoader dataloader.

## TODOs
* Clean up uneeded code from nanoGPT
* Fix the dataloading, I hacked together my dataloader with the original code and its ugly
* TODOs throughout code
* Rename GPTConfig to something more general


