# nano{GPT|BERT}
This directory contains a GPT and BERT heavily based on [nanoGPT](https://github.com/karpathy/nanoGPT). Also has functionality for learning the activation using the package [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu). To use KAN with these models, a GPU must be available and if using docker, run with `--gpus=all`.

* I think it should be noted that KAT doesn't use KANs to replicate the transformer but instead just replaces the activation function with a learnable one.


## rational_kat_cu
Implements KAT Group which is required for using KANBERT. Not sure if there is a required python version for this but I used 3.10


## Configuration
nano{BERT|GPT}/config/train_shakespeare_char.py
* Use this file to set configuration values such as hyperparameters.
* The nanoGPT repository has examples of finetuning and using a GPT tokenizer

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
For GPT this will generate several samples. For BERT this will (TODO: Maybe generate samples of original and with guesses replacing MASK tokens)
* Need to change the device for sampling in sample.py

## BERT Implementation Details (changes from nanoGPT)
* Should use full self attention instead of casual.
* Masks tokens to create labels instead of shifting by 1.
    - Does a masking of 15% randomly. This worked better in my tests than the original paper that does masking for 15% of tokens of those, 80% are random, 10% are incorrect and 10% are original token. (TEST)
    * Just my idea but might work better if this percent wasnt fixed and it changed during training. (TEST)


## TODO
* Need to move training and sampling code into experiments
* Move configurations to a central point that can be shared between nanoGPT and nanoBERT?
* Read up on flash attention while using full attention
* Clean up uneeded code from nanoGPT
* Load same config for sampling from the config for training.
* Migrate training code to experiments
* Autodetect if GPU is available and load KAN Group
* Integrate NLPDataLoader into nanoGPT
* Fix the dataloading, I hacked together my dataloader with the original code and its ugly
* Maybe just combine the BERT and GPT code, pretty much all of it is the same, just might need to create separate trainer.py's
* Can I go back to normal dataloader?
* Does dockerfile execute without crashing?





