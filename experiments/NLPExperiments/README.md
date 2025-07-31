# NLP Experiments
This will contain the experiments for all of the developed NLP models. Running Group KAN requires a GPU because of its dependency on rational_kat_cu. Currently just the Dockerfile exists here and the training code is with the model code (need to migrate here since I reused another repo heavily)

## Current NLP Options
* BERT
* GPT
* KANBERT
* KANGPT
* DiffusionBERT

# nano{GPT|BERT}
This directory contains a GPT and BERT heavily based on [nanoGPT](https://github.com/karpathy/nanoGPT). Also has functionality for learning the activation using the package [rational_kat_cu](https://github.com/Adamdad/rational_kat_cu). To use KAN with these models, a GPU must be available and if using docker, run with `--gpus=all`.


## rational_kat_cu
Implements KAT Group which is required for using KANBERT. Not sure if there is a required python version for this but I used 3.10

## efficient_kan
Switched to using this for KAN since its a real version of KAN.


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
* attention
* dataloader


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
    - The masks are added to the token sequence but maybe it should be added inside of the attention calculations?
* Masks tokens to create labels instead of shifting by 1.
    - Does a masking of 15% randomly. This worked better in my tests than the original paper that does masking for 15% of tokens of those, 80% are random, 10% are incorrect and 10% are original token. (TEST)
    * Just my idea but might work better if this percent wasnt fixed and it changed during training. (TEST)

## KAN Notes
* Originally used rational_kat_cu for KAN transformers but this isn't really KAN.
* Needs to be tested more

## Diffusion Notes
D3PMs used a noise schedule but here I am just choosing a random number of tokens to mask during training like LLaDa does.
* I am not really sure which method is best. The method from LLaDa is easier to implement and probably faster.
### Sampling
Options:
* Choose top-k probable samples from each step
* Set timesteps as hyperparameter and use timesteps/batch_size to determine how many to unmask at each timestep
    - Maybe would be good to have training match but not sure it matters much

LLaDa:
- Chooses most probable tokens and goes until the sequence is complete
- Also does blockwise generation

DiffuLLama:
* unmasks based on a random probability, might be similar to scheduling
* Also does blocks at a time


## Loader
These experiments use the NLPLoader dataloader. There are 3 options for dataloaders:
* gpt: shifts y by 1 token forward of x (i.e abcd -> bcde).
* bert: Randomly selects 15% of tokens and masks them.
* diffusion: Randomly chooses a % of tokens to mask.

## Attention
There exists 3 types of attention
* full: Each token can see eachother.
* casual: Tokens can only look back.
* random: experimental, not fully defined at the moment.

## TODOs
* Clean up uneeded code from nanoGPT
* Fix the dataloading, I hacked together my dataloader with the original code and its ugly
* TODOs throughout code
* Rename GPTConfig to something more general
* Make the MASK token grabbed dynamically
* DiffusionDataset - self.eps - should this be configurable?
    - Maybe create a dataloader config too and use that
* For diffusionGPT is the only difference in sampling?
* Make the sampling code simpler and use 1 for all possible models.
* LLaDa sets 1% of pretraining data to [random length](https://github.com/ML-GSAI/LLaDA/blob/3f5e0d047382695c4c96224774d074d99e16befa/GUIDELINES.md)
    - I think it would be better and more efficient to just randomly switch to casual attention
* Need to test out efficient KAN some more. Initial experiments were quite slow still.
* For diffusion make a generator by blocks and allow it to do fill in like BERT


# Random Ideas
## Noisy Causual Masking
Use normal casual attention but add noise to the future tokens. Also test using an attention mask with values less than -inf to control how much is learned from the future tokens.

## Mixed attention
* Try using both casual and full attention


