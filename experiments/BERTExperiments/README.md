# Install
* Can use the included Dockerfile or just use the environment.yaml file for conda to run the ModernBERT based code

# Steps
## Download dataset and convert to proper format
```
cd CPE883-2025-02/dataloaders/BERTLoader/examples/examples/benchmarks/bert/
uv venv .venv --python=3.10
uv pip install -r requirements.txt
python src/convert_dataset.py --dataset Skylion007/openwebtext --out_root ./openwebtext-data --splits train val
```
- Move the folder with dataset to experiments/BERTExperiments/

https://github.com/mosaicml/examples/tree/main/examples/benchmarks/bert#prepare-your-data
* Link to the original code for help

## Select run configuration
* Select a configuration for the specific model
    - ModernBERT (tiny) - modernbert-base-pretrain.yaml
    - KANBERT - modernbert-base-pretrain-kan.yaml
    - Diffusion - modernbert-base-pretrain-diffusion.yaml


## Pretrain
composer main.py yamls/modernbert/modernbert-base-pretrain-diffusion.yaml


## Convert from .pt to hugging face model
convert_to_hf.py --output-name modernbert-kan-1.5b --output-dir ./output_model --input-checkpoint "./checkpoints/modernbert-base-pretrain/latest-rank0.pt"
                "--pytorch-bin"

## Finetune
finetune_modernbert_on_glue.ipynb has code for finetuning and evaluating on Glue tasks. Can also perform k-folds.


# Model Notes
## KANBERT
* Replaced the Linear layer in attention and the mlp layer with kan, also removed activation function
    - Left prediction layer the same, so maybe change that
* init_weights
    - skipping over kan in this function since kan does not have a weights/bias parameter to use

## DiffusionBERT
* not doing the 1% of batches with random sequence length since we are just doing classification and not generation
* Removed a potential optimization in the masked model forward function that is being used for diffusion
* Eval is done the same way as the masked language model.

## ModernBERT
* Changed the masking rules from using the 80-10-10 objective to doing full normal masking.

# TODO
* look into masking specifics for modernbert
* add eps as a config option?
* remove unescesary diffusion config options
* split code for github
* Using accuracy in stsb crashes out :(


# BERT
n_params=1.3300e+07
started:6:16pm
ended:7:28pm

# BERT full mask
started:4:31pm


# Diffusion
n_params=1.3300e+07
started:9:55pm 1/2 batch size

# KAN
* Would be 1.3329e+08 with same configuration
    - So using that decoder block added 100M parameters
    
* Final: 1.3872e+07

# Results

