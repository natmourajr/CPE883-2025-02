import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import wandb

from models.mlp import MLP
from models.cnn import CNN
from models.deeponet import DeepONet
from models.ckan import CKAN
from training import Trainer, MultitaskTrainer, calculate_class_weights
from dataloaders.sonar.loader import load_raw_data
from dataloaders.sonar.signal.passivesonar import lofar
from dataloaders.sonar.signal.utils import resample
from dataloaders.sonar.data_handling import CustomDataloader, LoroCV
from visualization import plot_lofargram, plot_tsne_embeddings, palette
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')

def save_embeddings_and_targets(config, embeddings, all_targets, results_path):
    """
    Save each embedding and target array into separate .npy files.

    Parameters:
        embeddings (list of arrays): List of embedding arrays from each fold.
        all_targets (list of arrays): List of target arrays from each fold.
        results_path (Path): Path object to the directory where files will be saved.
    """
    for i, (embedding, target) in enumerate(zip(embeddings, all_targets)):
        embedding = np.array(embedding)
        target = np.array(target)

        np.save(results_path / "data" / f"embeddings_fold_{config.fold}.npy", embedding)
        np.save(results_path / "data" / f"all_targets_fold_{config.fold}.npy", target)

def load_data():
    datapath = Path("../data/4classes/")
    raw_data = load_raw_data(datapath)

    n_pts_fft = 1024
    n_overlap = 0
    decimation_rate = 3
    final_fs = 22050//decimation_rate

    raw_data = raw_data.apply(lambda rr: resample(rr['signal'], rr['fs'], final_fs = final_fs))
    lofar_data = raw_data.apply(lofar,
                                fs=final_fs,
                                n_pts_fft = n_pts_fft,
                                n_overlap=n_overlap,
                                spectrum_bins_left=512)

    class_map = {
        "ClassA": 0,
        "ClassB": 1,
        "ClassC": 2,
        "ClassD": 3,
    }

    trgt = np.concatenate([class_map[cls_name]*np.ones(Sxx.shape[0])
                    for cls_name, run in lofar_data.items()
                    for run_name, (Sxx, _, _) in run.items()])

    data = np.concatenate([ Sxx
                    for cls_name, run in lofar_data.items()
                    for run_name, (Sxx, _, _) in run.items()], axis=0)

    print("=" * 75)
    print("Completed Data Preprocessing with the Following Configuration:")
    print(f" - FFT Points               : {n_pts_fft}")
    print(f" - Window Overlap           : {n_overlap}")
    print(f" - Decimation Rate          : {decimation_rate}")
    print(f" - Final Sampling Frequency : {final_fs}")
    print()
    print("Data Shapes:")
    print(f" - Input Data Shape         : {data.shape}")
    print(f" - Target Data Shape        : {trgt.shape}")
    print("=" * 75)

    return lofar_data, data, trgt

def model_select(config, branch_net = None):
    window_size = config.window_size
    
    elif config.model_name == "MLP":
        return lambda input_size, coords: MLP(input_shape=input_size, hidden_channels=config.hidden_channels, n_targets=4, dropout=config.dropout)
    
    elif config.model_name == "DeepONet-MLP":
        return lambda input_size, coords: DeepONet(branch_net= MLP(input_shape=input_size,
                                                           hidden_channels=config.hidden_channels, 
                                                           n_targets=config.embedding_dim, 
                                                           dropout=config.dropout),
                                           trunk_net= MLP(input_shape=coords,
                                                           hidden_channels=config.hidden_channels, 
                                                           n_targets=config.embedding_dim, 
                                                           dropout=config.dropout))
    elif config.model_name == "CNN":
        return lambda input_size: CNN(input_shape=input_size,
                                      conv_n_neurons=config.conv_n_neurons,
                                      conv_activation=torch.nn.PReLU,
                                      conv_pooling=torch.nn.MaxPool2d,
                                      conv_pooling_size=config.conv_pooling_size,
                                      conv_dropout=config.conv_dropout,
                                      batch_norm=torch.nn.BatchNorm2d,
                                      kernel_size=config.kernel_size,
                                      has_class_head=True,
                                      hidden_channels=config.classification_n_neurons,
                                      n_targets=4,
                                      dropout=config.classification_dropout)
        
    elif config.model_name == "CKAN":
        return lambda input_size: CKAN(input_shape=input_size,
                                       window_size=window_size,
                                       grid_size=config.grid_size,
                                       dropout_rate=config.dropout)


    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def run_experiment(config, lofar_data, results_path, device):
    # Initialize the model, optimizer, and criterion
    alpha = config.alpha if hasattr(config, 'alpha') else None
    window_size = config.window_size

    if window_size is None:
        overlap = None
    elif window_size == 16:
        overlap = 14
    elif window_size == 32:
        overlap = 28
    else:
        raise ValueError(f"Window size {window_size} not recognized.")

    model_builder = model_select(config)
    # Perform cross-validation using LoroCV
    accuracies = []
    embeddings = []
    all_targets    = []
    lorocv_no_window = LoroCV(n_splits=5, window_size=window_size, overlap=overlap, random_seed=42)

    fold = config.fold
    for i, (X_train, y_train, X_test, y_test, _, _) in enumerate(lorocv_no_window.split(lofar_data)):
        if i != fold:
            continue
        # Compute class weights for loss balancing
        class_weights = calculate_class_weights(y_train).to(device)
        
        if config.model_name in ["CNN", "CKAN"]:
            X_train = np.expand_dims(X_train, axis=1)
            X_test = np.expand_dims(X_test, axis=1)
        
        # Create DataLoader instances for the fold
        is2d = window_size is not None and config.model_name != "MLP"
        train_dataset_fold = CustomDataloader(X_train, y_train, is2d=is2d, device=device)
        test_dataset_fold = CustomDataloader(X_test, y_test, is2d=is2d, device=device)
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=32, shuffle=True)
        test_loader_fold = DataLoader(test_dataset_fold, batch_size=32, shuffle=False)
        
        input_size = X_train.shape[1:] if config.model_name in ["CNN", "CKAN"] else X_train.shape[1]
        model_fold = model_builder(input_size).to(device)
        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_fold, gamma=0.93)
        clf_criterion_fold = torch.nn.CrossEntropyLoss(weight=class_weights)

        
        trainer_fold = Trainer(model_fold, optimizer_fold, scheduler, clf_criterion_fold,
                                num_epochs=100, verbose=True, wandb_logging=True)
        trainer_fold.train(train_loader_fold, test_loader_fold, patience=10)
        
        _, accuracy, precision, recall, f1, roc_auc, y_pred, y_target = trainer_fold.evaluate(test_loader_fold)
        wandb.log({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        })
        
        np.save(results_path / "data" / f"predictions_fold_{fold}.npy", y_pred)
        np.save(results_path / "data" / f"targets_fold_{fold}.npy", y_target)
        
        
        fold_embeddings, fold_targets, fold_scores = trainer_fold.evaluate_embeddings(test_loader_fold)
        embeddings.append(fold_embeddings)
        all_targets.append(fold_targets)

        accuracies.append(accuracy)

        wandb.log(fold_scores)
        
        fig, ax = plt.subplots(figsize=(12, 12/ 1.618))
        plot_tsne_embeddings(ax, fold_embeddings, fold_targets, palette=palette)
        plot_name = f"t-SNE_embeddings_fold_{i}"
        fig.savefig(results_path / "plots" / "png" / f"{plot_name}.png", bbox_inches='tight', dpi=300)
        fig.savefig(results_path / "plots" / "svg" / f"{plot_name}.svg", bbox_inches='tight')

        wandb.log({"t-SNE plot": wandb.Image(fig)})
        plt.close(fig)
        
    return embeddings, all_targets

def make_hp_name(config):
    alpha = config.alpha if hasattr(config, 'alpha') else 'na'
    latent_dim_size = config.latent_dim_size if hasattr(config, 'latent_dim_size') else 'na'
    output_size = config.output_size if hasattr(config, 'output_size') else 'na'
    window_size = config.window_size
    learning_rate = config.learning_rate
    

    if config.model_name == "MLP":
        hidden_str = '_'.join(map(str, config.hidden_channels))
        return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}"
    if config.model_name == "DeepONet-MLP":
         hidden_str = '_'.join(map(str, config.hidden_channels))
         return f"hidden_{hidden_str}_dropout_{config.dropout}_lr_{learning_rate}_embedding_{config.embedding_dim}"
    elif config.model_name == "CNN":
        return f"conv_neurons_{config.conv_n_neurons}_pooling_{config.conv_pooling_size}_dropout_{config.conv_dropout}_kernel_{config.kernel_size}_class_neurons_{config.classification_n_neurons}_class_dropout_{config.classification_dropout}_lr_{learning_rate}"
    elif config.model_name == "CKAN":
        return f"window_{window_size}_grid_{config.grid_size}_dropout_{config.dropout}_lr_{learning_rate}"
    else:
        raise ValueError(f"Model name {config.model_name} not recognized.")

def has_been_run(hash):
    hash_file = "config_hashes.txt"
    if not os.path.exists(hash_file):
        return False
    with open(hash_file, "r") as file:
        existing_hashes = file.read().split()
    return hash in existing_hashes

def store_hash(hash):
    with open("config_hashes.txt", "a") as file:
        file.write(hash + "\n")

def sweep_experiment(project_name):
    wandb.init(project=project_name)
    config = wandb.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = config.model_name
    hp_name = make_hp_name(config)
    fold = config.fold

    model_id = f"{model_name}_{hp_name}"
    model_hash = f"Fold_{fold}_{model_id}"

    if has_been_run(model_hash):
        print("Configuration has already been run. Skipping...")
        wandb.log({"duplicate": True})
        return
    config.model_id = model_id

    if args.debug:
        results_path = Path(f"./results/debug/{config.model_name}/{hp_name}")
    else:
        results_path = Path(f"./results/production/{config.model_name}/{hp_name}")

    (results_path / "plots" / "svg").mkdir(parents=True, exist_ok=True)
    (results_path / "plots" / "png").mkdir(parents=True, exist_ok=True)
    (results_path / "data").mkdir(parents=True, exist_ok=True)

    lofar_data, _, _ = load_data()

    embeddings, all_targets = run_experiment(config, lofar_data, results_path, device)

    if config.model_name != "MLP":
        save_embeddings_and_targets(config, embeddings, all_targets, results_path)

    store_hash(model_hash)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the model.')
    parser.add_argument('--config', type=str, default='config', help='Path to the configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # lofar_data, _, _ = load_data()

    config_file = f"./configs/{args.config}.json"
    with open(config_file, 'r') as f:
        sweep_configuration = json.load(f)

    if args.debug:
        project_name = f'{args.config}-debug-v2'
    else:
        project_name = f'{args.config}-v2'
    sweep_configuration['name'] = f"{project_name}-sweep"

    sweep_id = wandb.sweep(sweep_configuration, project=project_name)

    wandb.agent(sweep_id, function=lambda : sweep_experiment(project_name))
