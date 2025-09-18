import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import argparse
from pathlib import Path
import glob
import csv
from datetime import datetime

def bootstrap_confidence_interval(scores, n_iterations=10000):
    """Calcula a média e o intervalo de confiança de 95% usando bootstrap."""
    bootstrap_means = []
    n_scores = len(scores)
    for _ in range(n_iterations):
        sample = np.random.choice(scores, size=n_scores, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    mean_score = np.mean(scores)
    
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    
    return mean_score, lower_bound, upper_bound

def plot_and_save_metrics(results_path: Path):
    """
    Calculates performance metrics, plots confusion matrices, and appends the
    final metrics to a central CSV file in percentage format 'mean% ± std%'.
    """
    pred_files = sorted(glob.glob(str(results_path / "data" / "predictions_fold_*.npy")))
    target_files = sorted(glob.glob(str(results_path / "data" / "targets_fold_*.npy")))

    if not pred_files or not target_files:
        print(f"Erro: Predict/target files not found in {results_path / 'data'}")
        return
    
    fold_accuracies = []
    macro_precisions, macro_recalls, macro_f1s = [], [], []
    weighted_precisions, weighted_recalls, weighted_f1s = [], [], []
    
    for fold_idx, (pred_file, target_file) in enumerate(zip(pred_files, target_files), 1):
        print(f"--- Processing Fold {fold_idx} ---")
        preds = np.load(pred_file)
        targets = np.load(target_file)
        
        fold_accuracies.append(accuracy_score(preds, targets))
        
        report = classification_report(targets, preds, output_dict=True, zero_division=0)
        
        macro_precisions.append(report['macro avg']['precision'])
        macro_recalls.append(report['macro avg']['recall'])
        macro_f1s.append(report['macro avg']['f1-score'])
        weighted_precisions.append(report['weighted avg']['precision'])
        weighted_recalls.append(report['weighted avg']['recall'])
        weighted_f1s.append(report['weighted avg']['f1-score'])
        
    if fold_accuracies:
        csv_path = results_path.parent.parent / "model_bootstrap.csv"

        # Calcula o IC para a acurácia
        acc_mean, acc_lower, acc_upper = bootstrap_confidence_interval(fold_accuracies)
        
        # Calcula o IC para o F1-Score ponderado
        f1_mean, f1_lower, f1_upper = bootstrap_confidence_interval(weighted_f1s)
        
        # Salva os scores brutos para o teste de Wilcoxon posterior
        f1_scores_str = ','.join([f'{score:.4f}' for score in weighted_f1s])
        
        results_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': results_path.parent.name,
            'hyperparameters': results_path.name,
            'accuracy_ci': f"{acc_mean:.2%} [{acc_lower:.2%}, {acc_upper:.2%}]",
            'weighted_f1_ci': f"{f1_mean:.2%} [{f1_lower:.2%}, {f1_upper:.2%}]",
            'f1_fold_scores': f1_scores_str 
        }

        fieldnames = list(results_dict.keys())
        file_exists = csv_path.is_file()

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)
        
        print(f"\n--- Resultados com Intervalo de Confiança Bootstrap salvos em: {csv_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates performance metrics and appends them to a summary CSV.')
    parser.add_argument('results_path', type=str, 
                        help='Path for results directory. e.g.: ./results/production/MLP/hidden_256_128')
    args = parser.parse_args()
    results_path = Path(args.results_path)
    if not results_path.is_dir():
        print(f"Error: Specified directory does not exists: {results_path}")
    else:
        plot_and_save_metrics(results_path)