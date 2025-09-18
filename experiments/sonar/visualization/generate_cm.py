import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import argparse
from pathlib import Path
import glob

def plot_confusion_matrix(results_path: Path):
    """
    Calculates and plots confusion matrices, and also calculates mean test 
    accuracy, precision, recall, and f1-score with std deviation.
    """
    pred_files = sorted(glob.glob(str(results_path / "data" / "predictions_fold_*.npy")))
    target_files = sorted(glob.glob(str(results_path / "data" / "targets_fold_*.npy")))

    if not pred_files or not target_files:
        print(f"Erro: Predict/target files not found in {results_path / 'data'}")
        return
    
    if len(pred_files) != len(target_files):
        print(f"Erro: Predict/target files inconsistent in {results_path / 'data'}")
        return

    normalized_cms = []
    fold_accuracies = []
    
    macro_precisions, macro_recalls, macro_f1s = [], [], []
    weighted_precisions, weighted_recalls, weighted_f1s = [], [], []
    
    class_labels = [0, 1, 2, 3]
    class_names = ['Class A', 'Class B', 'Class C', 'Class D']

    for fold_idx, (pred_file, target_file) in enumerate(zip(pred_files, target_files), 1):
        print(f"--- Processing Fold {fold_idx} ---")
        preds = np.load(pred_file)
        targets = np.load(target_file)
        
        acc = accuracy_score(targets, preds)
        fold_accuracies.append(acc)
        
        report = classification_report(targets, preds, labels=class_labels, 
                                       target_names=class_names, output_dict=True, zero_division=0)
        
        macro_precisions.append(report['macro avg']['precision'])
        macro_recalls.append(report['macro avg']['recall'])
        macro_f1s.append(report['macro avg']['f1-score'])

        weighted_precisions.append(report['weighted avg']['precision'])
        weighted_recalls.append(report['weighted avg']['recall'])
        weighted_f1s.append(report['weighted avg']['f1-score'])
        
        cm = confusion_matrix(targets, preds, labels=class_labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
        normalized_cms.append(cm_normalized)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt=".1%", cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})
        plt.title(f'Confusion Matrix - Fold {fold_idx}\nModel: {results_path.parent.name} - HPs: {results_path.name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        fold_save_path = results_path / f"confusion_matrix_fold_{fold_idx}.png"
        plt.savefig(fold_save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("\n====================================================")
    print("      Resumo das Métricas de Desempenho       ")
    print("====================================================")

    if fold_accuracies:
        print(f"\nAcurácia Média:       {np.mean(fold_accuracies):.2%} ± {np.std(fold_accuracies):.2%}")
        
        print("\n--- Média Ponderada (Weighted) ---")
        print(f"Precisão Ponderada:   {np.mean(weighted_precisions):.2%} ± {np.std(weighted_precisions):.2%}")
        print(f"Recall Ponderado:     {np.mean(weighted_recalls):.2%} ± {np.std(weighted_recalls):.2%}")
        print(f"F1-Score Ponderado:   {np.mean(weighted_f1s):.2%} ± {np.std(weighted_f1s):.2%}")

        print("\n--- Média Macro ---")
        print(f"Precisão Macro:       {np.mean(macro_precisions):.2%} ± {np.std(macro_precisions):.2%}")
        print(f"Recall Macro:         {np.mean(macro_recalls):.2%} ± {np.std(macro_recalls):.2%}")
        print(f"F1-Score Macro:       {np.mean(macro_f1s):.2%} ± {np.std(macro_f1s):.2%}")
        print("====================================================\n")
    
    print("--- Calculating Mean and Std Dev of Confusion Matrix ---")
    mean_cm = np.mean(normalized_cms, axis=0)
    std_cm = np.std(normalized_cms, axis=0)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates performance metrics (accuracy, precision, recall, f1) and confusion matrices for a trained model.')
    parser.add_argument('results_path', type=str, 
                        help='Path for results directory of a specific model. e.g.: ./results/production/MLP/hidden_256_128_dropout_0.2_lr_0.001')
    args = parser.parse_args()
    results_path = Path(args.results_path)
    if not results_path.is_dir():
        print(f"Error: Specified directory does not exists: {results_path}")
    else:
        plot_confusion_matrix(results_path)