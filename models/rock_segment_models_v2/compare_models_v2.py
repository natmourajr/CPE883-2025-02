#compare_models_v2.py
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# ===== Caminhos dos resultados =====
model_dirs = {
    "DeepONet": "Results/deeponet/results.txt",
    "CKAN": "Results/CKAN/results.txt",
    "CapsNet": "Results/CapsNet/results.txt",
    "CNN": "Results/cnn/results.txt",
}

# ===== Função para carregar os resultados =====
def load_results(filepath):
    if not os.path.exists(filepath):
        print(f"[AVISO] Não encontrei {filepath}")
        return None
    df = pd.read_csv(filepath, sep="|", engine="python")
    df.columns = [c.strip() for c in df.columns]  # remove espaços extras
    return df

# ===== Carregar dados =====
results = {}
for name, path in model_dirs.items():
    df = load_results(path)
    if df is not None:
        results[name] = df

# ===== Criar pasta de comparação =====
comparison_dir = "Results/comparisons"
os.makedirs(comparison_dir, exist_ok=True)

# ===== Função para plotar métricas =====
def plot_metric(metric, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for name, df in results.items():
        if metric in df.columns:
            plt.plot(df["epoch"], df[metric], label=name)
    plt.xlabel("Época")
    plt.ylabel(ylabel)
    plt.title(f"Comparação de {ylabel}")
    plt.legend()
    plt.grid(True)

    # Usar escala logarítmica apenas para loss
    if "loss" in metric.lower():
        plt.yscale('log')
        plt.title(f"Comparação de {ylabel} (Escala Log)")

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, filename))
    plt.close()

# ===== Gráficos comparativos =====
plot_metric("train_loss", "Train Loss", "train_loss.png")
plot_metric("val_loss", "Val Loss", "val_loss.png")
plot_metric("test_loss", "Test Loss", "test_loss.png")
plot_metric("val_Dice", "Dice (Validação)", "val_dice.png")
plot_metric("val_IoU", "IoU (Validação)", "val_iou.png")
plot_metric("test_Dice", "Dice (Teste)", "test_dice.png")
plot_metric("test_IoU", "IoU (Teste)", "test_iou.png")

# ===== Tabela resumo (última época) =====
summary = []
for name, df in results.items():
    last = df.iloc[-1]
    summary.append({
        "Modelo": name,
        "Train Loss": last["train_loss"],
        "Val Loss": last["val_loss"],
        "Test Loss": last["test_loss"],
        "Val Dice": last.get("val_Dice", None),
        "Val IoU": last.get("val_IoU", None),
        "Test Dice": last.get("test_Dice", None),
        "Test IoU": last.get("test_IoU", None),
    })

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(comparison_dir, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print(f"✅ Comparações salvas em {comparison_dir}")
