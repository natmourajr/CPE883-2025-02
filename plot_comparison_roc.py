import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
# 1. Defina o diretório principal onde todos os resultados são salvos.
RESULTS_DIR = 'results'
N_SPLITS = 10 

# 2. Liste os nomes das pastas dos modelos e os nomes que você quer na legenda.
#    IMPORTANTE: O 'path' deve ser o nome da pasta exata dentro de 'results'.
#    O 'name' é como ele aparecerá no gráfico.
models_to_compare = {
    'ResNet-18_Baseline': {
        'name': 'ResNet-18',
        'color': 'blue'
    },
    'CKAN': {
        'name': 'CKAN',
        'color': 'green'
    },
    'CapsNet_Strided': {
        'name': 'CapsNet',
        'color': 'red'
    },
   
    'ViT_Baseline': {
        'name': 'ViT-B/16',
        'color': 'purple' 
    }
}

# ============================================================================
# LÓGICA DO SCRIPT
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for ax, (model_path, model_info) in zip(axes, models_to_compare.items()):
    model_dir = os.path.join(RESULTS_DIR, model_path)
    
    try:
        latest_experiment = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])[-1]
        experiment_path = os.path.join(model_dir, latest_experiment)
        
        print(f"Processando modelo '{model_info['name']}' do experimento: {latest_experiment}")
        
        # --- BLOCO 1: CALCULAR A INCERTEZA DA VALIDAÇÃO CRUZADA ---
        tprs_validation = []
        base_fpr = np.linspace(0, 1, 101)

        for i in range(1, N_SPLITS + 1):
            fold_dir = os.path.join(experiment_path, f"fold_{i}")
            npz_file_path = os.path.join(fold_dir, 'roc_vectors_Validação.npz')
            
            if os.path.exists(npz_file_path):
                data = np.load(npz_file_path)
                fpr, tpr = data['fpr'], data['tpr']
                interp_tpr = np.interp(base_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs_validation.append(interp_tpr)
        
        if not tprs_validation:
            raise FileNotFoundError("Nenhum arquivo de validação encontrado.")
            
        # Calcula apenas o desvio padrão das curvas de validação
        std_tprs_validation = np.std(tprs_validation, axis=0)

        # --- BLOCO 2: CARREGAR A CURVA DO HOLD-OUT ---
        holdout_npz_path = os.path.join(experiment_path, 'holdout_results', 'roc_vectors_Hold-Out.npz')
        if not os.path.exists(holdout_npz_path):
            raise FileNotFoundError("Arquivo ROC do Hold-Out não encontrado.")
            
        holdout_data = np.load(holdout_npz_path)
        fpr_holdout = holdout_data['fpr']
        tpr_holdout = holdout_data['tpr']
        auc_holdout = holdout_data['auc']

        # --- BLOCO 3: PLOTAR A CURVA DO HOLD-OUT COM A INCERTEZA DA VALIDAÇÃO ---
        
        # Interpola a curva do hold-out para a mesma base de FPR
        interp_tpr_holdout = np.interp(base_fpr, fpr_holdout, tpr_holdout)
        
        # Define os limites da faixa de erro usando o desvio padrão da validação
        tprs_upper = np.minimum(interp_tpr_holdout + std_tprs_validation, 1)
        tprs_lower = np.maximum(interp_tpr_holdout - std_tprs_validation, 0)
        
        # Plota a curva do Hold-Out como linha principal
        label = f'AUC Hold-Out = {auc_holdout:.3f}'
        ax.plot(base_fpr, interp_tpr_holdout, color=model_info['color'], label=label, lw=2)
        # Plota a faixa de incerteza da validação ao redor da curva do hold-out
        ax.fill_between(base_fpr, tprs_lower, tprs_upper, color=model_info['color'], alpha=0.2)
        
        # Configurações visuais do gráfico
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        ax.set_title(model_info['name'], fontsize=16, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True)

    except (FileNotFoundError, IndexError) as e:
        print(f"AVISO: Falha ao processar {model_info['name']}. Erro: {e}")
        ax.text(0.5, 0.5, 'Dados não encontrados', ha='center', va='center', fontsize=12)
        ax.set_title(model_info['name'], fontsize=16, fontweight='bold')

# Título e layout final
fig.suptitle('Curva ROC no Conjunto Hold-Out (com Incerteza da Validação Cruzada)', fontsize=20)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Salva e mostra a figura
output_filename = 'roc_holdout_com_incerteza_cv.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nGráfico final salvo em: {output_filename}")

plt.show()