import pandas as pd
from scipy.stats import wilcoxon
import ast 

def compare_models(csv_path, model1_index, model2_index):
    """
    Carrega os resultados do CSV e realiza o Teste de Wilcoxon
    entre dois modelos especificados pelos seus índices na tabela.
    """
    # Carrega os resultados
    df = pd.read_csv(csv_path)

    # Seleciona as linhas dos modelos que você quer comparar
    model1 = df.iloc[model1_index]
    model2 = df.iloc[model2_index]

    print("--- Comparando Modelos ---")
    print(f"Modelo A: {model1['model']} ({model1['hyperparameters']})")
    print(f"Modelo B: {model2['model']} ({model2['hyperparameters']})\n")

    # Extrai os scores dos folds, que estão salvos como string
    # e converte de volta para uma lista de números
    scores1 = [float(s) for s in model1['f1_fold_scores'].split(',')]
    scores2 = [float(s) for s in model2['f1_fold_scores'].split(',')]

    # Verifica se o número de folds é o mesmo
    if len(scores1) != len(scores2):
        raise ValueError("Os modelos não foram avaliados no mesmo número de folds.")

    # --- Realiza o Teste de Wilcoxon ---
    # A função retorna a estatística do teste e o p-valor
    stat, p_value = wilcoxon(scores1, scores2)

    print(f"Estatística do Teste de Wilcoxon: {stat:.4f}")
    print(f"P-valor: {p_value:.4f}")

    # --- Interpretação do Resultado ---
    alpha = 0.05  # Nível de significância
    if p_value < alpha:
        print("\nConclusão: A diferença entre os modelos é ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05).")
        # Para saber qual é melhor, comparamos a média ou mediana dos scores
        if np.mean(scores1) > np.mean(scores2):
            print("O Modelo A teve um desempenho significativamente superior ao Modelo B.")
        else:
            print("O Modelo B teve um desempenho significativamente superior ao Modelo A.")
    else:
        print("\nConclusão: NÃO há evidência de uma diferença estatisticamente significativa entre os modelos (p >= 0.05).")
        print("Pode-se considerar que os modelos tiveram um desempenho equivalente.")

if __name__ == '__main__':
    
    csv_file = './results/production/model_bootstrap.csv' 
    compare_models(csv_path=csv_file, model1_index=0, model2_index=1)