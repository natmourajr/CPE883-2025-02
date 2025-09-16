import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

def find_optimal_k_kmeans(embeddings, k_range=None, max_k=10):
    """
    Find optimal K for K-means clustering using elbow method and silhouette analysis.

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        k_range: range of K values to test (default: 2 to max_k)
        max_k: maximum K to test (default: 10)

    Returns:
        dict: Contains optimal K, inertias, silhouette scores, and fitted model
    """
    if k_range is None:
        k_range = range(2, min(max_k + 1, embeddings.shape[0]))

    inertias = []
    silhouette_scores = []
    models = {}

    print(f"Testing K-means with K values: {list(k_range)}")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(sil_score)
        models[k] = kmeans

        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

    # Find optimal K using elbow method
    knee_locator = KneeLocator(list(k_range), inertias, curve="convex", direction="decreasing")
    optimal_k_elbow = knee_locator.elbow

    # Find optimal K using silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]

    print(f"\nOptimal K (Elbow method): {optimal_k_elbow}")
    print(f"Optimal K (Silhouette): {optimal_k_silhouette}")

    # Choose the K that appears in both methods or has highest silhouette
    if optimal_k_elbow == optimal_k_silhouette:
        optimal_k = optimal_k_elbow
        print(f"Both methods agree on K={optimal_k}")
    else:
        optimal_k = optimal_k_silhouette
        print(f"Using silhouette-based K={optimal_k}")

    return {
        'optimal_k': optimal_k,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_silhouette': optimal_k_silhouette,
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'best_model': models[optimal_k],
        'all_models': models
    }

def plot_k_selection_metrics(results):
    """
    Plot elbow curve and silhouette scores for K selection.

    Args:
        results: dict returned from find_optimal_k_kmeans
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Elbow curve
    ax1.plot(results['k_range'], results['inertias'], 'bo-')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)')
    ax1.set_title('Elbow Method for Optimal K')
    ax1.grid(True)
    if results['optimal_k_elbow']:
        ax1.axvline(x=results['optimal_k_elbow'], color='r', linestyle='--',
                   label=f'Optimal K = {results["optimal_k_elbow"]}')
        ax1.legend()

    # Silhouette scores
    ax2.plot(results['k_range'], results['silhouette_scores'], 'go-')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal K')
    ax2.grid(True)
    ax2.axvline(x=results['optimal_k_silhouette'], color='r', linestyle='--',
               label=f'Optimal K = {results["optimal_k_silhouette"]}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample embeddings (replace with your actual embeddings)
    np.random.seed(42)
    n_samples, n_features = 300, 50
    embeddings = np.random.randn(n_samples, n_features)

    print(f"Embeddings shape: {embeddings.shape}")

    # Find optimal K
    results = find_optimal_k_kmeans(embeddings, max_k=15)

    # Plot results
    plot_k_selection_metrics(results)

    # Get final clustering with optimal K
    optimal_model = results['best_model']
    cluster_labels = optimal_model.fit_predict(embeddings)

    print(f"\nFinal clustering with K={results['optimal_k']}:")
    print(f"Cluster centers shape: {optimal_model.cluster_centers_.shape}")
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Unique clusters: {np.unique(cluster_labels)}")
    print(f"Cluster distribution: {np.bincount(cluster_labels)}")