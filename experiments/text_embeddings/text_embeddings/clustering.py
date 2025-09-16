import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils import resample


def bootstrap_clustering_metrics(
    embeddings: np.array,
    labels: np.array,
    k: int=20,
    n_bootstrap: int=30,
    random_state: int | np.random.RandomState=202509,
):
    """
    Perform k-means clustering with bootstrap resampling to estimate variance
    for silhouette score and adjusted rand index.
    """
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    silhouette_scores = []
    rand_scores = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = resample(range(len(embeddings)), random_state=random_state)
        X_boot = embeddings[indices]
        y_boot = labels[indices]

        # Fit k-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X_boot)

        # Calculate metrics
        sil_score = silhouette_score(X_boot, cluster_labels)
        rand_score = adjusted_rand_score(y_boot, cluster_labels)

        silhouette_scores.append(sil_score)
        rand_scores.append(rand_score)

    # Calculate statistics
    results = {
        'silhouette': {
            'mean': np.mean(silhouette_scores),
            'std': np.std(silhouette_scores),
            'scores': silhouette_scores
        },
        'adjusted_rand_index': {
            'mean': np.mean(rand_scores),
            'std': np.std(rand_scores),
            'scores': rand_scores
        }
    }

    return results