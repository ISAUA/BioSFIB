"""Label-free neighborhood preservation used for the retained Y7_T analysis."""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def calculate_spamv_knn_jaccard(learned_embedding, omics_features, k=50):
    """Per-spot kNN Jaccard used by SpaMV, without external labels."""
    if isinstance(learned_embedding, torch.Tensor):
        learned_embedding = learned_embedding.detach().cpu().numpy()
    if isinstance(omics_features, torch.Tensor):
        omics_features = omics_features.detach().cpu().numpy()

    learned_embedding = np.asarray(learned_embedding, dtype=np.float32)
    omics_features = np.asarray(omics_features, dtype=np.float32)
    if learned_embedding.ndim != 2 or omics_features.ndim != 2:
        raise ValueError("Jaccard inputs must both be two-dimensional matrices.")
    if learned_embedding.shape[0] != omics_features.shape[0]:
        raise ValueError(
            "Jaccard inputs must contain the same spots: "
            f"{learned_embedding.shape[0]} != {omics_features.shape[0]}"
        )
    if learned_embedding.shape[0] < 2:
        raise ValueError("Jaccard evaluation requires at least two spots.")
    if not np.isfinite(learned_embedding).all() or not np.isfinite(omics_features).all():
        raise ValueError("Jaccard inputs contain NaN or infinite values.")

    requested_k = int(k)
    if requested_k < 1:
        raise ValueError("Jaccard k must be a positive integer.")
    effective_k = min(requested_k, learned_embedding.shape[0] - 1)

    def neighbor_sets(matrix):
        model = NearestNeighbors(
            n_neighbors=effective_k + 1,
            metric="euclidean",
            algorithm="auto",
        )
        indices = model.fit(matrix).kneighbors(matrix, return_distance=False)
        result = []
        for spot_index, row in enumerate(indices):
            without_self = row[row != spot_index][:effective_k]
            if without_self.shape[0] != effective_k:
                raise RuntimeError("Failed to construct a self-excluded kNN set.")
            result.append(set(int(value) for value in without_self))
        return result

    embedding_neighbors = neighbor_sets(learned_embedding)
    omics_neighbors = neighbor_sets(omics_features)
    per_spot = np.asarray([
        len(embedding_set.intersection(omics_set))
        / len(embedding_set.union(omics_set))
        for embedding_set, omics_set in zip(embedding_neighbors, omics_neighbors)
    ], dtype=np.float32)
    summary = {
        "k_requested": requested_k,
        "k_effective": int(effective_k),
        "mean": float(per_spot.mean()),
        "median": float(np.median(per_spot)),
        "std": float(per_spot.std()),
        "q25": float(np.quantile(per_spot, 0.25)),
        "q75": float(np.quantile(per_spot, 0.75)),
        "n_spots": int(per_spot.shape[0]),
    }
    return per_spot, summary
