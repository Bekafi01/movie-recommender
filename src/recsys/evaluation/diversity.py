"""Beyond-accuracy evaluation metrics: Catalog Coverage, Novelty, and Intra-List Diversity."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def catalog_coverage(
    all_recommended_items: Sequence[Sequence[int]], total_catalog_items: int
) -> float:
    """Compute proportion of total catalog items recommended at least once across all users."""
    if total_catalog_items <= 0 or not all_recommended_items:
        return 0.0
    unique_recommended: set[int] = set()
    for user_recs in all_recommended_items:
        unique_recommended.update(user_recs)
    return float(len(unique_recommended) / total_catalog_items)


def novelty_at_k(
    all_recommended_items: Sequence[Sequence[int]],
    item_popularity_prob: dict[int, float],
    k: int = 10,
) -> float:
    """Compute average Novelty (Self-Information) across all recommendations: mean(-log2(p(i)))."""
    if not all_recommended_items or k <= 0:
        return 0.0

    self_info_scores: list[float] = []
    for user_recs in all_recommended_items:
        for item in user_recs[:k]:
            prob = item_popularity_prob.get(item, 1e-12)
            prob = max(prob, 1e-12)
            self_info_scores.append(-math.log2(prob))

    return float(np.mean(self_info_scores)) if self_info_scores else 0.0


def intra_list_diversity(
    recommended_items: Sequence[int],
    embeddings: np.ndarray,
    id_to_idx: dict[int, int],
) -> float:
    """Compute average pairwise cosine distance (1 - cosine_similarity) between items in a recommendation list."""
    valid_indices = [id_to_idx[m] for m in recommended_items if m in id_to_idx]
    if len(valid_indices) < 2:
        return 0.0

    vectors = embeddings[valid_indices]
    # Cosine similarity matrix (assumes vectors are L2-normalized)
    sim_matrix = np.dot(vectors, vectors.T)

    n = len(valid_indices)
    distances: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_dist = max(0.0, 1.0 - float(sim_matrix[i, j]))
            distances.append(cos_dist)

    return float(np.mean(distances)) if distances else 0.0


def mean_intra_list_diversity(
    all_recommended_items: Sequence[Sequence[int]],
    embeddings: np.ndarray,
    id_to_idx: dict[int, int],
    k: int = 10,
) -> float:
    """Compute Mean Intra-List Diversity across all users."""
    scores = [
        intra_list_diversity(recs[:k], embeddings, id_to_idx)
        for recs in all_recommended_items
        if len(recs) >= 2
    ]
    return float(np.mean(scores)) if scores else 0.0
