"""Unit tests for beyond-accuracy metrics: coverage, novelty, and intra-list diversity."""

import numpy as np

from recsys.evaluation.diversity import (
    catalog_coverage,
    intra_list_diversity,
    mean_intra_list_diversity,
    novelty_at_k,
)


def test_catalog_coverage() -> None:
    recs = [[1, 2, 3], [2, 3, 4], [1, 5]]
    # Unique = {1, 2, 3, 4, 5} = 5 items out of 10 total
    assert catalog_coverage(recs, total_catalog_items=10) == 0.5


def test_novelty_at_k() -> None:
    recs = [[1, 2]]
    # Item 1 prob = 0.5 -> -log2(0.5) = 1.0
    # Item 2 prob = 0.25 -> -log2(0.25) = 2.0
    # Mean novelty = 1.5
    item_probs = {1: 0.5, 2: 0.25}
    score = novelty_at_k(recs, item_popularity_prob=item_probs, k=2)
    assert abs(score - 1.5) < 1e-6


def test_intra_list_diversity() -> None:
    # 2 orthogonal vectors: cosine_sim = 0 -> cosine_dist = 1.0
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    id_to_idx = {10: 0, 20: 1}

    ild = intra_list_diversity([10, 20], embeddings, id_to_idx)
    assert abs(ild - 1.0) < 1e-6

    # Test identical vectors: cosine_sim = 1 -> cosine_dist = 0.0
    same_embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    same_ild = intra_list_diversity([10, 20], same_embeddings, id_to_idx)
    assert abs(same_ild - 0.0) < 1e-6

    # Mean ILD
    mean_ild = mean_intra_list_diversity([[10, 20]], embeddings, id_to_idx, k=2)
    assert abs(mean_ild - 1.0) < 1e-6
