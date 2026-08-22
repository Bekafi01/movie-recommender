"""Offline ranking and retrieval metrics for Recommender Systems."""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_rate_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Hit Rate at K: 1.0 if at least one actual item is in top-K, else 0.0."""
    if not actual or k <= 0:
        return 0.0
    pred_k = set(predicted[:k])
    return 1.0 if any(item in pred_k for item in actual) else 0.0


def precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Precision at K: (# relevant items in top-K) / K."""
    if not actual or k <= 0:
        return 0.0
    pred_k = predicted[:k]
    if not pred_k:
        return 0.0
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / k)


def recall_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Recall at K: (# relevant items in top-K) / (# total relevant items)."""
    if not actual or k <= 0:
        return 0.0
    pred_k = set(predicted[:k])
    actual_set = set(actual)
    hits = sum(1 for item in pred_k if item in actual_set)
    return float(hits / len(actual_set))


def mrr_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Mean Reciprocal Rank at K: 1 / rank of first relevant item, or 0.0."""
    if not actual or k <= 0:
        return 0.0
    actual_set = set(actual)
    for rank, item in enumerate(predicted[:k], start=1):
        if item in actual_set:
            return 1.0 / float(rank)
    return 0.0


def average_precision_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Average Precision at K (AP@K) for a single user."""
    if not actual or k <= 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]
    score = 0.0
    hits = 0

    for i, item in enumerate(pred_k, start=1):
        if item in actual_set:
            hits += 1
            score += hits / float(i)

    num_relevant = min(len(actual_set), k)
    return float(score / num_relevant) if num_relevant > 0 else 0.0


def ndcg_at_k(actual: Sequence[int], predicted: Sequence[int], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance."""
    if not actual or k <= 0:
        return 0.0

    actual_set = set(actual)
    pred_k = predicted[:k]

    # Compute DCG@K
    dcg = 0.0
    for rank, item in enumerate(pred_k, start=1):
        if item in actual_set:
            # Binary relevance: rel_i = 1 -> (2^1 - 1) = 1
            dcg += 1.0 / math.log2(rank + 1)

    # Compute Ideal DCG@K (IDCG@K)
    num_relevant = min(len(actual_set), k)
    if num_relevant == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0
