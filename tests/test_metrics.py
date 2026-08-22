"""Unit tests for offline ranking and evaluation metrics."""

import math

from recsys.evaluation.metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_hit_rate_at_k() -> None:
    actual = [10, 20]
    pred_hit = [5, 10, 15]
    pred_miss = [1, 2, 3]

    assert hit_rate_at_k(actual, pred_hit, k=2) == 1.0
    assert hit_rate_at_k(actual, pred_miss, k=3) == 0.0
    assert hit_rate_at_k([], pred_hit, k=3) == 0.0


def test_precision_and_recall_at_k() -> None:
    actual = [10, 20, 30, 40]
    predicted = [10, 5, 20, 99, 100]

    # At k=3: predicted[:3] = [10, 5, 20] -> hits = {10, 20} (2 hits)
    assert precision_at_k(actual, predicted, k=3) == 2.0 / 3.0
    assert recall_at_k(actual, predicted, k=3) == 2.0 / 4.0

    # At k=1: predicted[:1] = [10] -> hits = 1
    assert precision_at_k(actual, predicted, k=1) == 1.0
    assert recall_at_k(actual, predicted, k=1) == 1.0 / 4.0


def test_mrr_at_k() -> None:
    actual = [20, 30]
    pred1 = [20, 5, 10]  # Hit at rank 1 -> MRR = 1/1 = 1.0
    pred2 = [5, 20, 10]  # Hit at rank 2 -> MRR = 1/2 = 0.5
    pred3 = [5, 6, 20]  # Hit at rank 3 -> MRR = 1/3
    pred4 = [1, 2, 3]  # No hits -> MRR = 0.0

    assert mrr_at_k(actual, pred1, k=3) == 1.0
    assert mrr_at_k(actual, pred2, k=3) == 0.5
    assert abs(mrr_at_k(actual, pred3, k=3) - (1.0 / 3.0)) < 1e-6
    assert mrr_at_k(actual, pred4, k=3) == 0.0


def test_average_precision_at_k() -> None:
    actual = [10, 20]
    # Hits at rank 1 and 3: P@1 = 1/1, P@3 = 2/3. AP = (1 + 2/3) / 2
    predicted = [10, 5, 20, 99]
    expected_ap = (1.0 + (2.0 / 3.0)) / 2.0
    assert abs(average_precision_at_k(actual, predicted, k=3) - expected_ap) < 1e-6


def test_ndcg_at_k() -> None:
    actual = [10, 20]
    # Perfect ranking: hits at rank 1 and 2 -> NDCG = 1.0
    perfect_pred = [10, 20, 30]
    assert abs(ndcg_at_k(actual, perfect_pred, k=3) - 1.0) < 1e-6

    # Hit at rank 2 only: DCG = 1 / log2(3), IDCG = 1/log2(2) + 1/log2(3)
    pred_second = [99, 10, 88]
    dcg = 1.0 / math.log2(3)
    idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = dcg / idcg
    assert abs(ndcg_at_k(actual, pred_second, k=3) - expected_ndcg) < 1e-6
