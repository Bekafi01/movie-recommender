"""Unit tests for NegativeSampler and NCF dataset."""

import pandas as pd

from recsys.features.sampler import NCFDataset, NegativeSampler


def test_negative_sampler() -> None:
    """Test generating negative samples for positive interactions."""
    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "movie_id": [10, 20, 10],
            "rating": [5.0, 4.0, 4.5],
        }
    )

    user_to_idx = {1: 0, 2: 1}
    movie_to_idx = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4}

    sampler = NegativeSampler(num_negatives=2, positive_threshold=3.5, random_seed=42)
    users, items, labels = sampler.sample(ratings_df, user_to_idx, movie_to_idx)

    # 3 positives * (1 pos + 2 neg) = 9 samples
    assert len(users) == 9
    assert len(items) == 9
    assert len(labels) == 9
    assert sum(labels) == 3.0

    # Dataset tests
    dataset = NCFDataset(users, items, labels)
    assert len(dataset) == 9
    u, i, y = dataset[0]
    assert u.dtype.is_floating_point is False
    assert i.dtype.is_floating_point is False
    assert y.dtype.is_floating_point is True
