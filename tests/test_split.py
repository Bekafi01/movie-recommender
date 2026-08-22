"""Unit tests for temporal evaluation splitting."""

import pandas as pd

from recsys.evaluation.split import create_evaluation_split


def test_create_evaluation_split() -> None:
    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 1, 2, 2],
            "movie_id": [10, 20, 30, 40, 50, 60, 10, 20],
            "rating": [4.0, 4.5, 3.0, 5.0, 4.0, 5.0, 5.0, 4.0],
            "timestamp": [100, 200, 300, 400, 500, 600, 100, 200],
        }
    )

    # User 1 has 6 interactions (>= min 5), leave_k = 2 -> history: [10, 20, 30, 40], test: [50, 60] (both >= 3.5)
    # User 2 has only 2 interactions (< min 5) -> filtered out
    dataset = create_evaluation_split(
        ratings_df=ratings_df,
        total_catalog_size=100,
        leave_k=2,
        min_interactions=5,
        positive_threshold=3.5,
    )

    assert dataset.eval_user_ids == [1]
    assert dataset.user_histories[1] == [10, 20, 30, 40]
    assert dataset.ground_truth[1] == [50, 60]
    assert dataset.total_catalog_size == 100
    assert 10 in dataset.item_popularity_prob
