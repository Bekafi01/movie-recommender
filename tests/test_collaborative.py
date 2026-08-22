"""Unit tests for SVD Collaborative Filtering Recommender."""

from pathlib import Path

import pandas as pd
import pytest

from recsys.models.collaborative import SVDCollaborativeRecommender
from recsys.utils.exceptions import UserNotFoundError


def test_svd_collaborative_recommendations(tmp_path: Path) -> None:
    """Test SVD matrix factorization predictions and ranking."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "tmdb_id": [101, 102, 103, 104],
            "title": ["Inception", "Interstellar", "Dark Knight", "Prestige"],
            "release_year": [2010, 2014, 2008, 2006],
            "genres_str": ["sci-fi", "sci-fi", "action", "drama"],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "movie_id": [1, 2, 1, 3, 2, 4, 3, 4],
            "rating": [5.0, 4.5, 4.0, 5.0, 4.5, 4.0, 5.0, 4.5],
            "timestamp": [100, 200, 300, 400, 500, 600, 700, 800],
        }
    )

    model = SVDCollaborativeRecommender()
    model.fit(ratings_df=ratings_df, movies_df=movies_df)

    # User 1 rated movie 1 and 2. Should recommend movie 3 or 4
    recs = model.recommend(query=1, top_k=2, exclude_rated=True)
    assert len(recs) == 2
    assert 1 not in recs["movie_id"].values
    assert 2 not in recs["movie_id"].values

    # Test unknown user
    with pytest.raises(UserNotFoundError):
        model.recommend(query=999)

    # Test save and load
    save_path = tmp_path / "svd_model.pkl"
    model.save(save_path)
    loaded = SVDCollaborativeRecommender.load(save_path)
    loaded_recs = loaded.recommend(query=1, top_k=2)
    assert len(loaded_recs) == 2
