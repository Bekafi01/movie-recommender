"""Unit tests for PopularityRecommender."""

import pandas as pd

from recsys.models.popularity import PopularityRecommender


def test_popularity_recommender_ranking() -> None:
    """Test that popularity recommender sorts by weighted rating."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Movie Low", "Movie Mid", "Movie High"],
            "release_year": [2010, 2015, 2020],
            "genres_str": ["action", "comedy", "action drama"],
            "vote_average": [6.0, 7.5, 8.8],
            "vote_count": [100, 500, 5000],
            "weighted_rating": [6.2, 7.3, 8.6],
        }
    )

    model = PopularityRecommender()
    model.fit(movies_df)

    recs = model.recommend(top_k=2)
    assert len(recs) == 2
    assert recs.iloc[0]["title"] == "Movie High"
    assert recs.iloc[1]["title"] == "Movie Mid"


def test_popularity_recommender_filtering() -> None:
    """Test filtering by genre and release year."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Old Action", "New Action", "New Comedy"],
            "release_year": [1990, 2018, 2020],
            "genres_str": ["action", "action", "comedy"],
            "vote_average": [8.0, 8.5, 9.0],
            "vote_count": [1000, 2000, 3000],
            "weighted_rating": [7.8, 8.4, 8.9],
        }
    )

    model = PopularityRecommender()
    model.fit(movies_df)

    # Filter by action only + min_year >= 2000
    recs = model.recommend(genre="action", min_year=2000, top_k=5)
    assert len(recs) == 1
    assert recs.iloc[0]["title"] == "New Action"
