"""Unit tests for TFIDFRecommender and SemanticVectorRecommender."""

from pathlib import Path

import pandas as pd

from recsys.models.content_based import SemanticVectorRecommender, TFIDFRecommender


def test_tfidf_recommender(tmp_path: Path) -> None:
    """Test fitting, recommending, saving, and loading TFIDFRecommender."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Inception", "The Matrix", "The Notebook"],
            "release_year": [2010, 1999, 2004],
            "genres_str": ["action sci-fi", "action sci-fi", "romance drama"],
        }
    )

    soup_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Inception", "The Matrix", "The Notebook"],
            "soup": [
                "dream virtual reality sci-fi action christopher_nolan leonardo_dicaprio",
                "virtual reality simulation sci-fi action wachowskis keanu_reeves",
                "love romance drama nicholas_sparks ryan_gosling",
            ],
        }
    )

    recommender = TFIDFRecommender()
    recommender.fit(movies_df, soup_df)

    # Inception should recommend The Matrix as #1 match
    recs = recommender.recommend("Inception", top_k=2)
    assert len(recs) == 2
    assert recs.iloc[0]["title"] == "The Matrix"

    # Free-text search query
    text_recs = recommender.recommend("virtual reality action", top_k=2)
    assert len(text_recs) == 2
    assert text_recs.iloc[0]["title"] in ["Inception", "The Matrix"]

    # Test save & load
    save_path = tmp_path / "tfidf_model.pkl"
    recommender.save(save_path)
    loaded = TFIDFRecommender.load(save_path)
    loaded_recs = loaded.recommend("Inception", top_k=2)
    assert loaded_recs.iloc[0]["title"] == "The Matrix"


def test_semantic_vector_recommender(tmp_path: Path) -> None:
    """Test fitting, vector search, saving, and loading SemanticVectorRecommender."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Inception", "Interstellar", "Finding Nemo"],
            "release_year": [2010, 2014, 2003],
            "genres_str": ["action sci-fi", "drama sci-fi", "animation family"],
        }
    )

    soup_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3],
            "tmdb_id": [101, 102, 103],
            "title": ["Inception", "Interstellar", "Finding Nemo"],
            "soup": [
                "dreams subconscious mind space heist christopher_nolan",
                "space wormhole black hole gravity exploration christopher_nolan",
                "ocean fish clownfish clown animated family pixar",
            ],
        }
    )

    recommender = SemanticVectorRecommender()
    recommender.fit(movies_df, soup_df)

    # Inception should recommend Interstellar
    recs = recommender.recommend("Inception", top_k=2)
    assert len(recs) == 2
    assert recs.iloc[0]["title"] == "Interstellar"

    # Free-text semantic search
    text_recs = recommender.recommend("deep space voyage through wormholes", top_k=1)
    assert len(text_recs) == 1
    assert text_recs.iloc[0]["title"] == "Interstellar"

    # Test save & load
    save_path = tmp_path / "semantic_meta.pkl"
    recommender.save(save_path)
    loaded = SemanticVectorRecommender.load(save_path)
    loaded_recs = loaded.recommend("Inception", top_k=2)
    assert loaded_recs.iloc[0]["title"] == "Interstellar"
