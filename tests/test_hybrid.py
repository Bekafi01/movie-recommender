"""Unit tests for Hybrid Recommender, MMR re-ranking, and ExplainabilityEngine."""

from pathlib import Path

import pandas as pd

from recsys.models.explainability import ExplainabilityEngine
from recsys.models.hybrid import HybridRecommender


def test_explainability_engine() -> None:
    """Test feature overlap reasoning generation."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "title": ["Inception", "Interstellar"],
            "directors_list": [["christopher_nolan"], ["christopher_nolan"]],
            "cast_list": [
                ["leonardo_dicaprio", "michael_caine"],
                ["matthew_mcconaughey", "michael_caine"],
            ],
            "genres_list": [["action", "sci-fi"], ["drama", "sci-fi"]],
            "keywords_list": [["space", "dream"], ["space", "wormhole"]],
        }
    )

    engine = ExplainabilityEngine(movies_df=movies_df)
    explanation = engine.explain(source_movie_id=1, recommended_movie_id=2, similarity_score=0.92)

    assert explanation["source_title"] == "Inception"
    assert explanation["recommended_title"] == "Interstellar"
    assert "Christopher Nolan" in explanation["shared_directors"]
    assert "Michael Caine" in explanation["shared_cast"]
    assert "Sci-Fi" in explanation["shared_genres"]
    assert "Space" in explanation["shared_keywords"]
    assert "Christopher Nolan" in explanation["summary"]


def test_hybrid_full_workflow(tmp_path: Path) -> None:
    """Test full HybridRecommender fit, user recommendation, and taste profile."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "tmdb_id": [101, 102, 103, 104],
            "title": ["Inception", "Interstellar", "Dark Knight", "Prestige"],
            "genres_str": ["sci-fi", "sci-fi", "action", "drama"],
            "genres_list": [["sci-fi"], ["sci-fi"], ["action"], ["drama"]],
            "directors_list": [
                ["christopher_nolan"],
                ["christopher_nolan"],
                ["christopher_nolan"],
                ["christopher_nolan"],
            ],
            "cast_list": [
                ["leonardo_dicaprio"],
                ["matthew_mcconaughey"],
                ["christian_bale"],
                ["christian_bale"],
            ],
            "keywords_list": [["dream"], ["space"], ["gotham"], ["magic"]],
            "release_year": [2010, 2014, 2008, 2006],
            "vote_average": [8.4, 8.6, 9.0, 8.5],
            "vote_count": [20000, 18000, 25000, 15000],
            "weighted_rating": [8.3, 8.5, 8.9, 8.4],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "movie_id": [1, 2, 1, 3],
            "rating": [5.0, 4.5, 4.0, 5.0],
            "timestamp": [100, 200, 300, 400],
        }
    )

    soup_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "tmdb_id": [101, 102, 103, 104],
            "title": ["Inception", "Interstellar", "Dark Knight", "Prestige"],
            "soup": [
                "dreams sci-fi christopher_nolan leonardo_dicaprio",
                "space wormhole sci-fi christopher_nolan matthew_mcconaughey",
                "batman gotham action christopher_nolan christian_bale",
                "magician illusion drama christopher_nolan christian_bale hugh_jackman",
            ],
        }
    )

    hybrid = HybridRecommender()
    hybrid.fit(movies_df=movies_df, ratings_df=ratings_df, soup_df=soup_df)

    # 1. Recommend for existing user 1
    user_recs = hybrid.recommend(user_id=1, top_k=2, apply_mmr=True)
    assert len(user_recs) == 2

    # 2. Recommend from favorite movie IDs (Taste Profile)
    taste_recs = hybrid.recommend(favorite_movie_ids=[1], top_k=2)
    assert len(taste_recs) == 2

    # 3. Recommend from single movie query
    single_recs = hybrid.recommend(query="Inception", top_k=2)
    assert len(single_recs) == 2

    # 4. Explain recommendation
    expl = hybrid.explain_recommendation(1, 2, 0.9)
    assert "Christopher Nolan" in expl["summary"]
