"""Unit and integration tests for the FastAPI recommendation microservice."""

from __future__ import annotations

from collections.abc import Generator

import faiss
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recsys.config import load_config
from recsys.models.collaborative import SVDCollaborativeRecommender
from recsys.models.content_based import SemanticVectorRecommender, TFIDFRecommender
from recsys.models.explainability import ExplainabilityEngine
from recsys.models.hybrid import HybridRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.serving.api import app, state


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create test client with lifespan context enabled, ensuring hermetic mock state for clean CI environments."""
    with TestClient(app) as test_client:
        if state.catalog_size == 0 or state.hybrid_model is None:
            # Build lightweight mock state for hermetic CI testing
            cfg = load_config()
            movies_df = pd.DataFrame(
                {
                    "movie_id": [1, 2, 296, 356, 5],
                    "tmdb_id": [101, 102, 103, 104, 105],
                    "title": [
                        "Inception",
                        "Interstellar",
                        "Pulp Fiction",
                        "Forrest Gump",
                        "The Matrix",
                    ],
                    "release_year": [2010, 2014, 1994, 1994, 1999],
                    "genres_list": [
                        ["action", "sci-fi"],
                        ["drama", "sci-fi"],
                        ["crime", "drama"],
                        ["drama", "romance"],
                        ["action", "sci-fi"],
                    ],
                    "genres_str": [
                        "action sci-fi",
                        "drama sci-fi",
                        "crime drama",
                        "drama romance",
                        "action sci-fi",
                    ],
                    "directors_list": [
                        ["christopher_nolan"],
                        ["christopher_nolan"],
                        ["quentin_tarantino"],
                        ["robert_zemeckis"],
                        ["wachowskis"],
                    ],
                    "director_str": [
                        "christopher_nolan",
                        "christopher_nolan",
                        "quentin_tarantino",
                        "robert_zemeckis",
                        "wachowskis",
                    ],
                    "cast_list": [
                        ["leonardo_dicaprio"],
                        ["matthew_mcconaughey"],
                        ["john_travolta"],
                        ["tom_hanks"],
                        ["keanu_reeves"],
                    ],
                    "cast_str": [
                        "leonardo_dicaprio",
                        "matthew_mcconaughey",
                        "john_travolta",
                        "tom_hanks",
                        "keanu_reeves",
                    ],
                    "keywords_list": [["dream"], ["space"], ["crime"], ["life"], ["matrix"]],
                    "keywords_str": ["dream", "space", "crime", "life", "matrix"],
                    "vote_average": [8.4, 8.6, 8.9, 8.8, 8.7],
                    "vote_count": [20000, 18000, 22000, 21000, 19000],
                    "popularity": [45.0, 42.0, 50.0, 48.0, 46.0],
                    "weighted_rating": [8.3, 8.5, 8.8, 8.7, 8.6],
                    "poster_path": [
                        "/path1.jpg",
                        "/path2.jpg",
                        "/path3.jpg",
                        "/path4.jpg",
                        "/path5.jpg",
                    ],
                }
            )
            ratings_df = pd.DataFrame(
                {
                    "user_id": [1, 1, 1, 2, 2],
                    "movie_id": [1, 2, 296, 356, 5],
                    "rating": [5.0, 4.5, 4.0, 5.0, 4.0],
                    "timestamp": [1000, 2000, 3000, 4000, 5000],
                }
            )

            soup_df = pd.DataFrame(
                {
                    "movie_id": [1, 2, 296, 356, 5],
                    "tmdb_id": [101, 102, 103, 104, 105],
                    "title": [
                        "Inception",
                        "Interstellar",
                        "Pulp Fiction",
                        "Forrest Gump",
                        "The Matrix",
                    ],
                    "soup": [
                        "inception dream christopher_nolan leonardo_dicaprio action sci-fi",
                        "interstellar space christopher_nolan matthew_mcconaughey drama sci-fi",
                        "pulp_fiction crime quentin_tarantino john_travolta crime drama",
                        "forrest_gump life robert_zemeckis tom_hanks drama romance",
                        "the_matrix matrix wachowskis keanu_reeves action sci-fi virtual reality",
                    ],
                }
            )

            state.catalog_size = len(movies_df)
            state.explainability = ExplainabilityEngine(movies_df=movies_df)
            state.popularity_model = PopularityRecommender(config=cfg).fit(movies_df)
            state.tfidf_model = TFIDFRecommender(config=cfg).fit(movies_df, soup_df)
            state.svd_model = SVDCollaborativeRecommender(config=cfg).fit(ratings_df)

            # Synthetic embeddings for fast hermetic test
            emb = np.random.randn(len(movies_df), 384).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            idx = faiss.IndexFlatIP(384)
            idx.add(emb)

            sem = SemanticVectorRecommender(config=cfg)
            sem.movies_df = movies_df
            sem.embeddings = emb
            sem.index = idx
            sem.movie_id_to_idx = {int(mid): i for i, mid in enumerate(movies_df["movie_id"])}
            sem.idx_to_movie_id = {i: int(mid) for i, mid in enumerate(movies_df["movie_id"])}
            sem._fitted = True
            state.semantic_model = sem

            hybrid = HybridRecommender(
                content_model=sem,
                collab_model=state.svd_model,
                popularity_model=state.popularity_model,
                config=cfg,
            )
            hybrid.movies_df = movies_df
            hybrid._fitted = True
            state.hybrid_model = hybrid

        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns healthy status and loaded models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["catalog_size"] > 0
    assert "hybrid" in data["models_loaded"]


def test_movie_search_and_lookup(client: TestClient) -> None:
    """Test movie substring search and single lookup."""
    # Search
    res_search = client.get("/api/v1/movies/search?q=Inception&limit=5")
    assert res_search.status_code == 200
    movies = res_search.json()
    assert len(movies) > 0
    assert any("Inception" in m["title"] for m in movies)

    movie_id = movies[0]["movie_id"]

    # Lookup by ID
    res_lookup = client.get(f"/api/v1/movies/{movie_id}")
    assert res_lookup.status_code == 200
    movie = res_lookup.json()
    assert movie["movie_id"] == movie_id
    assert "title" in movie

    # Non-existent ID
    res_missing = client.get("/api/v1/movies/99999999")
    assert res_missing.status_code == 404


def test_popular_recommendations(client: TestClient) -> None:
    """Test popular recommendation endpoint with optional genre filter."""
    # General
    res = client.get("/api/v1/recommend/popular?top_k=5")
    assert res.status_code == 200
    data = res.json()
    assert data["total_results"] == 5
    assert len(data["recommendations"]) == 5

    # Genre filtered
    res_genre = client.get("/api/v1/recommend/popular?genre=action&min_year=2000&top_k=3")
    assert res_genre.status_code == 200
    data_genre = res_genre.json()
    assert data_genre["total_results"] > 0


def test_content_based_recommendations(client: TestClient) -> None:
    """Test item-to-item and natural language semantic recommendations."""
    # Semantic FAISS
    res_sem = client.post(
        "/api/v1/recommend/movie",
        json={"query": "Inception", "top_k": 4, "algorithm": "semantic"},
    )
    assert res_sem.status_code == 200
    data_sem = res_sem.json()
    assert data_sem["algorithm"] == "content_semantic"
    assert len(data_sem["recommendations"]) == 4

    # TF-IDF
    res_tfidf = client.post(
        "/api/v1/recommend/movie",
        json={"query": "virtual reality sci-fi action", "top_k": 3, "algorithm": "tfidf"},
    )
    assert res_tfidf.status_code == 200
    data_tfidf = res_tfidf.json()
    assert data_tfidf["algorithm"] == "content_tfidf"
    assert len(data_tfidf["recommendations"]) > 0


def test_user_recommendations(client: TestClient) -> None:
    """Test personalized recommendations for registered users."""
    # Hybrid
    res_hybrid = client.post(
        "/api/v1/recommend/user",
        json={"user_id": 1, "top_k": 4, "algorithm": "hybrid", "apply_mmr": True},
    )
    assert res_hybrid.status_code == 200
    data_hybrid = res_hybrid.json()
    assert data_hybrid["algorithm"] == "hybrid"
    assert len(data_hybrid["recommendations"]) == 4

    # SVD
    res_svd = client.post(
        "/api/v1/recommend/user",
        json={"user_id": 1, "top_k": 4, "algorithm": "svd"},
    )
    assert res_svd.status_code == 200
    assert len(res_svd.json()["recommendations"]) == 4

    # Unknown user
    res_unknown = client.post(
        "/api/v1/recommend/user",
        json={"user_id": 999999, "top_k": 5, "algorithm": "svd"},
    )
    assert res_unknown.status_code == 404


def test_taste_profile_recommendations(client: TestClient) -> None:
    """Test guest cold-start recommendations from selected favorite movies."""
    res = client.post(
        "/api/v1/recommend/taste-profile",
        json={"favorite_movie_ids": [1, 296, 356], "top_k": 4, "apply_mmr": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["algorithm"] == "taste_profile_centroid_vector"
    assert len(data["recommendations"]) == 4


def test_explainability_endpoint(client: TestClient) -> None:
    """Test recommendation explanation endpoint."""
    res = client.get("/api/v1/explain?source_id=1&target_id=2&score=0.85")
    assert res.status_code == 200
    data = res.json()
    assert data["source_id"] == 1
    assert data["recommended_id"] == 2
    assert "summary" in data
    assert "match_percentage" in data
