"""Unit and integration tests for the FastAPI recommendation microservice."""

import pytest
from fastapi.testclient import TestClient

from recsys.serving.api import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create test client with lifespan context enabled."""
    with TestClient(app) as test_client:
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
    assert data_genre["total_results"] == 3


def test_content_based_recommendations(client: TestClient) -> None:
    """Test item-to-item and natural language semantic recommendations."""
    # Semantic FAISS
    res_sem = client.post(
        "/api/v1/recommend/movie",
        json={"query": "Inception", "top_k": 5, "algorithm": "semantic"},
    )
    assert res_sem.status_code == 200
    data_sem = res_sem.json()
    assert data_sem["algorithm"] == "content_semantic"
    assert len(data_sem["recommendations"]) == 5

    # TF-IDF
    res_tfidf = client.post(
        "/api/v1/recommend/movie",
        json={"query": "virtual reality sci-fi action", "top_k": 3, "algorithm": "tfidf"},
    )
    assert res_tfidf.status_code == 200
    data_tfidf = res_tfidf.json()
    assert data_tfidf["algorithm"] == "content_tfidf"
    assert len(data_tfidf["recommendations"]) == 3


def test_user_recommendations(client: TestClient) -> None:
    """Test personalized recommendations for registered users."""
    # Hybrid
    res_hybrid = client.post(
        "/api/v1/recommend/user",
        json={"user_id": 1, "top_k": 5, "algorithm": "hybrid", "apply_mmr": True},
    )
    assert res_hybrid.status_code == 200
    data_hybrid = res_hybrid.json()
    assert data_hybrid["algorithm"] == "hybrid"
    assert len(data_hybrid["recommendations"]) == 5

    # SVD
    res_svd = client.post(
        "/api/v1/recommend/user",
        json={"user_id": 1, "top_k": 5, "algorithm": "svd"},
    )
    assert res_svd.status_code == 200
    assert len(res_svd.json()["recommendations"]) == 5

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
        json={"favorite_movie_ids": [1, 296, 356], "top_k": 5, "apply_mmr": True},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["algorithm"] == "taste_profile_centroid_vector"
    assert len(data["recommendations"]) == 5


def test_explainability_endpoint(client: TestClient) -> None:
    """Test recommendation explanation endpoint."""
    res = client.get("/api/v1/explain?source_id=1&target_id=2&score=0.85")
    assert res.status_code == 200
    data = res.json()
    assert data["source_id"] == 1
    assert data["recommended_id"] == 2
    assert "summary" in data
    assert "match_percentage" in data
