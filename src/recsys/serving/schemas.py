"""Pydantic schemas for the FastAPI recommendation microservice."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MovieSummary(BaseModel):
    """Schema representing a recommended or catalog movie item."""

    rank: int | None = None
    movie_id: int
    tmdb_id: int | None = None
    title: str
    release_year: int | None = None
    genres_str: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    weighted_rating: float | None = None
    poster_path: str | None = None
    score: float | None = None


class RecommendationResponse(BaseModel):
    """Standard response envelope for recommendation endpoints."""

    query: Any
    algorithm: str
    total_results: int
    latency_ms: float
    recommendations: list[MovieSummary]


class MovieRecommendationRequest(BaseModel):
    """Request payload for movie-to-movie or natural language recommendations."""

    query: str = Field(
        ...,
        description="Movie title substring or natural language thematic query",
        examples=["Inception"],
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    algorithm: Literal["semantic", "tfidf"] = Field(
        "semantic", description="Content-based retrieval algorithm"
    )


class UserRecommendationRequest(BaseModel):
    """Request payload for personalized user recommendations."""

    user_id: int = Field(..., ge=1, description="Registered User ID", examples=[1])
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    algorithm: Literal["hybrid", "svd", "neural_cf"] = Field(
        "hybrid", description="Recommendation algorithm"
    )
    apply_mmr: bool = Field(True, description="Apply Maximal Marginal Relevance for diversity")


class TasteProfileRequest(BaseModel):
    """Request payload for cold-start guest user recommendations from selected favorite movies."""

    favorite_movie_ids: list[int] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of movie IDs selected by the user",
        examples=[[1, 296, 356]],
    )
    top_k: int = Field(10, ge=1, le=50, description="Number of recommendations to return")
    apply_mmr: bool = Field(True, description="Apply Maximal Marginal Relevance for diversity")


class ExplainabilityResponse(BaseModel):
    """Response payload explaining recommendation rationale."""

    source_id: int
    source_title: str
    recommended_id: int
    recommended_title: str
    similarity_score: float
    match_percentage: str
    summary: str
    shared_directors: list[str]
    shared_cast: list[str]
    shared_genres: list[str]
    shared_keywords: list[str]


class HealthResponse(BaseModel):
    """Microservice health check and metadata status."""

    status: str
    version: str
    catalog_size: int
    models_loaded: list[str]
