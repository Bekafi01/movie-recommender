"""High-Performance FastAPI Recommendation Microservice."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import AppConfig, load_config
from ..data.db import DataRepository
from ..models.collaborative import SVDCollaborativeRecommender
from ..models.content_based import SemanticVectorRecommender, TFIDFRecommender
from ..models.explainability import ExplainabilityEngine
from ..models.hybrid import HybridRecommender
from ..models.neural_cf import NeuralCollaborativeRecommender
from ..models.popularity import PopularityRecommender
from ..utils.exceptions import MovieNotFoundError, RecSysError, UserNotFoundError
from ..utils.logger import get_logger
from .schemas import (
    ExplainabilityResponse,
    HealthResponse,
    MovieRecommendationRequest,
    MovieSummary,
    RecommendationResponse,
    TasteProfileRequest,
    UserRecommendationRequest,
)

logger = get_logger("recsys.serving.api")


class AppState:
    """Singleton container holding loaded models and shared state."""

    config: AppConfig
    repo: DataRepository
    popularity_model: PopularityRecommender | None = None
    tfidf_model: TFIDFRecommender | None = None
    semantic_model: SemanticVectorRecommender | None = None
    svd_model: SVDCollaborativeRecommender | None = None
    neural_cf_model: NeuralCollaborativeRecommender | None = None
    hybrid_model: HybridRecommender | None = None
    explainability: ExplainabilityEngine | None = None
    catalog_size: int = 0


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: preload models into memory on server startup."""
    logger.info("Initializing RecSys Microservice and preloading model artifacts...")
    cfg = load_config()
    state.config = cfg
    state.repo = DataRepository(config=cfg)

    # 1. Load movies catalog from database
    try:
        movies_df = state.repo.load_movies()
        state.catalog_size = len(movies_df)
        state.explainability = ExplainabilityEngine(movies_df=movies_df)
    except Exception as e:
        logger.warning(f"Could not load movies table: {e}")
        movies_df = None

    # 2. Load Models
    models_dir = cfg.paths.models_path
    embed_dir = cfg.paths.embeddings_path

    # Popularity
    pop_path = models_dir / "popularity_model.pkl"
    if pop_path.exists():
        state.popularity_model = PopularityRecommender.load(pop_path, config=cfg)

    # TF-IDF
    tfidf_path = models_dir / "tfidf_model.pkl"
    if tfidf_path.exists():
        state.tfidf_model = TFIDFRecommender.load(tfidf_path, config=cfg)

    # Semantic FAISS
    sem_path = embed_dir / "semantic_meta.pkl"
    if sem_path.exists():
        state.semantic_model = SemanticVectorRecommender.load(sem_path, config=cfg)

    # SVD
    svd_path = models_dir / "svd_model.pkl"
    if svd_path.exists():
        state.svd_model = SVDCollaborativeRecommender.load(svd_path, config=cfg)

    # Neural CF
    ncf_path = models_dir / "neumf_model.pt"
    if ncf_path.exists():
        state.neural_cf_model = NeuralCollaborativeRecommender.load(ncf_path, config=cfg)

    # Hybrid
    state.hybrid_model = HybridRecommender(
        content_model=state.semantic_model,
        collab_model=state.svd_model,
        popularity_model=state.popularity_model,
        config=cfg,
    )
    if movies_df is not None:
        state.hybrid_model.movies_df = movies_df
        state.hybrid_model._fitted = True

    logger.info("All model artifacts preloaded successfully! Server is ready to handle traffic.")
    yield
    logger.info("Shutting down RecSys Microservice...")


app = FastAPI(
    title="Movie Recommender System API",
    description="Production-grade asynchronous inference service for multi-paradigm recommendation engines.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Middleware adding server execution latency header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Exception Handlers
@app.exception_handler(MovieNotFoundError)
async def handle_movie_not_found(request: Request, exc: MovieNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"error": "MovieNotFoundError", "message": str(exc)}
    )


@app.exception_handler(UserNotFoundError)
async def handle_user_not_found(request: Request, exc: UserNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404, content={"error": "UserNotFoundError", "message": str(exc)}
    )


@app.exception_handler(RecSysError)
async def handle_recsys_error(request: Request, exc: RecSysError) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"error": exc.__class__.__name__, "message": str(exc)}
    )


# --- Health & Metadata ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint returning active models and catalog size."""
    loaded_models: list[str] = []
    if state.popularity_model is not None:
        loaded_models.append("popularity")
    if state.tfidf_model is not None:
        loaded_models.append("tfidf")
    if state.semantic_model is not None:
        loaded_models.append("semantic_faiss")
    if state.svd_model is not None:
        loaded_models.append("svd")
    if state.neural_cf_model is not None:
        loaded_models.append("neural_cf")
    if state.hybrid_model is not None:
        loaded_models.append("hybrid")

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        catalog_size=state.catalog_size,
        models_loaded=loaded_models,
    )


# --- Catalog Search ---
@app.get("/api/v1/movies/search", response_model=list[MovieSummary], tags=["Catalog"])
async def search_movies(
    q: str = Query(..., min_length=1, description="Movie title substring to search"),
    limit: int = Query(10, ge=1, le=50),
) -> list[MovieSummary]:
    """Search movies in the catalog by title substring."""
    results = state.repo.search_movies(q, limit=limit)
    return [MovieSummary(**r) for r in results]


@app.get("/api/v1/movies/{movie_id}", response_model=MovieSummary, tags=["Catalog"])
async def get_movie_by_id(movie_id: int) -> MovieSummary:
    """Retrieve full movie metadata by movie_id."""
    movie = state.repo.get_movie_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie with ID {movie_id} not found.")
    return MovieSummary(**movie)


# --- Recommendation Endpoints ---
@app.get(
    "/api/v1/recommend/popular", response_model=RecommendationResponse, tags=["Recommendations"]
)
async def recommend_popular(
    genre: str | None = Query(None, description="Optional genre filter (e.g. Action, Comedy)"),
    min_year: int | None = Query(None, ge=1900, le=2030, description="Minimum release year"),
    top_k: int = Query(10, ge=1, le=50),
) -> RecommendationResponse:
    """Demographic / Popularity recommendations using IMDb Weighted Rating."""
    if state.popularity_model is None:
        raise HTTPException(status_code=503, detail="Popularity model is not loaded.")

    start = time.perf_counter()
    recs_df = state.popularity_model.recommend(genre=genre, min_year=min_year, top_k=top_k)
    latency = (time.perf_counter() - start) * 1000.0

    items = [MovieSummary(**row) for row in recs_df.to_dict(orient="records")]
    return RecommendationResponse(
        query={"genre": genre, "min_year": min_year},
        algorithm="popularity_weighted_rating",
        total_results=len(items),
        latency_ms=round(latency, 2),
        recommendations=items,
    )


@app.post(
    "/api/v1/recommend/movie", response_model=RecommendationResponse, tags=["Recommendations"]
)
async def recommend_similar_movies(req: MovieRecommendationRequest) -> RecommendationResponse:
    """Content-Based recommendations given a movie title or natural language query."""
    start = time.perf_counter()

    if req.algorithm == "semantic":
        if state.semantic_model is None:
            raise HTTPException(status_code=503, detail="Semantic FAISS model is not loaded.")
        recs_df = state.semantic_model.recommend(query=req.query, top_k=req.top_k)
    else:
        if state.tfidf_model is None:
            raise HTTPException(status_code=503, detail="TF-IDF model is not loaded.")
        recs_df = state.tfidf_model.recommend(query=req.query, top_k=req.top_k)

    latency = (time.perf_counter() - start) * 1000.0
    items = [MovieSummary(**row) for row in recs_df.to_dict(orient="records")]

    return RecommendationResponse(
        query=req.query,
        algorithm=f"content_{req.algorithm}",
        total_results=len(items),
        latency_ms=round(latency, 2),
        recommendations=items,
    )


@app.post("/api/v1/recommend/user", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_for_user(req: UserRecommendationRequest) -> RecommendationResponse:
    """Personalized recommendations for a registered user ID."""
    start = time.perf_counter()

    if req.algorithm == "hybrid":
        if state.hybrid_model is None:
            raise HTTPException(status_code=503, detail="Hybrid model is not loaded.")
        recs_df = state.hybrid_model.recommend(
            user_id=req.user_id, top_k=req.top_k, apply_mmr=req.apply_mmr
        )
    elif req.algorithm == "svd":
        if state.svd_model is None:
            raise HTTPException(status_code=503, detail="SVD model is not loaded.")
        recs_df = state.svd_model.recommend(query=req.user_id, top_k=req.top_k, exclude_rated=True)
    elif req.algorithm == "neural_cf":
        if state.neural_cf_model is None:
            raise HTTPException(status_code=503, detail="Neural CF model is not loaded.")
        recs_df = state.neural_cf_model.recommend(
            query=req.user_id, top_k=req.top_k, exclude_rated=True
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm '{req.algorithm}'.")

    latency = (time.perf_counter() - start) * 1000.0
    items = [MovieSummary(**row) for row in recs_df.to_dict(orient="records")]

    return RecommendationResponse(
        query={"user_id": req.user_id},
        algorithm=req.algorithm,
        total_results=len(items),
        latency_ms=round(latency, 2),
        recommendations=items,
    )


@app.post(
    "/api/v1/recommend/taste-profile",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
)
async def recommend_from_taste_profile(req: TasteProfileRequest) -> RecommendationResponse:
    """Cold-start recommendations by centroid vector aggregation of selected favorite movie IDs."""
    if state.hybrid_model is None:
        raise HTTPException(status_code=503, detail="Hybrid model is not loaded.")

    start = time.perf_counter()
    recs_df = state.hybrid_model.recommend(
        favorite_movie_ids=req.favorite_movie_ids,
        top_k=req.top_k,
        apply_mmr=req.apply_mmr,
    )
    latency = (time.perf_counter() - start) * 1000.0
    items = [MovieSummary(**row) for row in recs_df.to_dict(orient="records")]

    return RecommendationResponse(
        query={"favorite_movie_ids": req.favorite_movie_ids},
        algorithm="taste_profile_centroid_vector",
        total_results=len(items),
        latency_ms=round(latency, 2),
        recommendations=items,
    )


@app.get("/api/v1/explain", response_model=ExplainabilityResponse, tags=["Explainability"])
async def explain_recommendation(
    source_id: int = Query(..., description="Source movie ID liked by user"),
    target_id: int = Query(..., description="Recommended target movie ID"),
    score: float = Query(0.0, ge=0.0, le=1.0, description="Similarity score"),
) -> ExplainabilityResponse:
    """Generate human-readable explanation for why a movie was recommended."""
    if state.explainability is None:
        raise HTTPException(status_code=503, detail="Explainability engine is not loaded.")

    explanation = state.explainability.explain(
        source_movie_id=source_id,
        recommended_movie_id=target_id,
        similarity_score=score,
    )
    return ExplainabilityResponse(**explanation)
