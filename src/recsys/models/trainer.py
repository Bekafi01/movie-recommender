"""Model training coordinator that trains and persists all recommendation models."""

from __future__ import annotations

from typing import Any

from ..config import AppConfig, load_config
from ..data.db import DataRepository
from ..utils.logger import get_logger
from ..utils.timer import timed
from .collaborative import SVDCollaborativeRecommender
from .content_based import SemanticVectorRecommender, TFIDFRecommender
from .neural_cf import NeuralCollaborativeRecommender
from .popularity import PopularityRecommender

logger = get_logger("recsys.models.trainer")


@timed("Training All Recommendation Models")
def train_all_models(
    config: AppConfig | None = None, train_neural_cf: bool = True
) -> dict[str, Any]:
    """Load clean processed data and train all multi-paradigm recommendation models."""
    cfg = config or load_config()
    repo = DataRepository(config=cfg)

    logger.info("Loading clean processed datasets for model training...")
    movies_df = repo.load_movies()
    ratings_df = repo.load_ratings()
    soup_df = repo.load_metadata_soup()

    models_dir = cfg.paths.models_path
    embeddings_dir = cfg.paths.embeddings_path
    models_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}

    # 1. Train Popularity Model
    logger.info("--- [1/5] Training Popularity Recommender ---")
    pop_model = PopularityRecommender(config=cfg)
    pop_model.fit(movies_df=movies_df)
    pop_model.save(models_dir / "popularity_model.pkl")
    results["popularity"] = str(models_dir / "popularity_model.pkl")

    # 2. Train TF-IDF Model
    logger.info("--- [2/5] Training TF-IDF Content Recommender ---")
    tfidf_model = TFIDFRecommender(config=cfg)
    tfidf_model.fit(movies_df=movies_df, soup_df=soup_df)
    tfidf_model.save(models_dir / "tfidf_model.pkl")
    results["tfidf"] = str(models_dir / "tfidf_model.pkl")

    # 3. Train Dense Semantic FAISS Vector Index
    logger.info("--- [3/5] Generating Dense Embeddings & FAISS Index ---")
    semantic_model = SemanticVectorRecommender(config=cfg)
    semantic_model.fit(movies_df=movies_df, soup_df=soup_df)
    semantic_model.save(embeddings_dir / "semantic_meta.pkl")
    results["semantic_faiss"] = str(embeddings_dir / "faiss_index.bin")

    # 4. Train SVD Matrix Factorization
    logger.info("--- [4/5] Training SVD Matrix Factorization ---")
    svd_model = SVDCollaborativeRecommender(config=cfg)
    svd_model.fit(ratings_df=ratings_df, movies_df=movies_df)
    svd_model.save(models_dir / "svd_model.pkl")
    results["svd"] = str(models_dir / "svd_model.pkl")

    # 5. Train Neural Collaborative Filtering (NeuMF)
    if train_neural_cf:
        logger.info("--- [5/5] Training PyTorch Neural Collaborative Filtering (NeuMF) ---")
        ncf_model = NeuralCollaborativeRecommender(config=cfg)
        ncf_model.fit(ratings_df=ratings_df, movies_df=movies_df)
        ncf_model.save(models_dir / "neumf_model.pt")
        results["neural_cf"] = str(models_dir / "neumf_model.pt")

    logger.info("All recommendation models successfully trained and persisted to artifacts!")
    return {
        "status": "success",
        "trained_models": results,
    }


if __name__ == "__main__":
    train_all_models()
