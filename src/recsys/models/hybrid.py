"""Two-Stage Hybrid Recommender with Maximal Marginal Relevance (MMR) Diversity Re-Ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, load_config
from ..utils.logger import get_logger
from .base import BaseRecommender
from .collaborative import SVDCollaborativeRecommender
from .content_based import SemanticVectorRecommender
from .explainability import ExplainabilityEngine
from .popularity import PopularityRecommender

logger = get_logger("recsys.models.hybrid")


class HybridRecommender(BaseRecommender):
    """Hybrid Recommender combining Content, Collaborative Filtering, and MMR diversity re-ranking."""

    def __init__(
        self,
        content_model: SemanticVectorRecommender | None = None,
        collab_model: SVDCollaborativeRecommender | None = None,
        popularity_model: PopularityRecommender | None = None,
        config: AppConfig | None = None,
    ):
        self.config = config or load_config()
        self.hybrid_cfg = self.config.hybrid

        self.content_model = content_model or SemanticVectorRecommender(config=self.config)
        self.collab_model = collab_model or SVDCollaborativeRecommender(config=self.config)
        self.popularity_model = popularity_model or PopularityRecommender(config=self.config)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.explainability: ExplainabilityEngine | None = None
        self._fitted = False

    def fit(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        soup_df: pd.DataFrame,
        **kwargs: Any,
    ) -> HybridRecommender:
        """Fit all component sub-models."""
        self.movies_df = movies_df.copy()
        self.explainability = ExplainabilityEngine(movies_df=self.movies_df)

        self.popularity_model.fit(movies_df=self.movies_df)
        self.content_model.fit(movies_df=self.movies_df, soup_df=soup_df)
        self.collab_model.fit(ratings_df=ratings_df, movies_df=self.movies_df)

        self._fitted = True
        logger.info(
            "HybridRecommender fitted successfully with Content, Collab, and Popularity engines."
        )
        return self

    def recommend(
        self,
        query: Any = None,
        user_id: int | None = None,
        favorite_movie_ids: list[int] | None = None,
        top_k: int = 10,
        apply_mmr: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate hybrid recommendations for existing users, guest taste profiles, or single movie queries."""
        if not self._fitted:
            raise RuntimeError("HybridRecommender must be fitted before recommend().")

        # Case 1: Existing User with User ID
        if user_id is not None and user_id in self.collab_model.user_to_idx:
            return self._recommend_for_user(user_id=user_id, top_k=top_k, apply_mmr=apply_mmr)

        # Case 2: Guest User with Selected Favorite Movies (Taste Profile)
        if favorite_movie_ids and len(favorite_movie_ids) > 0:
            return self._recommend_from_favorites(
                favorite_movie_ids=favorite_movie_ids, top_k=top_k, apply_mmr=apply_mmr
            )

        # Case 3: Single Movie Query (Title or ID)
        if query is not None:
            return self.content_model.recommend(query=query, top_k=top_k)

        # Case 4: Completely Cold User -> Fallback to Popularity
        return self.popularity_model.recommend(top_k=top_k)

    def _recommend_for_user(self, user_id: int, top_k: int, apply_mmr: bool) -> pd.DataFrame:
        """Blend Collaborative Filtering predictions with Content similarity from user's favorite movies."""
        candidate_pool_size = self.hybrid_cfg.candidate_pool_size

        # 1. Fetch top candidates from collaborative model
        collab_candidates = self.collab_model.recommend(
            query=user_id, top_k=candidate_pool_size, exclude_rated=True
        )
        if collab_candidates.empty:
            return self.popularity_model.recommend(top_k=top_k)

        # Normalize collab scores to [0, 1]
        c_scores = collab_candidates["score"].values
        min_s, max_s = c_scores.min(), c_scores.max()
        norm_collab = (
            (c_scores - min_s) / (max_s - min_s + 1e-8) if max_s > min_s else np.ones_like(c_scores)
        )

        # 2. Blend with Popularity / Weighted Rating prior
        w_scores = (
            collab_candidates["weighted_rating"].values
            if "weighted_rating" in collab_candidates
            else norm_collab
        )
        min_w, max_w = w_scores.min(), w_scores.max()
        norm_weighted = (
            (w_scores - min_w) / (max_w - min_w + 1e-8) if max_w > min_w else np.ones_like(w_scores)
        )

        # Blend score
        alpha = self.hybrid_cfg.collaborative_weight
        beta = 1.0 - alpha
        final_scores = alpha * norm_collab + beta * norm_weighted

        candidate_df = collab_candidates.copy()
        candidate_df["hybrid_score"] = final_scores

        if apply_mmr and self.content_model.pipeline.embeddings is not None:
            reordered = self._apply_mmr_reranking(candidate_df=candidate_df, top_k=top_k)
            return reordered

        candidate_df = (
            candidate_df.sort_values(by="hybrid_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        candidate_df["rank"] = range(1, len(candidate_df) + 1)
        candidate_df["score"] = np.round(candidate_df["hybrid_score"], 4)
        return candidate_df

    def _recommend_from_favorites(
        self, favorite_movie_ids: list[int], top_k: int, apply_mmr: bool
    ) -> pd.DataFrame:
        """Aggregate semantic embeddings from multiple favorite movies to create a taste profile."""
        valid_ids = [m for m in favorite_movie_ids if m in self.content_model.id_to_idx]
        if not valid_ids or self.content_model.pipeline.embeddings is None:
            return self.popularity_model.recommend(top_k=top_k)

        # Average the embedding vectors of favorite movies
        indices = [self.content_model.id_to_idx[m] for m in valid_ids]
        fav_vectors = self.content_model.pipeline.embeddings[indices]
        taste_vector = np.mean(fav_vectors, axis=0, keepdims=True)
        # Normalize
        taste_vector = taste_vector / (np.linalg.norm(taste_vector) + 1e-8)

        # Search FAISS index
        distances, candidate_indices = self.content_model.pipeline.search_vector(
            taste_vector, top_k=self.hybrid_cfg.candidate_pool_size
        )

        seen_ids = set(valid_ids)
        filtered_indices: list[int] = []
        filtered_scores: list[float] = []

        for idx, dist in zip(candidate_indices, distances, strict=False):
            if idx >= 0 and idx < len(self.movies_df):
                m_id = self.movies_df.iloc[idx]["movie_id"]
                if m_id not in seen_ids:
                    filtered_indices.append(idx)
                    filtered_scores.append(float(dist))

        candidates = self.movies_df.iloc[filtered_indices].copy().reset_index(drop=True)
        candidates["hybrid_score"] = filtered_scores

        if apply_mmr and len(candidates) > top_k:
            return self._apply_mmr_reranking(candidate_df=candidates, top_k=top_k)

        candidates = candidates.head(top_k).reset_index(drop=True)
        candidates["rank"] = range(1, len(candidates) + 1)
        candidates["score"] = np.round(candidates["hybrid_score"], 4)
        return candidates

    def _apply_mmr_reranking(self, candidate_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """Maximal Marginal Relevance: balances relevance and intra-list diversity."""
        mmr_lambda = self.hybrid_cfg.mmr_lambda
        cand_indices = [
            self.content_model.id_to_idx[m_id]
            for m_id in candidate_df["movie_id"]
            if m_id in self.content_model.id_to_idx
        ]

        if not cand_indices or self.content_model.pipeline.embeddings is None:
            return candidate_df.head(top_k).reset_index(drop=True)

        cand_vectors = self.content_model.pipeline.embeddings[cand_indices]
        # Pairwise cosine similarity matrix among candidates
        sim_matrix = np.dot(cand_vectors, cand_vectors.T)

        relevance_scores = candidate_df["hybrid_score"].values[: len(cand_indices)]
        selected: list[int] = []
        unselected = list(range(len(cand_indices)))

        # 1. Pick highest scoring item
        first_pick = int(np.argmax(relevance_scores))
        selected.append(first_pick)
        unselected.remove(first_pick)

        # 2. Iteratively pick items maximizing MMR score
        while len(selected) < top_k and unselected:
            mmr_scores: list[float] = []
            for i in unselected:
                rel = relevance_scores[i]
                max_sim = max(sim_matrix[i, j] for j in selected)
                mmr = mmr_lambda * rel - (1.0 - mmr_lambda) * max_sim
                mmr_scores.append(mmr)

            best_idx = unselected[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            unselected.remove(best_idx)

        final_df = candidate_df.iloc[selected].copy().reset_index(drop=True)
        final_df["rank"] = range(1, len(final_df) + 1)
        final_df["score"] = np.round(final_df["hybrid_score"], 4)
        return final_df

    def explain_recommendation(
        self,
        source_movie_id: int,
        recommended_movie_id: int,
        similarity_score: float = 0.0,
    ) -> dict[str, Any]:
        """Delegate explanation generation to ExplainabilityEngine."""
        if self.explainability is None:
            self.explainability = ExplainabilityEngine(movies_df=self.movies_df)
        return self.explainability.explain(source_movie_id, recommended_movie_id, similarity_score)

    def save(self, path: Path | None = None) -> None:
        """Save sub-models to artifacts."""
        cfg = self.config
        self.popularity_model.save(cfg.paths.models_path / "popularity_model.pkl")
        self.content_model.save(cfg.paths.embeddings_path / "semantic_meta.pkl")
        self.collab_model.save(cfg.paths.models_path / "svd_model.pkl")
        logger.info("Saved all Hybrid sub-models to artifacts.")

    @classmethod
    def load(
        cls, path: Path | None = None, config: AppConfig | None = None, **kwargs: Any
    ) -> HybridRecommender:
        """Load all sub-models from artifacts."""
        cfg = config or load_config()
        instance = cls(config=cfg)

        pop_path = cfg.paths.models_path / "popularity_model.pkl"
        if pop_path.exists():
            instance.popularity_model = PopularityRecommender.load(pop_path, config=cfg)
            instance.movies_df = instance.popularity_model.movies_df

        sem_path = cfg.paths.embeddings_path / "semantic_meta.pkl"
        if sem_path.exists():
            instance.content_model = SemanticVectorRecommender.load(sem_path, config=cfg)
            if instance.movies_df.empty:
                instance.movies_df = instance.content_model.movies_df

        svd_path = cfg.paths.models_path / "svd_model.pkl"
        if svd_path.exists():
            instance.collab_model = SVDCollaborativeRecommender.load(svd_path, config=cfg)

        instance.explainability = ExplainabilityEngine(movies_df=instance.movies_df)
        instance._fitted = True
        logger.info("Loaded HybridRecommender from artifacts.")
        return instance
