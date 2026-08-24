"""Content-Based Recommenders: Sparse TF-IDF and Dense FAISS Semantic Search."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, load_config
from ..features.embeddings import DenseEmbeddingPipeline
from ..features.text_vectorizer import TFIDFVectorizerWrapper
from ..utils.logger import get_logger
from .base import BaseRecommender

logger = get_logger("recsys.models.content_based")


class TFIDFRecommender(BaseRecommender):
    """Content-based recommender using sparse TF-IDF metadata vectors and cosine similarity."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.vectorizer_wrapper = TFIDFVectorizerWrapper(config=self.config)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.title_to_idx: dict[str, int] = {}
        self.id_to_idx: dict[int, int] = {}
        self._fitted = False

    def fit(
        self, movies_df: pd.DataFrame, soup_df: pd.DataFrame, **kwargs: Any
    ) -> TFIDFRecommender:
        """Fit TF-IDF matrix on metadata soup."""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.title_to_idx = {
            str(title).lower(): idx for idx, title in enumerate(self.movies_df["title"])
        }
        self.id_to_idx = {int(m_id): idx for idx, m_id in enumerate(self.movies_df["movie_id"])}

        # Build TF-IDF matrix
        self.vectorizer_wrapper.fit_transform(soup_df["soup"])
        self._fitted = True
        logger.info(f"TFIDFRecommender fitted on {len(self.movies_df)} movies.")
        return self

    def recommend(
        self,
        query: str | int,
        top_k: int = 10,
        exclude_self: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Recommend Top-K movies based on TF-IDF cosine similarity."""
        if not self._fitted:
            raise RuntimeError("TFIDFRecommender must be fitted before recommend().")

        movie_idx = self._resolve_query_index(query)

        # If query matches an existing movie in catalog
        if movie_idx is not None:
            query_vec = self.vectorizer_wrapper.tfidf_matrix[movie_idx]  # type: ignore[index]
            sim_scores = self.vectorizer_wrapper.compute_similarity(query_vec)[0]

            if exclude_self:
                sim_scores[movie_idx] = -1.0

            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            scores = sim_scores[top_indices]
        else:
            # Free-text search query
            query_vec = self.vectorizer_wrapper.transform([str(query)])
            sim_scores = self.vectorizer_wrapper.compute_similarity(query_vec)[0]
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            scores = sim_scores[top_indices]

        recs = self.movies_df.iloc[top_indices].copy().reset_index(drop=True)
        recs["rank"] = range(1, len(recs) + 1)
        recs["score"] = np.round(scores, 4)

        output_cols = [
            "rank",
            "movie_id",
            "tmdb_id",
            "title",
            "release_year",
            "genres_str",
            "vote_average",
            "score",
            "poster_path",
        ]
        available_cols = [c for c in output_cols if c in recs.columns]
        return recs[available_cols]

    def _resolve_query_index(self, query: str | int) -> int | None:
        if isinstance(query, int):
            return self.id_to_idx.get(query)
        q_str = str(query).strip().lower()
        if q_str in self.title_to_idx:
            return self.title_to_idx[q_str]
        return None

    def save(self, path: Path) -> None:
        """Persist model and metadata to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "movies_df": self.movies_df,
                    "title_to_idx": self.title_to_idx,
                    "id_to_idx": self.id_to_idx,
                    "_fitted": self._fitted,
                },
                f,
            )
        vec_path = path.parent / "tfidf_vectorizer.pkl"
        self.vectorizer_wrapper.save(vec_path)
        logger.info(f"Saved TFIDFRecommender to {path}")

    @classmethod
    def load(cls, path: Path, config: AppConfig | None = None, **kwargs: Any) -> TFIDFRecommender:
        """Load model state and vectorizer from disk."""
        instance = cls(config=config)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.movies_df = data["movies_df"]
        instance.title_to_idx = data["title_to_idx"]
        instance.id_to_idx = data["id_to_idx"]
        instance._fitted = data["_fitted"]

        vec_path = path.parent / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            instance.vectorizer_wrapper = TFIDFVectorizerWrapper.load(vec_path, config=config)
        logger.info(f"Loaded TFIDFRecommender from {path}")
        return instance


class SemanticVectorRecommender(BaseRecommender):
    """Dense Semantic Search Recommender using Sentence Transformers & FAISS vector index."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.pipeline = DenseEmbeddingPipeline(config=self.config)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.title_to_idx: dict[str, int] = {}
        self.id_to_idx: dict[int, int] = {}
        self._fitted = False

    def fit(
        self, movies_df: pd.DataFrame, soup_df: pd.DataFrame, **kwargs: Any
    ) -> SemanticVectorRecommender:
        """Generate dense embeddings and build FAISS index."""
        self.movies_df = movies_df.copy().reset_index(drop=True)
        self.movie_ids = self.movies_df["movie_id"].tolist()
        self.title_to_idx = {
            str(title).lower(): idx for idx, title in enumerate(self.movies_df["title"])
        }
        self.id_to_idx = {int(m_id): idx for idx, m_id in enumerate(self.movies_df["movie_id"])}

        self.pipeline.generate_embeddings(soup_df)
        self._fitted = True
        logger.info(f"SemanticVectorRecommender fitted with FAISS on {len(self.movies_df)} movies.")
        return self

    def recommend(
        self,
        query: str | int,
        top_k: int = 10,
        exclude_self: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Recommend Top-K movies using FAISS vector search."""
        # Bridge directly assigned index/embeddings if present
        if self.pipeline.index is None and hasattr(self, "index"):
            self.pipeline.index = self.index
        if self.pipeline.embeddings is None and hasattr(self, "embeddings"):
            self.pipeline.embeddings = self.embeddings

        if not self.title_to_idx and not self.movies_df.empty and "title" in self.movies_df:
            self.title_to_idx = {str(t).lower(): i for i, t in enumerate(self.movies_df["title"])}
        if not self.id_to_idx and not self.movies_df.empty and "movie_id" in self.movies_df:
            self.id_to_idx = {int(m): i for i, m in enumerate(self.movies_df["movie_id"])}

        if not self._fitted or self.pipeline.index is None:
            raise RuntimeError("SemanticVectorRecommender must be fitted before recommend().")

        movie_idx = self._resolve_query_index(query)

        if movie_idx is not None:
            # Query is a known movie in catalog
            query_vec = self.pipeline.embeddings[movie_idx : movie_idx + 1]  # type: ignore[index]
            fetch_k = top_k + 1 if exclude_self else top_k
            distances, indices = self.pipeline.search_vector(query_vec, top_k=fetch_k)

            res_indices: list[int] = []
            res_scores: list[float] = []

            for idx, dist in zip(indices, distances, strict=False):
                if exclude_self and idx == movie_idx:
                    continue
                if idx >= 0:
                    res_indices.append(idx)
                    res_scores.append(float(dist))
                if len(res_indices) >= top_k:
                    break
        else:
            # Query is a natural language text query (e.g., "mind-bending sci-fi about black holes")
            query_vec = self.pipeline.encode_text(str(query))
            distances, indices = self.pipeline.search_vector(query_vec, top_k=top_k)
            res_indices = [idx for idx in indices if idx >= 0]
            res_scores = [
                float(dist) for idx, dist in zip(indices, distances, strict=False) if idx >= 0
            ]

        recs = self.movies_df.iloc[res_indices].copy().reset_index(drop=True)
        recs["rank"] = range(1, len(recs) + 1)
        recs["score"] = np.round(res_scores, 4)

        output_cols = [
            "rank",
            "movie_id",
            "tmdb_id",
            "title",
            "release_year",
            "genres_str",
            "vote_average",
            "score",
            "poster_path",
        ]
        available_cols = [c for c in output_cols if c in recs.columns]
        return recs[available_cols]

    def _resolve_query_index(self, query: str | int) -> int | None:
        if isinstance(query, int):
            return self.id_to_idx.get(query)
        q_str = str(query).strip().lower()
        if q_str in self.title_to_idx:
            return self.title_to_idx[q_str]
        return None

    def save(self, path: Path | None = None) -> None:
        """Save FAISS index and metadata state."""
        target_dir = path.parent if path else self.config.paths.embeddings_path
        target_dir.mkdir(parents=True, exist_ok=True)

        meta_path = target_dir / "semantic_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "movies_df": self.movies_df,
                    "title_to_idx": self.title_to_idx,
                    "id_to_idx": self.id_to_idx,
                    "_fitted": self._fitted,
                },
                f,
            )
        self.pipeline.save(target_dir)
        logger.info(f"Saved SemanticVectorRecommender to {target_dir}")

    @classmethod
    def load(
        cls, path: Path | None = None, config: AppConfig | None = None, **kwargs: Any
    ) -> SemanticVectorRecommender:
        """Load FAISS index, embeddings, and metadata."""
        cfg = config or load_config()
        target_dir = path.parent if path else cfg.paths.embeddings_path

        instance = cls(config=cfg)
        meta_path = target_dir / "semantic_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
            instance.movies_df = data["movies_df"]
            instance.movie_ids = (
                instance.movies_df["movie_id"].tolist() if "movie_id" in instance.movies_df else []
            )
            instance.title_to_idx = data["title_to_idx"]
            instance.id_to_idx = data["id_to_idx"]
            instance._fitted = data["_fitted"]

        instance.pipeline = DenseEmbeddingPipeline.load(embeddings_dir=target_dir, config=cfg)
        logger.info(f"Loaded SemanticVectorRecommender from {target_dir}")
        return instance
