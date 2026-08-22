"""Collaborative Filtering Recommender using SVD / Matrix Factorization."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from ..config import AppConfig, load_config
from ..utils.exceptions import UserNotFoundError
from ..utils.logger import get_logger
from ..utils.timer import timed
from .base import BaseRecommender

logger = get_logger("recsys.models.collaborative")


class SVDCollaborativeRecommender(BaseRecommender):
    """Model-based collaborative filtering using SVD Matrix Factorization."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.n_factors = self.config.collaborative.svd.n_factors
        self.random_state = self.config.collaborative.svd.random_state

        self.svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.user_ids: list[int] = []
        self.movie_ids: list[int] = []
        self.user_to_idx: dict[int, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.user_means: np.ndarray = np.array([])
        self.predicted_matrix: np.ndarray = np.array([])
        self.user_rated_items: dict[int, set[int]] = {}
        self.item_factors: np.ndarray = np.array([])
        self._fitted = False

    @timed("Fitting SVD Matrix Factorization")
    def fit(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, **kwargs: Any
    ) -> SVDCollaborativeRecommender:
        """Construct user-item matrix, perform SVD, and compute predicted rating matrix."""
        self.movies_df = movies_df.copy()

        # Build ID mappings
        self.user_ids = sorted(ratings_df["user_id"].unique())
        self.movie_ids = sorted(movies_df["movie_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}

        num_users = len(self.user_ids)
        num_movies = len(self.movie_ids)

        # Track observed interactions per user
        self.user_rated_items = ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        # Fill sparse matrix
        matrix = np.zeros((num_users, num_movies), dtype=np.float32)
        for _, row in ratings_df.iterrows():
            u = int(row["user_id"])
            m = int(row["movie_id"])
            if u in self.user_to_idx and m in self.movie_to_idx:
                matrix[self.user_to_idx[u], self.movie_to_idx[m]] = float(row["rating"])

        # Center ratings by user mean (ignoring zeros)
        user_ratings_count = (matrix > 0).sum(axis=1)
        user_ratings_sum = matrix.sum(axis=1)
        self.user_means = np.divide(
            user_ratings_sum,
            user_ratings_count,
            out=np.full(num_users, 3.0, dtype=np.float32),
            where=user_ratings_count > 0,
        )

        centered_matrix = matrix.copy()
        for i in range(num_users):
            mask = matrix[i] > 0
            centered_matrix[i, mask] -= self.user_means[i]

        # Fit Truncated SVD
        n_comp = min(self.n_factors, min(num_users, num_movies) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        u_factors = self.svd.fit_transform(centered_matrix)
        self.item_factors = self.svd.components_

        # Reconstruct predicted rating matrix: U * V^T + user_mean
        reconstructed = np.dot(u_factors, self.item_factors)
        self.predicted_matrix = reconstructed + self.user_means[:, np.newaxis]

        self._fitted = True
        logger.info(
            f"SVD fitted: {num_users} users, {num_movies} movies, "
            f"{n_comp} latent factors (explained variance ratio: {self.svd.explained_variance_ratio_.sum():.2%})."
        )
        return self

    def recommend(
        self,
        query: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        exclude_item_ids: set[int] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Recommend Top-K predicted movies for a given user_id."""
        if not self._fitted:
            raise RuntimeError("SVDCollaborativeRecommender must be fitted before recommend().")

        user_id = int(query)
        if user_id not in self.user_to_idx:
            raise UserNotFoundError(user_id)

        u_idx = self.user_to_idx[user_id]
        user_predictions = self.predicted_matrix[u_idx].copy()

        # Mask movies to exclude (either custom set or all rated)
        if exclude_item_ids is not None:
            for item_id in exclude_item_ids:
                if item_id in self.movie_to_idx:
                    user_predictions[self.movie_to_idx[item_id]] = -999.0
        elif exclude_rated and user_id in self.user_rated_items:
            for rated_movie_id in self.user_rated_items[user_id]:
                if rated_movie_id in self.movie_to_idx:
                    user_predictions[self.movie_to_idx[rated_movie_id]] = -999.0

        top_indices = np.argsort(user_predictions)[::-1][:top_k]
        scores = user_predictions[top_indices]
        recommended_movie_ids = [self.movie_ids[i] for i in top_indices]

        # Fetch metadata
        recs_df = (
            self.movies_df[self.movies_df["movie_id"].isin(recommended_movie_ids)]
            .set_index("movie_id")
            .loc[recommended_movie_ids]
            .reset_index()
        )
        recs_df["rank"] = range(1, len(recs_df) + 1)
        recs_df["score"] = np.round(scores, 3)

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
        available_cols = [c for c in output_cols if c in recs_df.columns]
        return recs_df[available_cols]

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        """Predict specific rating score for (user_id, movie_id)."""
        if not self._fitted:
            raise RuntimeError("SVDCollaborativeRecommender must be fitted.")
        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return float(self.user_means.mean()) if len(self.user_means) else 3.0

        u_idx = self.user_to_idx[user_id]
        m_idx = self.movie_to_idx[movie_id]
        return float(self.predicted_matrix[u_idx, m_idx])

    def save(self, path: Path) -> None:
        """Save SVD model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "movies_df": self.movies_df,
                    "user_ids": self.user_ids,
                    "movie_ids": self.movie_ids,
                    "user_to_idx": self.user_to_idx,
                    "movie_to_idx": self.movie_to_idx,
                    "user_means": self.user_means,
                    "predicted_matrix": self.predicted_matrix,
                    "user_rated_items": self.user_rated_items,
                    "item_factors": self.item_factors,
                    "_fitted": self._fitted,
                },
                f,
            )
        logger.info(f"Saved SVDCollaborativeRecommender to {path}")

    @classmethod
    def load(
        cls, path: Path, config: AppConfig | None = None, **kwargs: Any
    ) -> SVDCollaborativeRecommender:
        """Load SVD model from disk."""
        instance = cls(config=config)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.movies_df = data["movies_df"]
        instance.user_ids = data["user_ids"]
        instance.movie_ids = data["movie_ids"]
        instance.user_to_idx = data["user_to_idx"]
        instance.movie_to_idx = data["movie_to_idx"]
        instance.user_means = data["user_means"]
        instance.predicted_matrix = data["predicted_matrix"]
        instance.user_rated_items = data["user_rated_items"]
        instance.item_factors = data["item_factors"]
        instance._fitted = data["_fitted"]
        logger.info(f"Loaded SVDCollaborativeRecommender from {path}")
        return instance
