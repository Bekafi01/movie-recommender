"""Demographic and Popularity-based Recommender using the IMDb Weighted Rating formula."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, load_config
from ..data.cleaner import calculate_imdb_weighted_rating
from ..utils.logger import get_logger
from .base import BaseRecommender

logger = get_logger("recsys.models.popularity")


class PopularityRecommender(BaseRecommender):
    """Recommender that ranks movies based on the IMDb Weighted Rating formula."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self._fitted = False

    def fit(self, movies_df: pd.DataFrame, **kwargs: Any) -> PopularityRecommender:
        """Fit popularity model by storing movies DataFrame and ensuring weighted_rating exists."""
        self.movies_df = movies_df.copy()

        if "weighted_rating" not in self.movies_df.columns:
            percentile = self.config.popularity.min_vote_percentile
            self.movies_df["weighted_rating"] = calculate_imdb_weighted_rating(
                self.movies_df, percentile=percentile
            )

        self._fitted = True
        logger.info(f"PopularityRecommender fitted with {len(self.movies_df)} movies.")
        return self

    def recommend(
        self,
        query: Any = None,
        top_k: int = 10,
        genre: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        min_votes: int | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return Top-K globally popular or genre/year-filtered movies."""
        if not self._fitted:
            raise RuntimeError("PopularityRecommender must be fitted before calling recommend().")

        df = self.movies_df.copy()

        # 1. Filter by Genre
        if genre and genre.strip():
            g_clean = genre.strip().lower()
            df = df[df["genres_str"].str.contains(g_clean, case=False, na=False)]

        # 2. Filter by Year range
        if min_year is not None:
            df = df[df["release_year"].fillna(0) >= min_year]
        if max_year is not None:
            df = df[df["release_year"].fillna(9999) <= max_year]

        # 3. Filter by min votes
        if min_votes is not None:
            df = df[df["vote_count"] >= min_votes]

        # 4. Sort by Weighted Rating
        df = df.sort_values(by=["weighted_rating", "vote_count"], ascending=[False, False])
        top_recs = df.head(top_k).copy()

        # Build output format
        top_recs["rank"] = range(1, len(top_recs) + 1)
        top_recs["score"] = top_recs["weighted_rating"]

        output_cols = [
            "rank",
            "movie_id",
            "tmdb_id",
            "title",
            "release_year",
            "genres_str",
            "vote_average",
            "vote_count",
            "score",
            "poster_path",
        ]
        available_cols = [c for c in output_cols if c in top_recs.columns]
        return top_recs[available_cols].reset_index(drop=True)

    def save(self, path: Path) -> None:
        """Persist model state to a pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"movies_df": self.movies_df, "_fitted": self._fitted}, f)
        logger.info(f"Saved PopularityRecommender to {path}")

    @classmethod
    def load(
        cls, path: Path, config: AppConfig | None = None, **kwargs: Any
    ) -> PopularityRecommender:
        """Load model state from a pickle file."""
        instance = cls(config=config)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.movies_df = data["movies_df"]
        instance._fitted = data["_fitted"]
        logger.info(f"Loaded PopularityRecommender from {path}")
        return instance
