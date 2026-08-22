"""Abstract Base Class for all recommendation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class establishing the contract for all recommender models."""

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> BaseRecommender:
        """Train the model on input data."""

    @abstractmethod
    def recommend(
        self,
        query: Any,
        top_k: int = 10,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate top-K recommendations.

        Returns a DataFrame with at least:
        ['rank', 'movie_id', 'tmdb_id', 'title', 'score']
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model artifacts to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path, **kwargs: Any) -> BaseRecommender:
        """Load model artifacts from disk."""
