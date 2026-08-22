"""Sparse TF-IDF text feature extraction and cosine similarity matrix."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import AppConfig, load_config
from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.features.tfidf")


class TFIDFVectorizerWrapper:
    """Wrapper around scikit-learn TfidfVectorizer for metadata soup feature extraction."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        tfidf_cfg = self.config.content_based.tfidf

        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.max_features,
            ngram_range=tuple(tfidf_cfg.ngram_range),
            stop_words=tfidf_cfg.stop_words,
            sublinear_tf=tfidf_cfg.sublinear_tf,
        )
        self.tfidf_matrix: csr_matrix | None = None
        self._fitted = False

    @timed("Fitting TF-IDF Vectorizer")
    def fit_transform(self, soup_series: pd.Series) -> csr_matrix:
        """Fit vectorizer on metadata soup strings and transform to sparse matrix."""
        cleaned_soup = soup_series.fillna("").astype(str)
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_soup)
        self._fitted = True
        logger.info(
            f"TF-IDF matrix built with shape {self.tfidf_matrix.shape} "
            f"({self.tfidf_matrix.nnz:,} non-zero elements)."
        )
        return self.tfidf_matrix

    def transform(self, texts: list[str] | pd.Series) -> csr_matrix:
        """Transform new text queries using the fitted vectorizer."""
        if not self._fitted:
            raise RuntimeError("TF-IDF Vectorizer is not fitted yet.")
        if isinstance(texts, list):
            texts = pd.Series(texts)
        return self.vectorizer.transform(texts.fillna("").astype(str))

    def compute_similarity(
        self, query_matrix: csr_matrix, target_matrix: csr_matrix | None = None
    ) -> np.ndarray:
        """Compute cosine similarity between query and target matrices."""
        targets = target_matrix if target_matrix is not None else self.tfidf_matrix
        if targets is None:
            raise RuntimeError("No target TF-IDF matrix available for similarity computation.")
        return cosine_similarity(query_matrix, targets)

    def save(self, path: Path) -> None:
        """Persist vectorizer and matrix to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "_fitted": self._fitted,
                },
                f,
            )
        logger.info(f"Saved TFIDFVectorizerWrapper to {path}")

    @classmethod
    def load(cls, path: Path, config: AppConfig | None = None) -> TFIDFVectorizerWrapper:
        """Load vectorizer and matrix from disk."""
        instance = cls(config=config)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.vectorizer = data["vectorizer"]
        instance.tfidf_matrix = data["tfidf_matrix"]
        instance._fitted = data["_fitted"]
        logger.info(f"Loaded TFIDFVectorizerWrapper from {path}")
        return instance
