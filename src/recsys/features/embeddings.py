"""Dense Sentence-Transformer embeddings generator and FAISS vector index."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..config import AppConfig, load_config
from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.features.embeddings")


class DenseEmbeddingPipeline:
    """Generates dense semantic embeddings and manages FAISS vector index."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.embed_cfg = self.config.content_based.embeddings
        self._model: SentenceTransformer | None = None
        self.embeddings: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self.movie_ids: list[int] = []

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.embed_cfg.model_name}'...")
            self._model = SentenceTransformer(self.embed_cfg.model_name)
        return self._model

    @timed("Generating Dense Embeddings with Sentence-Transformers")
    def generate_embeddings(
        self,
        soup_df: pd.DataFrame,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Generate 384-dimensional dense vectors for all movie metadata soups."""
        b_size = batch_size or self.embed_cfg.batch_size
        texts = soup_df["soup"].fillna("").tolist()
        self.movie_ids = soup_df["movie_id"].tolist()

        logger.info(f"Encoding {len(texts)} movie soups with batch size {b_size}...")
        raw_embeddings = self.model.encode(
            texts,
            batch_size=b_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.embed_cfg.normalize_embeddings,
        )

        self.embeddings = np.ascontiguousarray(raw_embeddings, dtype=np.float32)
        logger.info(
            f"Dense embeddings generated: shape {self.embeddings.shape}, "
            f"size: {self.embeddings.nbytes / (1024 * 1024):.2f} MB."
        )

        # Build FAISS Index
        self.build_faiss_index()
        return self.embeddings

    def build_faiss_index(self) -> faiss.Index:
        """Build FAISS IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings have not been generated yet.")

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        logger.info(
            f"FAISS IndexFlatIP built with {self.index.ntotal} vectors of dimension {dimension}."
        )
        return self.index

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single natural language search query."""
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.embed_cfg.normalize_embeddings,
        )
        return np.ascontiguousarray(vec, dtype=np.float32)

    def search_vector(
        self, query_vec: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search FAISS index with query vector, returning (similarity_scores, movie_indices)."""
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded or built.")
        query = np.ascontiguousarray(query_vec, dtype=np.float32)
        distances, indices = self.index.search(query, top_k)
        return distances[0], indices[0]

    def save(self, embeddings_dir: Path | None = None) -> dict[str, Path]:
        """Persist embeddings array and FAISS index to disk."""
        target_dir = embeddings_dir or self.config.paths.embeddings_path
        target_dir.mkdir(parents=True, exist_ok=True)

        emb_path = target_dir / "movie_embeddings.npy"
        idx_path = target_dir / "faiss_index.bin"
        meta_path = target_dir / "movie_ids.npy"

        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
            np.save(meta_path, np.array(self.movie_ids, dtype=np.int64))

        if self.index is not None:
            faiss.write_index(self.index, str(idx_path))

        logger.info(f"Saved FAISS index and embeddings to {target_dir}")
        return {"embeddings": emb_path, "faiss_index": idx_path, "movie_ids": meta_path}

    @classmethod
    def load(
        cls,
        embeddings_dir: Path | None = None,
        config: AppConfig | None = None,
    ) -> DenseEmbeddingPipeline:
        """Load persisted embeddings and FAISS index from disk."""
        cfg = config or load_config()
        target_dir = embeddings_dir or cfg.paths.embeddings_path

        emb_path = target_dir / "movie_embeddings.npy"
        idx_path = target_dir / "faiss_index.bin"
        meta_path = target_dir / "movie_ids.npy"

        instance = cls(config=cfg)

        if emb_path.exists():
            instance.embeddings = np.load(emb_path)
        if meta_path.exists():
            instance.movie_ids = np.load(meta_path).tolist()
        if idx_path.exists():
            instance.index = faiss.read_index(str(idx_path))
            logger.info(f"Loaded FAISS index with {instance.index.ntotal} vectors.")

        return instance
