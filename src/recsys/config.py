"""Configuration models and YAML loader for the RecSys application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectSettings(BaseModel):
    name: str = "movie-recommender"
    version: str = "0.1.0"
    random_seed: int = 42
    log_level: str = "INFO"


class PathSettings(BaseModel):
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    models_dir: str = "artifacts/models"
    embeddings_dir: str = "artifacts/embeddings"
    benchmarks_dir: str = "artifacts/benchmarks"

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.data_raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve(self.data_processed_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def embeddings_path(self) -> Path:
        return self.resolve(self.embeddings_dir)

    @property
    def benchmarks_path(self) -> Path:
        return self.resolve(self.benchmarks_dir)


class RawFilesSettings(BaseModel):
    movies_metadata: str = "movies_metadata.csv"
    ratings: str = "ratings_small.csv"
    credits: str = "credits.csv"
    keywords: str = "keywords.csv"
    links: str = "links_small.csv"


class ProcessedFilesSettings(BaseModel):
    movies_clean: str = "movies_clean.parquet"
    ratings_clean: str = "ratings_clean.parquet"
    metadata_soup: str = "metadata_soup.parquet"
    movies_sqlite: str = "movies.db"


class TMDBClientSettings(BaseModel):
    image_base_url: str = "https://image.tmdb.org/t/p/w500"
    api_base_url: str = "https://api.themoviedb.org/3"


class PopularityModelSettings(BaseModel):
    min_vote_percentile: float = 0.80
    default_top_k: int = 10


class TFIDFSettings(BaseModel):
    max_features: int = 10000
    ngram_range: list[int] = Field(default_factory=lambda: [1, 2])
    stop_words: str = "english"
    sublinear_tf: bool = True


class EmbeddingsSettings(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True
    faiss_index_type: str = "IndexFlatIP"


class ContentBasedSettings(BaseModel):
    tfidf: TFIDFSettings = Field(default_factory=TFIDFSettings)
    embeddings: EmbeddingsSettings = Field(default_factory=EmbeddingsSettings)


class SVDSettings(BaseModel):
    n_factors: int = 50
    random_state: int = 42


class NeuralCFSettings(BaseModel):
    latent_dim_gmf: int = 32
    latent_dim_mlp: int = 32
    mlp_layers: list[int] = Field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 10
    negative_samples_ratio: int = 4
    positive_rating_threshold: float = 3.5


class CollaborativeSettings(BaseModel):
    svd: SVDSettings = Field(default_factory=SVDSettings)
    neural_cf: NeuralCFSettings = Field(default_factory=NeuralCFSettings)


class HybridSettings(BaseModel):
    content_weight: float = 0.5
    collaborative_weight: float = 0.5
    mmr_lambda: float = 0.7
    candidate_pool_size: int = 50


class EvaluationSettings(BaseModel):
    top_k_list: list[int] = Field(default_factory=lambda: [5, 10, 20])
    default_top_k: int = 10
    min_user_interactions: int = 5
    positive_rating_threshold: float = 3.5
    test_ratio_leave_k: int = 2
    random_state: int = 42
    metrics: list[str] = Field(
        default_factory=lambda: [
            "ndcg@k",
            "map@k",
            "recall@k",
            "precision@k",
            "hit_rate@k",
            "mrr@k",
            "catalog_coverage",
            "novelty",
            "intra_list_diversity",
        ]
    )


class AppConfig(BaseModel):
    """Unified application configuration."""

    project: ProjectSettings = Field(default_factory=ProjectSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    raw_files: RawFilesSettings = Field(default_factory=RawFilesSettings)
    processed_files: ProcessedFilesSettings = Field(default_factory=ProcessedFilesSettings)
    tmdb: TMDBClientSettings = Field(default_factory=TMDBClientSettings)
    popularity: PopularityModelSettings = Field(default_factory=PopularityModelSettings)
    content_based: ContentBasedSettings = Field(default_factory=ContentBasedSettings)
    collaborative: CollaborativeSettings = Field(default_factory=CollaborativeSettings)
    hybrid: HybridSettings = Field(default_factory=HybridSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)

    def get_raw_file_path(self, key: str) -> Path:
        filename = getattr(self.raw_files, key, key)
        return self.paths.raw_dir / filename

    def get_processed_file_path(self, key: str) -> Path:
        filename = getattr(self.processed_files, key, key)
        return self.paths.processed_dir / filename


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


def load_config(configs_dir: Path | None = None) -> AppConfig:
    """Load configuration from YAML files in the configs directory."""
    if configs_dir is None:
        configs_dir = PROJECT_ROOT / "configs"

    base_yaml = _load_yaml(configs_dir / "base_config.yaml")
    model_yaml = _load_yaml(configs_dir / "model_config.yaml")
    eval_yaml = _load_yaml(configs_dir / "eval_config.yaml")

    merged: dict[str, Any] = {}
    merged.update(base_yaml)

    if model_yaml:
        for k, v in model_yaml.items():
            merged[k] = v

    if eval_yaml and "evaluation" in eval_yaml:
        merged["evaluation"] = eval_yaml["evaluation"]

    return AppConfig(**merged)
