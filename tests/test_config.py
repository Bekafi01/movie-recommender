"""Unit tests for configuration loading and path resolution."""

from pathlib import Path

from recsys.config import AppConfig, load_config


def test_default_config_loading() -> None:
    """Test loading configuration from default YAML files."""
    config = load_config()

    assert isinstance(config, AppConfig)
    assert config.project.name == "movie-recommender"
    assert config.project.random_seed == 42
    assert config.popularity.min_vote_percentile == 0.80
    assert config.content_based.tfidf.max_features == 10000
    assert config.content_based.embeddings.model_name == "all-MiniLM-L6-v2"
    assert config.collaborative.svd.n_factors == 50
    assert config.collaborative.neural_cf.latent_dim_gmf == 32
    assert config.hybrid.content_weight == 0.5
    assert 10 in config.evaluation.top_k_list


def test_path_resolution() -> None:
    """Test that path resolution returns proper absolute Path objects."""
    config = load_config()

    assert isinstance(config.paths.raw_dir, Path)
    assert isinstance(config.paths.processed_dir, Path)
    assert isinstance(config.paths.artifacts_path, Path)
    assert isinstance(config.paths.models_path, Path)
    assert isinstance(config.paths.embeddings_path, Path)

    raw_movies = config.get_raw_file_path("movies_metadata")
    assert raw_movies.name == "movies_metadata.csv"
    assert "data" in str(raw_movies)

    clean_movies = config.get_processed_file_path("movies_clean")
    assert clean_movies.name == "movies_clean.parquet"


def test_empty_config_fallback(tmp_path: Path) -> None:
    """Test fallback to defaults when loading from an empty directory."""
    config = load_config(configs_dir=tmp_path)
    assert config.project.name == "movie-recommender"
    assert config.popularity.min_vote_percentile == 0.80
