"""Unit tests for DataRepository (Parquet & SQLite operations)."""

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import AppConfig
from recsys.data.db import DataRepository
from recsys.utils.exceptions import DataProcessingError


def test_repository_save_and_load(tmp_path: Path) -> None:
    """Test saving processed data to Parquet and SQLite and loading it back."""
    config = AppConfig()
    config.paths.data_processed_dir = str(tmp_path / "processed")

    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "tmdb_id": [101, 102],
            "title": ["Inception", "Interstellar"],
            "release_year": [2010, 2014],
            "genres_list": [["action", "sci-fi"], ["drama", "sci-fi"]],
            "genres_str": ["action sci-fi", "drama sci-fi"],
            "keywords_list": [["dream"], ["space"]],
            "keywords_str": ["dream", "space"],
            "cast_list": [["leonardo_dicaprio"], ["matthew_mcconaughey"]],
            "cast_str": ["leonardo_dicaprio", "matthew_mcconaughey"],
            "directors_list": [["christopher_nolan"], ["christopher_nolan"]],
            "director_str": ["christopher_nolan", "christopher_nolan"],
            "overview": ["dream heist", "space exploration"],
            "tagline": ["your mind is the scene", "mankind was born on earth"],
            "vote_average": [8.4, 8.6],
            "vote_count": [20000, 18000],
            "popularity": [45.0, 42.0],
            "weighted_rating": [8.3, 8.5],
            "poster_path": ["/path1.jpg", "/path2.jpg"],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "movie_id": [1, 2],
            "tmdb_id": [101, 102],
            "rating": [5.0, 4.5],
            "timestamp": [1000, 2000],
        }
    )

    soup_df = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "tmdb_id": [101, 102],
            "title": ["Inception", "Interstellar"],
            "soup": ["soup 1", "soup 2"],
        }
    )

    repo = DataRepository(config=config)
    paths = repo.save_processed(movies_df, ratings_df, soup_df)

    assert Path(paths["movies_parquet"]).exists()
    assert Path(paths["ratings_parquet"]).exists()
    assert Path(paths["soup_parquet"]).exists()
    assert Path(paths["sqlite_db"]).exists()

    # Test loading
    loaded_movies = repo.load_movies()
    assert len(loaded_movies) == 2
    assert loaded_movies.iloc[0]["title"] == "Inception"

    loaded_ratings = repo.load_ratings()
    assert len(loaded_ratings) == 2

    loaded_soup = repo.load_metadata_soup()
    assert len(loaded_soup) == 2

    # Test search
    search_results = repo.search_movies("Incept")
    assert len(search_results) == 1
    assert search_results[0]["title"] == "Inception"


def test_repository_missing_files_error(tmp_path: Path) -> None:
    """Test error when files are not present."""
    config = AppConfig()
    config.paths.data_processed_dir = str(tmp_path / "non_existent")
    repo = DataRepository(config=config)

    with pytest.raises(DataProcessingError):
        repo.load_movies()

    with pytest.raises(DataProcessingError):
        repo.load_ratings()

    with pytest.raises(DataProcessingError):
        repo.load_metadata_soup()
