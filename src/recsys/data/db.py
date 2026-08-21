"""Data persistence and query repository supporting Parquet and SQLite."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import AppConfig, load_config
from ..utils.exceptions import DataProcessingError
from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.data.db")


class DataRepository:
    """Repository for saving and loading processed datasets."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

    @timed("Saving processed Parquet and SQLite datasets")
    def save_processed(
        self,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        soup_df: pd.DataFrame,
    ) -> dict[str, Path]:
        """Persist clean data to Parquet and SQLite."""
        processed_dir = self.config.paths.processed_dir
        processed_dir.mkdir(parents=True, exist_ok=True)

        movies_path = self.config.get_processed_file_path("movies_clean")
        ratings_path = self.config.get_processed_file_path("ratings_clean")
        soup_path = self.config.get_processed_file_path("metadata_soup")
        sqlite_path = self.config.get_processed_file_path("movies_sqlite")

        # 1. Save Parquet
        # Convert list columns to JSON strings for SQLite compatibility, keep Parquet rich
        movies_df.to_parquet(movies_path, index=False)
        ratings_df.to_parquet(ratings_path, index=False)
        soup_df.to_parquet(soup_path, index=False)

        # 2. Save SQLite
        # Convert list columns to string representation for SQLite
        sqlite_movies = movies_df.copy()
        for col in ["genres_list", "keywords_list", "cast_list", "directors_list"]:
            if col in sqlite_movies.columns:
                sqlite_movies[col] = sqlite_movies[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        with sqlite3.connect(sqlite_path) as conn:
            sqlite_movies.to_sql("movies", conn, if_exists="replace", index=False)
            ratings_df.to_sql("ratings", conn, if_exists="replace", index=False)
            soup_df.to_sql("metadata_soup", conn, if_exists="replace", index=False)

            # Create indexes for sub-millisecond lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_movie_id ON movies (movie_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies (tmdb_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_title ON movies (title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings (user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings (movie_id)")

        logger.info(f"Successfully persisted datasets to {processed_dir}")
        return {
            "movies_parquet": movies_path,
            "ratings_parquet": ratings_path,
            "soup_parquet": soup_path,
            "sqlite_db": sqlite_path,
        }

    def load_movies(self) -> pd.DataFrame:
        """Load clean movies dataset from Parquet (or SQLite fallback)."""
        path = self.config.get_processed_file_path("movies_clean")
        if path.exists():
            return pd.read_parquet(path)
        sqlite_path = self.config.get_processed_file_path("movies_sqlite")
        if sqlite_path.exists():
            with sqlite3.connect(sqlite_path) as conn:
                return pd.read_sql_query("SELECT * FROM movies", conn)
        raise DataProcessingError(f"Clean movies dataset not found at {path}. Run data preprocessing first.")

    def load_ratings(self) -> pd.DataFrame:
        """Load clean ratings dataset from Parquet."""
        path = self.config.get_processed_file_path("ratings_clean")
        if path.exists():
            return pd.read_parquet(path)
        raise DataProcessingError(f"Clean ratings dataset not found at {path}. Run data preprocessing first.")

    def load_metadata_soup(self) -> pd.DataFrame:
        """Load metadata soup dataset from Parquet."""
        path = self.config.get_processed_file_path("metadata_soup")
        if path.exists():
            return pd.read_parquet(path)
        raise DataProcessingError(f"Metadata soup dataset not found at {path}. Run data preprocessing first.")

    def search_movies(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Fast SQL search for movies by title substring."""
        sqlite_path = self.config.get_processed_file_path("movies_sqlite")
        if not sqlite_path.exists():
            # Fallback to in-memory DataFrame search
            df = self.load_movies()
            matches = df[df["title"].str.contains(query, case=False, na=False)].head(limit)
            return matches.to_dict(orient="records")

        with sqlite3.connect(sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT movie_id, tmdb_id, title, release_year, genres_str, vote_average, weighted_rating, poster_path
                FROM movies
                WHERE title LIKE ?
                ORDER BY vote_count DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
