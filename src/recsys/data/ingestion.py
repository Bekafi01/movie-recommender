"""Raw data ingestion and file validation for The Movies Dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import AppConfig, load_config
from ..utils.exceptions import DataIngestionError
from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.data.ingestion")

REQUIRED_COLUMNS: dict[str, list[str]] = {
    "links": ["movieId", "tmdbId"],
    "ratings": ["userId", "movieId", "rating", "timestamp"],
    "movies_metadata": ["id", "title", "overview", "genres", "vote_average", "vote_count"],
    "keywords": ["id", "keywords"],
    "credits": ["id", "cast", "crew"],
}


class RawDataIngestor:
    """Handles verification and ingestion of the 5 raw CSV files."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

    def verify_files_exist(self) -> dict[str, Path]:
        """Verify all 5 required raw CSV files are present in data/raw."""
        paths: dict[str, Path] = {
            "links": self.config.get_raw_file_path("links"),
            "ratings": self.config.get_raw_file_path("ratings"),
            "movies_metadata": self.config.get_raw_file_path("movies_metadata"),
            "keywords": self.config.get_raw_file_path("keywords"),
            "credits": self.config.get_raw_file_path("credits"),
        }

        missing = [name for name, p in paths.items() if not p.exists()]
        if missing:
            msg = (
                f"Missing required raw data files in '{self.config.paths.raw_dir}': {missing}. "
                "Please download them from Kaggle (The Movies Dataset) and place them in data/raw/."
            )
            logger.error(msg)
            raise DataIngestionError(msg)

        logger.info("All 5 raw data files verified successfully in data/raw/.")
        return paths

    @timed("Loading Links CSV")
    def load_links(self) -> pd.DataFrame:
        path = self.config.get_raw_file_path("links")
        df = pd.read_csv(path)
        self._validate_schema("links", df)
        return df

    @timed("Loading Ratings CSV")
    def load_ratings(self) -> pd.DataFrame:
        path = self.config.get_raw_file_path("ratings")
        df = pd.read_csv(path)
        self._validate_schema("ratings", df)
        return df

    @timed("Loading Movies Metadata CSV")
    def load_movies_metadata(self) -> pd.DataFrame:
        path = self.config.get_raw_file_path("movies_metadata")
        df = pd.read_csv(path, low_memory=False)
        self._validate_schema("movies_metadata", df)
        return df

    @timed("Loading Keywords CSV")
    def load_keywords(self) -> pd.DataFrame:
        path = self.config.get_raw_file_path("keywords")
        df = pd.read_csv(path)
        self._validate_schema("keywords", df)
        return df

    @timed("Loading Credits CSV")
    def load_credits(self) -> pd.DataFrame:
        path = self.config.get_raw_file_path("credits")
        df = pd.read_csv(path)
        self._validate_schema("credits", df)
        return df

    def load_all(self) -> dict[str, pd.DataFrame]:
        """Load all 5 raw datasets into DataFrames."""
        self.verify_files_exist()
        return {
            "links": self.load_links(),
            "ratings": self.load_ratings(),
            "movies_metadata": self.load_movies_metadata(),
            "keywords": self.load_keywords(),
            "credits": self.load_credits(),
        }

    def _validate_schema(self, dataset_name: str, df: pd.DataFrame) -> None:
        required = REQUIRED_COLUMNS.get(dataset_name, [])
        missing = [col for col in required if col not in df.columns]
        if missing:
            msg = f"Dataset '{dataset_name}' is missing required columns: {missing}."
            logger.error(msg)
            raise DataIngestionError(msg)
