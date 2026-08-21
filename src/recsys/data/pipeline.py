"""End-to-end data pipeline orchestrating ingestion, cleaning, feature soup generation, and storage."""

from __future__ import annotations

from typing import Any

from ..config import AppConfig, load_config
from ..features.metadata_builder import build_metadata_soup
from ..utils.logger import get_logger
from ..utils.timer import timed
from .cleaner import DataCleaner
from .db import DataRepository
from .ingestion import RawDataIngestor

logger = get_logger("recsys.data.pipeline")


@timed("Full Data Ingestion and Preprocessing Pipeline")
def run_data_pipeline(config: AppConfig | None = None) -> dict[str, Any]:
    """Execute the complete data pipeline from raw 5 CSVs to clean Parquet and SQLite."""
    cfg = config or load_config()

    logger.info("Starting RecSys Data Pipeline...")

    # 1. Ingestion
    ingestor = RawDataIngestor(config=cfg)
    raw_data = ingestor.load_all()

    # 2. Cleaning & ID Mapping
    cleaner = DataCleaner(config=cfg)
    links_clean = cleaner.clean_links(raw_data["links"])
    movies_clean = cleaner.clean_movies(
        movies_df=raw_data["movies_metadata"],
        links_clean=links_clean,
        keywords_df=raw_data["keywords"],
        credits_df=raw_data["credits"],
    )

    valid_movie_ids = set(movies_clean["movie_id"])
    ratings_clean = cleaner.clean_ratings(
        ratings_df=raw_data["ratings"],
        links_clean=links_clean,
        valid_movie_ids=valid_movie_ids,
    )

    # 3. Metadata Soup Generation
    soup_df = build_metadata_soup(movies_clean)

    # 4. Storage to Parquet and SQLite
    repo = DataRepository(config=cfg)
    paths = repo.save_processed(
        movies_df=movies_clean,
        ratings_df=ratings_clean,
        soup_df=soup_df,
    )

    summary: dict[str, Any] = {
        "status": "success",
        "num_movies": len(movies_clean),
        "num_ratings": len(ratings_clean),
        "num_users": ratings_clean["user_id"].nunique(),
        "paths": {k: str(v) for k, v in paths.items()},
    }

    logger.info(
        f"Data Pipeline completed successfully!\n"
        f"  - Total Clean Movies: {summary['num_movies']:,}\n"
        f"  - Total Clean Ratings: {summary['num_ratings']:,}\n"
        f"  - Total Unique Users: {summary['num_users']:,}\n"
    )

    return summary


if __name__ == "__main__":
    run_data_pipeline()
