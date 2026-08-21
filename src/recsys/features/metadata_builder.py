"""Metadata soup builder combining keywords, cast, crew, genres, and text features."""

from __future__ import annotations

import pandas as pd

from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.features.metadata")


def build_metadata_soup_row(row: pd.Series) -> str:
    """Combine metadata fields into a weighted text soup for a single movie."""
    keywords = str(row.get("keywords_str", "")).strip()
    cast = str(row.get("cast_str", "")).strip()
    director = str(row.get("director_str", "")).strip()
    genres = str(row.get("genres_str", "")).strip()
    overview = str(row.get("overview", "")).strip().lower()
    tagline = str(row.get("tagline", "")).strip().lower()

    # Weighting: 3x director, 3x cast, 2x genres, 1x keywords, overview, tagline
    parts: list[str] = []
    if keywords:
        parts.append(keywords)
    if cast:
        parts.extend([cast] * 3)
    if director:
        parts.extend([director] * 3)
    if genres:
        parts.extend([genres] * 2)
    if tagline:
        parts.append(tagline)
    if overview:
        parts.append(overview)

    return " ".join(parts).strip()


@timed("Building Metadata Soup")
def build_metadata_soup(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Generate the metadata soup Series and return DataFrame with movie_id, tmdb_id, title, soup."""
    df = movies_df.copy()
    soup_series = df.apply(build_metadata_soup_row, axis=1)

    soup_df = pd.DataFrame(
        {
            "movie_id": df["movie_id"].astype(int),
            "tmdb_id": df["tmdb_id"].astype(int),
            "title": df["title"].astype(str),
            "soup": soup_series,
        }
    )

    logger.info(f"Built metadata soup for {len(soup_df)} movies.")
    return soup_df
