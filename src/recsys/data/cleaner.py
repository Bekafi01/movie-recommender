"""Data cleaning, JSON parsing, ID mapping, and IMDb weighted rating calculations."""

from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, load_config
from ..utils.logger import get_logger
from ..utils.timer import timed

logger = get_logger("recsys.data.cleaner")


def safe_parse_json(val: Any) -> list[dict[str, Any]]:
    """Safely parse stringified JSON lists using ast.literal_eval."""
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        parsed = ast.literal_eval(val)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def sanitize_entity_name(name: str) -> str:
    """Normalize names (e.g., 'Tom Hanks' -> 'tom_hanks')."""
    return re.sub(r"[^\w\s-]", "", str(name)).strip().lower().replace(" ", "_")


def extract_names(items: list[dict[str, Any]], max_items: int | None = None) -> list[str]:
    """Extract and sanitize 'name' fields from a list of dicts."""
    names: list[str] = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            clean = sanitize_entity_name(item["name"])
            if clean and clean not in names:
                names.append(clean)
        if max_items and len(names) >= max_items:
            break
    return names


def extract_directors(crew_items: list[dict[str, Any]]) -> list[str]:
    """Extract director names from crew list."""
    directors: list[str] = []
    for member in crew_items:
        if isinstance(member, dict) and member.get("job") == "Director" and member.get("name"):
            clean = sanitize_entity_name(member["name"])
            if clean and clean not in directors:
                directors.append(clean)
    return directors


def extract_release_year(date_str: Any) -> int | None:
    """Extract 4-digit release year from date string (e.g., '1995-10-30' -> 1995)."""
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None
    match = re.match(r"^(\d{4})", date_str.strip())
    if match:
        year = int(match.group(1))
        return year if 1880 <= year <= 2030 else None
    return None


def calculate_imdb_weighted_rating(
    df: pd.DataFrame,
    percentile: float = 0.80,
    vote_count_col: str = "vote_count",
    vote_avg_col: str = "vote_average",
) -> pd.Series:
    """Compute IMDb formula: WR = (v / (v + m)) * R + (m / (v + m)) * C."""
    v = df[vote_count_col].fillna(0)
    r = df[vote_avg_col].fillna(0)
    c = r.mean()
    m = v.quantile(percentile)

    if m == 0 or np.isnan(m):
        m = 1.0

    weighted = (v / (v + m)) * r + (m / (v + m)) * c
    return weighted.round(2)


class DataCleaner:
    """End-to-end cleaning and mapping pipeline for movies and ratings."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()

    @timed("Cleaning links data")
    def clean_links(self, links_df: pd.DataFrame) -> pd.DataFrame:
        """Filter valid integer pairs of movieId <-> tmdbId from links_small."""
        df = links_df.dropna(subset=["movieId", "tmdbId"]).copy()
        df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
        df["tmdbId"] = pd.to_numeric(df["tmdbId"], errors="coerce")
        df = df.dropna(subset=["movieId", "tmdbId"])
        df["movieId"] = df["movieId"].astype(int)
        df["tmdbId"] = df["tmdbId"].astype(int)
        df = df.drop_duplicates(subset=["movieId"]).drop_duplicates(subset=["tmdbId"])
        logger.info(f"Cleaned links: {len(df)} valid movieId <-> tmdbId pairs.")
        return df

    @timed("Cleaning movies metadata")
    def clean_movies(
        self,
        movies_df: pd.DataFrame,
        links_clean: pd.DataFrame,
        keywords_df: pd.DataFrame,
        credits_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter corrupt rows, bridge IDs, parse JSON fields, and compute weighted ratings."""
        df = movies_df.copy()

        # 1. Filter corrupted non-numeric IDs (e.g., raw dataset bug with dates in ID column)
        df["tmdb_id"] = pd.to_numeric(df["id"], errors="coerce")
        corrupted_count = df["tmdb_id"].isna().sum()
        if corrupted_count > 0:
            logger.warning(f"Filtered out {corrupted_count} corrupted rows from movies_metadata.csv.")
        df = df.dropna(subset=["tmdb_id"]).copy()
        df["tmdb_id"] = df["tmdb_id"].astype(int)
        df = df.drop_duplicates(subset=["tmdb_id"])

        # 2. Restrict to ~9k movie scope via inner merge with links_clean
        merged = df.merge(links_clean, left_on="tmdb_id", right_on="tmdbId", how="inner")
        merged = merged.rename(columns={"movieId": "movie_id"})
        logger.info(f"Merged movies with links_small: {len(merged)} candidate movies.")

        # 3. Clean keywords
        kw = keywords_df.copy()
        kw["tmdb_id"] = pd.to_numeric(kw["id"], errors="coerce")
        kw = kw.dropna(subset=["tmdb_id"]).copy()
        kw["tmdb_id"] = kw["tmdb_id"].astype(int)
        kw = kw.drop_duplicates(subset=["tmdb_id"])
        kw["keywords_list"] = kw["keywords"].apply(lambda val: extract_names(safe_parse_json(val)))
        kw["keywords_str"] = kw["keywords_list"].apply(lambda items: " ".join(items))

        # 4. Clean credits (directors & top 4 cast)
        cr = credits_df.copy()
        cr["tmdb_id"] = pd.to_numeric(cr["id"], errors="coerce")
        cr = cr.dropna(subset=["tmdb_id"]).copy()
        cr["tmdb_id"] = cr["tmdb_id"].astype(int)
        cr = cr.drop_duplicates(subset=["tmdb_id"])
        cr["cast_list"] = cr["cast"].apply(
            lambda val: extract_names(safe_parse_json(val), max_items=4)
        )
        cr["cast_str"] = cr["cast_list"].apply(lambda items: " ".join(items))
        cr["directors_list"] = cr["crew"].apply(lambda val: extract_directors(safe_parse_json(val)))
        cr["director_str"] = cr["directors_list"].apply(lambda items: " ".join(items))

        # 5. Merge keywords and credits with movies
        merged = merged.merge(
            kw[["tmdb_id", "keywords_list", "keywords_str"]], on="tmdb_id", how="left"
        )
        merged = merged.merge(
            cr[["tmdb_id", "cast_list", "cast_str", "directors_list", "director_str"]],
            on="tmdb_id",
            how="left",
        )

        # 6. Parse genres, overview, year, vote metrics
        merged["genres_list"] = merged["genres"].apply(
            lambda val: extract_names(safe_parse_json(val))
        )
        merged["genres_str"] = merged["genres_list"].apply(lambda items: " ".join(items))
        merged["title"] = merged["title"].fillna(merged.get("original_title", "Untitled")).astype(str)
        merged["overview"] = merged["overview"].fillna("").astype(str)
        merged["tagline"] = merged["tagline"].fillna("").astype(str)
        merged["release_year"] = merged["release_date"].apply(extract_release_year)
        merged["vote_average"] = pd.to_numeric(merged["vote_average"], errors="coerce").fillna(0.0)
        merged["vote_count"] = pd.to_numeric(merged["vote_count"], errors="coerce").fillna(0).astype(int)
        merged["popularity"] = pd.to_numeric(merged["popularity"], errors="coerce").fillna(0.0)
        merged["poster_path"] = merged["poster_path"].fillna("").astype(str)

        # Fill list columns with empty lists where missing
        for col in ["keywords_list", "cast_list", "directors_list", "genres_list"]:
            merged[col] = merged[col].apply(lambda x: x if isinstance(x, list) else [])

        for col in ["keywords_str", "cast_str", "director_str", "genres_str"]:
            merged[col] = merged[col].fillna("").astype(str)

        # 7. Compute IMDb weighted rating
        percentile = self.config.popularity.min_vote_percentile
        merged["weighted_rating"] = calculate_imdb_weighted_rating(
            merged, percentile=percentile
        )

        # Select standard columns
        cols = [
            "movie_id",
            "tmdb_id",
            "title",
            "release_year",
            "genres_list",
            "genres_str",
            "keywords_list",
            "keywords_str",
            "cast_list",
            "cast_str",
            "directors_list",
            "director_str",
            "overview",
            "tagline",
            "vote_average",
            "vote_count",
            "popularity",
            "weighted_rating",
            "poster_path",
        ]
        final_df = merged[cols].drop_duplicates(subset=["movie_id"]).copy()
        logger.info(f"Cleaned movies catalog finalized: {len(final_df)} movies.")
        return final_df

    @timed("Cleaning ratings interactions")
    def clean_ratings(
        self,
        ratings_df: pd.DataFrame,
        links_clean: pd.DataFrame,
        valid_movie_ids: set[int],
    ) -> pd.DataFrame:
        """Filter ratings to valid movies and map to tmdbId."""
        df = ratings_df.copy()
        df["userId"] = pd.to_numeric(df["userId"], errors="coerce")
        df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
        df["userId"] = df["userId"].astype(int)
        df["movieId"] = df["movieId"].astype(int)
        df["timestamp"] = df["timestamp"].astype(int)
        df["rating"] = df["rating"].astype(float)

        # Filter to movies that exist in our clean movie catalog
        df = df[df["movieId"].isin(valid_movie_ids)].copy()

        # Map tmdbId
        link_map = dict(zip(links_clean["movieId"], links_clean["tmdbId"], strict=False))
        df["tmdbId"] = df["movieId"].map(link_map)
        df = df.dropna(subset=["tmdbId"]).copy()
        df["tmdbId"] = df["tmdbId"].astype(int)

        df = df.rename(columns={"userId": "user_id", "movieId": "movie_id", "tmdbId": "tmdb_id"})
        df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
        logger.info(f"Cleaned ratings: {len(df)} ratings across {df['user_id'].nunique()} users.")
        return df
