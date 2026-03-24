from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd


def extract_title_year(title_str: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)\s\((\d{4})\)$", title_str.strip())
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title_str.strip(), None


def build_movies_database(data_dir: Path, db_path: Path) -> pd.DataFrame:
    movies_df = pd.read_csv(data_dir / "ml - 1m" / "movies.csv")
    ratings_df = pd.read_csv(data_dir / "ml - 1m" / "ratings.csv")

    movies_df[["clean_title", "year"]] = movies_df["title"].apply(
        lambda value: pd.Series(extract_title_year(value))
    )

    ratings_agg = ratings_df.groupby("movieId")["rating"].mean().reset_index()
    ratings_agg.columns = ["movieId", "rating"]
    ratings_agg["rating"] = ratings_agg["rating"].round().astype("Int64")

    merged_df = movies_df.merge(ratings_agg, on="movieId", how="left")
    merged_df = merged_df[
        ["movieId", "clean_title", "year", "genres", "rating"]
    ].rename(columns={"movieId": "movie_id", "clean_title": "title"})

    merged_df["rating"] = merged_df["rating"].fillna(0.0)
    merged_df["year"] = merged_df["year"].fillna(0).astype(int)
    merged_df = merged_df.drop_duplicates(subset=["movie_id"], keep="first")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        merged_df.to_sql("movies", conn, if_exists="replace", index=False)

    return merged_df


def load_movies_from_db(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM movies", conn)
