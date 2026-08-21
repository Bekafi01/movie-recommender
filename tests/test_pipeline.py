"""Integration tests for the full data pipeline."""

from pathlib import Path

import pandas as pd

from recsys.config import AppConfig
from recsys.data.pipeline import run_data_pipeline


def test_full_pipeline_mock_data(tmp_path: Path) -> None:
    """Test executing the end-to-end data pipeline on a mock mini dataset."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    # Create mock CSVs
    links_df = pd.DataFrame({"movieId": [1, 2], "imdbId": [1001, 1002], "tmdbId": [2001, 2002]})
    links_df.to_csv(raw_dir / "links_small.csv", index=False)

    ratings_df = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "rating": [4.0, 5.0, 3.5],
            "timestamp": [100, 200, 300],
        }
    )
    ratings_df.to_csv(raw_dir / "ratings_small.csv", index=False)

    movies_df = pd.DataFrame(
        {
            "id": ["2001", "2002", "1997-08-20"],  # Includes 1 corrupt row
            "title": ["Movie One", "Movie Two", "Corrupt Movie"],
            "overview": ["A great adventure", "A deep space thriller", "Corrupted"],
            "genres": [
                "[{'id': 1, 'name': 'Action'}]",
                "[{'id': 2, 'name': 'Sci-Fi'}]",
                "[{'id': 1, 'name': 'Action'}]",
            ],
            "release_date": ["1995-10-20", "2014-11-05", "2000-01-01"],
            "vote_average": ["7.5", "8.2", "5.0"],
            "vote_count": ["1000", "2500", "10"],
            "popularity": ["15.5", "28.3", "1.0"],
            "tagline": ["Tag 1", "Tag 2", "Tag 3"],
            "poster_path": ["/p1.jpg", "/p2.jpg", ""],
        }
    )
    movies_df.to_csv(raw_dir / "movies_metadata.csv", index=False)

    keywords_df = pd.DataFrame(
        {
            "id": [2001, 2002],
            "keywords": ["[{'id': 10, 'name': 'hero'}]", "[{'id': 20, 'name': 'space'}]"],
        }
    )
    keywords_df.to_csv(raw_dir / "keywords.csv", index=False)

    credits_df = pd.DataFrame(
        {
            "id": [2001, 2002],
            "cast": [
                "[{'name': 'Actor One'}, {'name': 'Actor Two'}]",
                "[{'name': 'Actor Three'}]",
            ],
            "crew": [
                "[{'job': 'Director', 'name': 'Director One'}]",
                "[{'job': 'Director', 'name': 'Director Two'}]",
            ],
        }
    )
    credits_df.to_csv(raw_dir / "credits.csv", index=False)

    # Configure paths
    config = AppConfig()
    config.paths.data_raw_dir = str(raw_dir)
    config.paths.data_processed_dir = str(processed_dir)

    summary = run_data_pipeline(config=config)

    assert summary["status"] == "success"
    assert summary["num_movies"] == 2
    assert summary["num_ratings"] == 3
    assert summary["num_users"] == 2

    # Verify output files exist
    for _, p in summary["paths"].items():
        assert Path(p).exists()
