"""Unit tests for metadata soup builder."""

import pandas as pd

from recsys.features.metadata_builder import build_metadata_soup, build_metadata_soup_row


def test_build_metadata_soup_row() -> None:
    """Test weighted soup row string creation."""
    row = pd.Series(
        {
            "keywords_str": "space time_travel",
            "cast_str": "matthew_mcconaughey anne_hathaway",
            "director_str": "christopher_nolan",
            "genres_str": "drama science_fiction",
            "overview": "A team of explorers travel through a wormhole in space.",
            "tagline": "Mankind was born on Earth. It was never meant to die here.",
        }
    )

    soup = build_metadata_soup_row(row)
    assert "space time_travel" in soup
    assert "christopher_nolan" in soup
    assert "matthew_mcconaughey" in soup
    # Director should appear 3 times in the text
    assert soup.count("christopher_nolan") == 3
    # Cast should appear 3 times
    assert soup.count("matthew_mcconaughey") == 3
    # Genres should appear 2 times
    assert soup.count("science_fiction") == 2


def test_build_metadata_soup_dataframe() -> None:
    """Test generating soup DataFrame."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2],
            "tmdb_id": [101, 102],
            "title": ["Movie A", "Movie B"],
            "keywords_str": ["alien", "robot"],
            "cast_str": ["actor_a", "actor_b"],
            "director_str": ["director_a", "director_b"],
            "genres_str": ["action", "sci-fi"],
            "overview": ["overview a", "overview b"],
            "tagline": ["tagline a", "tagline b"],
        }
    )

    soup_df = build_metadata_soup(movies_df)
    assert len(soup_df) == 2
    assert list(soup_df.columns) == ["movie_id", "tmdb_id", "title", "soup"]
    assert "director_a" in soup_df.iloc[0]["soup"]
