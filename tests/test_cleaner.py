"""Unit tests for JSON parsing, string normalization, and data cleaning functions."""

import pandas as pd

from recsys.data.cleaner import (
    calculate_imdb_weighted_rating,
    extract_directors,
    extract_names,
    extract_release_year,
    safe_parse_json,
    sanitize_entity_name,
)


def test_safe_parse_json() -> None:
    """Test parsing stringified JSON arrays."""
    valid_json_str = "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Adventure'}]"
    parsed = safe_parse_json(valid_json_str)
    assert len(parsed) == 2
    assert parsed[0]["name"] == "Action"

    # Invalid / empty cases
    assert safe_parse_json(None) == []
    assert safe_parse_json(float("nan")) == []
    assert safe_parse_json("not a list") == []
    assert safe_parse_json("['broken', json") == []


def test_sanitize_entity_name() -> None:
    """Test standardizing actor/director names into tokens."""
    assert sanitize_entity_name("Christopher Nolan") == "christopher_nolan"
    assert sanitize_entity_name("Tom Hanks!") == "tom_hanks"
    assert sanitize_entity_name("Jean-Claude Van Damme") == "jean-claude_van_damme"
    assert sanitize_entity_name("  Quentin Tarantino  ") == "quentin_tarantino"


def test_extract_names() -> None:
    """Test extracting names from dict lists."""
    items = [
        {"id": 1, "name": "Tom Hanks"},
        {"id": 2, "name": "Tim Allen"},
        {"id": 3, "name": "Don Rickles"},
    ]
    extracted = extract_names(items, max_items=2)
    assert extracted == ["tom_hanks", "tim_allen"]

    assert extract_names([]) == []
    assert extract_names([{"id": 1}]) == []


def test_extract_directors() -> None:
    """Test extracting directors from crew list."""
    crew = [
        {"job": "Director", "name": "Christopher Nolan"},
        {"job": "Producer", "name": "Emma Thomas"},
        {"job": "Director", "name": "Christopher Nolan"},  # Duplicate
    ]
    directors = extract_directors(crew)
    assert directors == ["christopher_nolan"]


def test_extract_release_year() -> None:
    """Test extracting 4-digit release year."""
    assert extract_release_year("1995-10-30") == 1995
    assert extract_release_year("2010-07-16") == 2010
    assert extract_release_year("Invalid date") is None
    assert extract_release_year(None) is None
    assert extract_release_year("1800-01-01") is None  # Below 1880 threshold


def test_calculate_imdb_weighted_rating() -> None:
    """Test IMDb weighted rating formula."""
    df = pd.DataFrame(
        {
            "vote_count": [1000, 50, 10, 500],
            "vote_average": [8.5, 9.0, 9.5, 7.0],
        }
    )
    weighted = calculate_imdb_weighted_rating(df, percentile=0.5)

    assert len(weighted) == 4
    # The movie with high votes (1000) should stay close to its vote_average
    assert abs(weighted.iloc[0] - 8.5) < 0.5
    # The movie with very few votes (10) should be pulled towards the mean
    assert weighted.iloc[2] < 9.5
