"""Unit tests for Streamlit UI helper components and PosterResolver."""

from pathlib import Path

from recsys.data.poster_fetcher import PosterResolver
from recsys.ui.components import (
    get_poster_url,
    render_movie_card,
    render_navbar,
    render_spotlight_hero,
)


def test_poster_resolver(tmp_path: Path) -> None:
    """Test PosterResolver cache creation and resolution."""
    cache_file = tmp_path / "posters_cache.json"
    resolver = PosterResolver(cache_file=cache_file)

    # Test cache persistence
    resolver.cache["Inception_2010"] = "https://image.tmdb.org/t/p/w500/test_inception.jpg"
    resolver.save_cache()

    # Re-load
    resolver2 = PosterResolver(cache_file=cache_file)
    assert (
        resolver2.resolve("Inception", year=2010)
        == "https://image.tmdb.org/t/p/w500/test_inception.jpg"
    )


def test_get_poster_url() -> None:
    """Test get_poster_url fallback behavior."""
    url = get_poster_url(title="Nonexistent Movie 12345", year=1900)
    assert url.startswith("http")


def test_ui_components_render() -> None:
    """Test that UI rendering functions execute without exceptions."""
    render_navbar()
    render_spotlight_hero()

    movie = {
        "movie_id": 1,
        "tmdb_id": 27205,
        "title": "Inception",
        "release_year": 2010,
        "vote_average": 8.4,
        "genres_str": "action sci-fi thriller",
        "score": 0.95,
        "poster_path": "/inception.jpg",
    }
    render_movie_card(movie, rank=1)
