"""Unit tests for Streamlit UI helper components."""

from recsys.ui.components import (
    get_poster_url,
    render_movie_card,
    render_navbar,
    render_spotlight_hero,
)


def test_get_poster_url() -> None:
    """Test TMDB poster path resolution and fallback placeholder."""
    # Valid TMDB poster path
    url_with_slash = get_poster_url("/poster123.jpg")
    assert url_with_slash == "https://image.tmdb.org/t/p/w500/poster123.jpg"

    url_without_slash = get_poster_url("poster123.jpg")
    assert url_without_slash == "https://image.tmdb.org/t/p/w500/poster123.jpg"

    # Missing / None / nan poster path -> returns Unsplash fallback placeholder
    assert "images.unsplash.com" in get_poster_url(None)
    assert "images.unsplash.com" in get_poster_url("")
    assert "images.unsplash.com" in get_poster_url("nan")


def test_ui_components_render() -> None:
    """Test that UI rendering functions execute without exceptions."""
    render_navbar()
    render_spotlight_hero()

    movie = {
        "movie_id": 1,
        "title": "Inception",
        "release_year": 2010,
        "vote_average": 8.4,
        "genres_str": "action sci-fi thriller",
        "score": 0.95,
        "poster_path": "/inception.jpg",
    }
    render_movie_card(movie, rank=1)
