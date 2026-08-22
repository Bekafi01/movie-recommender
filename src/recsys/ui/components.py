"""Ultra-Premium UI Components for the CineFlow AI Cinema Experience."""

from __future__ import annotations

from typing import Any

import streamlit as st

from recsys.data.poster_fetcher import PosterResolver

# Global cached poster resolver with persistent JSON cache
_POSTER_RESOLVER = PosterResolver()


def get_poster_url(
    title: str = "",
    year: int | str | None = None,
    tmdb_id: int | str | None = None,
    poster_path: str | None = None,
) -> str:
    """Resolve authentic movie poster URL via TMDB API v3 / persistent cache."""
    return _POSTER_RESOLVER.resolve(title=title, year=year, tmdb_id=tmdb_id, tmdb_path=poster_path)


def render_navbar() -> None:
    """Render top floating luxury navigation bar."""
    navbar_html = (
        '<div class="cine-navbar">'
        '<div class="cine-logo-container">'
        '<div class="cine-logo-icon">🎬</div>'
        '<div class="cine-logo-text">CINEFLOW <span style="color: #2563eb;">AI</span></div>'
        "</div>"
        '<div style="display: flex; align-items: center; gap: 1rem;">'
        '<div class="cine-badge-live">'
        '<div class="cine-pulse-dot"></div>'
        "<span>ENGINE ONLINE</span>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(navbar_html, unsafe_allow_html=True)


def render_spotlight_hero() -> None:
    """Render cinematic spotlight hero header with key platform metrics."""
    hero_html = (
        '<div class="spotlight-hero">'
        '<div class="spotlight-tag">⭐ Multi-Paradigm RecSys Engine</div>'
        '<div class="spotlight-title">Cinematic Intelligence,<br>Tailored To Your Taste.</div>'
        '<div class="spotlight-desc">'
        "Traverse 384-dimensional dense semantic vector space, collaborative latent factor manifolds, "
        "and deep neural architectures for ultra-fast, personalized movie discovery."
        "</div>"
        '<div class="spotlight-stats-row">'
        '<div class="spotlight-stat-item">'
        '<div class="spotlight-stat-num">9,082</div>'
        '<div class="spotlight-stat-lbl">Indexed Titles</div>'
        "</div>"
        '<div class="spotlight-stat-item">'
        '<div class="spotlight-stat-num">99,810</div>'
        '<div class="spotlight-stat-lbl">Ratings Modeled</div>'
        "</div>"
        '<div class="spotlight-stat-item">'
        '<div class="spotlight-stat-num">&lt; 3.5 ms</div>'
        '<div class="spotlight-stat-lbl">FAISS Vector Retrieval</div>'
        "</div>"
        '<div class="spotlight-stat-item">'
        '<div class="spotlight-stat-num">6 Engines</div>'
        '<div class="spotlight-stat-lbl">Multi-Paradigm ML</div>'
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)


def render_movie_card(movie: dict[str, Any], rank: int | None = None) -> None:
    """Render a luxury cinema movie card with authentic poster image."""
    title = (
        str(movie.get("title", "Untitled"))
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    year = movie.get("release_year", "")
    year_str = f"({int(year)})" if year and str(year) != "nan" and str(year).isdigit() else ""

    poster_url = get_poster_url(
        title=movie.get("title", ""),
        year=year,
        tmdb_id=movie.get("tmdb_id"),
        poster_path=movie.get("poster_path"),
    )
    vote_avg = movie.get("vote_average", 0.0)
    vote_str = f"★ {float(vote_avg):.1f}" if vote_avg and float(vote_avg) > 0 else ""

    # Floating Badges
    rank_badge = f'<div class="cine-badge-rank">#{rank}</div>' if rank else ""

    score = movie.get("score")
    score_badge = ""
    if score is not None:
        val = float(score)
        if 0.0 <= val <= 1.0:
            score_badge = f'<div class="cine-badge-match">{val * 100:.0f}% Match</div>'
        else:
            score_badge = f'<div class="cine-badge-score">{val:.2f} pts</div>'

    # Genres chips
    genres_str = str(movie.get("genres_str", ""))
    genres_list = [g.strip().replace("_", " ").title() for g in genres_str.split() if g.strip()][:3]
    genre_chips = "".join([f'<span class="cine-genre-chip">{g}</span>' for g in genres_list])

    card_html = (
        '<div class="cine-card">'
        '<div class="cine-poster-wrap">'
        f'<img src="{poster_url}" class="cine-poster-img" alt="{title}" onerror="this.onerror=null;this.src=\'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80\';" />'
        f"{rank_badge}"
        f"{score_badge}"
        "</div>"
        "<div>"
        f'<div class="cine-movie-title" title="{title}">{title}</div>'
        '<div class="cine-movie-meta">'
        f"<span>{year_str}</span>"
        f'<span class="cine-rating-gold">{vote_str}</span>'
        "</div>"
        f"<div>{genre_chips}</div>"
        "</div>"
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_movie_grid(
    movies: list[dict[str, Any]],
    num_cols: int = 5,
    show_explain: bool = False,
    explain_engine: Any = None,
    source_movie_id: int | None = None,
) -> None:
    """Render responsive grid of luxury movie cards with parallel pre-resolution for sub-100ms loading."""
    if not movies:
        st.info("No recommendations found matching current criteria.")
        return

    # Fast parallel batch resolution across all movies in current grid
    _POSTER_RESOLVER.bulk_enrich(movies, max_workers=min(len(movies), 10))

    for i in range(0, len(movies), num_cols):
        row_movies = movies[i : i + num_cols]
        cols = st.columns(num_cols)
        for col, movie_data in zip(cols, row_movies, strict=False):
            with col:
                rank = movie_data.get("rank", i + row_movies.index(movie_data) + 1)
                render_movie_card(movie_data, rank=rank)

                if show_explain and explain_engine and source_movie_id:
                    with st.expander("💡 Match Insights", expanded=False):
                        rec_id = int(movie_data.get("movie_id", 0))
                        sim = float(movie_data.get("score", 0.0))
                        explanation = explain_engine.explain(
                            source_movie_id=source_movie_id,
                            recommended_movie_id=rec_id,
                            similarity_score=sim,
                        )
                        explain_box = (
                            '<div class="cine-explain-box">'
                            f'<strong style="color: #1d4ed8;">{explanation["match_percentage"]} Thematic Match</strong><br>'
                            f"{explanation['summary']}"
                            "</div>"
                        )
                        st.markdown(explain_box, unsafe_allow_html=True)
