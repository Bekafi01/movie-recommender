"""Ultra-Premium UI Components for the CineFlow AI Cinema Experience."""

from __future__ import annotations

from typing import Any

import streamlit as st


def get_poster_url(poster_path: str | None) -> str:
    """Resolve TMDB CDN poster URL or high-quality dark cinema placeholder."""
    if (
        poster_path
        and str(poster_path).strip()
        and str(poster_path) != "None"
        and str(poster_path) != "nan"
    ):
        clean_path = str(poster_path).strip()
        if not clean_path.startswith("/"):
            clean_path = f"/{clean_path}"
        return f"https://image.tmdb.org/t/p/w500{clean_path}"
    return "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"


def render_navbar() -> None:
    """Render top floating luxury navigation bar."""
    st.markdown(
        """
        <div class="cine-navbar">
            <div class="cine-logo-container">
                <div class="cine-logo-icon">🎬</div>
                <div class="cine-logo-text">CINEFLOW <span style="color: #8b5cf6;">AI</span></div>
            </div>
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div class="cine-badge-live">
                    <div class="cine-pulse-dot"></div>
                    <span>ENGINE ONLINE</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_spotlight_hero() -> None:
    """Render cinematic spotlight hero header with key platform metrics."""
    st.markdown(
        """
        <div class="spotlight-hero">
            <div class="spotlight-tag">⭐ Multi-Paradigm RecSys Engine</div>
            <div class="spotlight-title">Cinematic Intelligence,<br>Tailored To Your Taste.</div>
            <div class="spotlight-desc">
                Traverse 384-dimensional dense semantic vector space, collaborative latent factor manifolds,
                and deep neural architectures for ultra-fast, personalized movie discovery.
            </div>
            <div class="spotlight-stats-row">
                <div class="spotlight-stat-item">
                    <div class="spotlight-stat-num">9,082</div>
                    <div class="spotlight-stat-lbl">Indexed Titles</div>
                </div>
                <div class="spotlight-stat-item">
                    <div class="spotlight-stat-num">99,810</div>
                    <div class="spotlight-stat-lbl">Ratings Modeled</div>
                </div>
                <div class="spotlight-stat-item">
                    <div class="spotlight-stat-num">&lt; 3.5 ms</div>
                    <div class="spotlight-stat-lbl">FAISS Vector Retrieval</div>
                </div>
                <div class="spotlight-stat-item">
                    <div class="spotlight-stat-num">6 Engines</div>
                    <div class="spotlight-stat-lbl">Multi-Paradigm ML</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_movie_card(movie: dict[str, Any], rank: int | None = None) -> None:
    """Render a luxury cinema movie card with floating badges and hover zoom."""
    poster_url = get_poster_url(movie.get("poster_path"))
    title = movie.get("title", "Untitled")
    year = movie.get("release_year", "")
    year_str = f"({int(year)})" if year and str(year) != "nan" and str(year).isdigit() else ""
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

    card_html = f"""
    <div class="cine-card">
        <div class="cine-poster-wrap">
            <img src="{poster_url}" class="cine-poster-img" alt="{title}" onerror="this.src='https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80';"/>
            {rank_badge}
            {score_badge}
        </div>
        <div>
            <div class="cine-movie-title" title="{title}">{title}</div>
            <div class="cine-movie-meta">
                <span>{year_str}</span>
                <span class="cine-rating-gold">{vote_str}</span>
            </div>
            <div>{genre_chips}</div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_movie_grid(
    movies: list[dict[str, Any]],
    num_cols: int = 5,
    show_explain: bool = False,
    explain_engine: Any = None,
    source_movie_id: int | None = None,
) -> None:
    """Render responsive grid of luxury movie cards with optional explainability drawers."""
    if not movies:
        st.info("No recommendations found matching current criteria.")
        return

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
                        st.markdown(
                            f"<div class='cine-explain-box'>"
                            f"<strong style='color: #a855f7;'>{explanation['match_percentage']} Thematic Match</strong><br>"
                            f"{explanation['summary']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
