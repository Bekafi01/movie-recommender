"""Main Streamlit Application for the CineFlow AI Movie Recommendation Experience."""

from __future__ import annotations

import importlib
from typing import Any

import pandas as pd
import streamlit as st

import recsys.ui.components
from recsys.config import load_config
from recsys.data.db import DataRepository
from recsys.models.collaborative import SVDCollaborativeRecommender
from recsys.models.content_based import SemanticVectorRecommender, TFIDFRecommender
from recsys.models.explainability import ExplainabilityEngine
from recsys.models.hybrid import HybridRecommender
from recsys.models.neural_cf import NeuralCollaborativeRecommender
from recsys.models.popularity import PopularityRecommender
from recsys.ui.components import render_movie_grid, render_navbar, render_spotlight_hero
from recsys.ui.styles import CUSTOM_CSS

# Ensure hot reloads in Streamlit always pick up latest component definitions
importlib.reload(recsys.ui.components)

# Page Configuration
st.set_page_config(
    page_title="CineFlow: Next-Gen Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Custom Dark Cinema Design System
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing CineFlow neural engines & FAISS vector manifolds...")
def load_all_resources() -> dict[str, Any]:
    """Preload datasets and model artifacts with caching for sub-millisecond UI interactions."""
    cfg = load_config()
    repo = DataRepository(config=cfg)
    movies_df = repo.load_movies()
    ratings_df = repo.load_ratings()

    models: dict[str, Any] = {
        "repo": repo,
        "movies_df": movies_df,
        "ratings_df": ratings_df,
        "explainability": ExplainabilityEngine(movies_df=movies_df),
    }

    # Load Model Artifacts
    models_dir = cfg.paths.models_path
    embed_dir = cfg.paths.embeddings_path

    # Popularity
    pop_path = models_dir / "popularity_model.pkl"
    if pop_path.exists():
        models["popularity"] = PopularityRecommender.load(pop_path, config=cfg)

    # TF-IDF
    tfidf_path = models_dir / "tfidf_model.pkl"
    if tfidf_path.exists():
        models["tfidf"] = TFIDFRecommender.load(tfidf_path, config=cfg)

    # Semantic FAISS
    sem_path = embed_dir / "semantic_meta.pkl"
    if sem_path.exists():
        models["semantic"] = SemanticVectorRecommender.load(sem_path, config=cfg)

    # SVD
    svd_path = models_dir / "svd_model.pkl"
    if svd_path.exists():
        models["svd"] = SVDCollaborativeRecommender.load(svd_path, config=cfg)

    # Neural CF
    ncf_path = models_dir / "neumf_model.pt"
    if ncf_path.exists():
        models["neural_cf"] = NeuralCollaborativeRecommender.load(ncf_path, config=cfg)

    # Hybrid
    models["hybrid"] = HybridRecommender(
        content_model=models.get("semantic"),
        collab_model=models.get("svd"),
        popularity_model=models.get("popularity"),
        config=cfg,
    )
    models["hybrid"].movies_df = movies_df
    models["hybrid"]._fitted = True

    return models


def main() -> None:
    """Main application loop."""
    resources = load_all_resources()
    movies_df: pd.DataFrame = resources["movies_df"]
    ratings_df: pd.DataFrame = resources["ratings_df"]
    explain_engine: ExplainabilityEngine = resources["explainability"]

    # Render Luxury Navigation Bar & Hero Spotlight
    render_navbar()
    render_spotlight_hero()

    # Mode Selector Tabs
    mode = st.radio(
        "Navigation Experience:",
        [
            "🎬 Because You Watched...",
            "🔮 Build Your Taste Profile",
            "👤 Personalized User Feed",
            "🏆 Blockbusters & Hidden Gems",
            "🔍 AI Thematic & Plot Search",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 1: BECAUSE YOU WATCHED (Item-to-Item Semantic Search)
    # =========================================================================
    if mode == "🎬 Because You Watched...":
        st.markdown("### 🎬 Item-to-Item Semantic & Thematic Discovery")
        st.caption("Discover titles sharing narrative DNA, directing styles, and cinematic tone.")

        col_search, col_algo, col_k = st.columns([3, 2, 1])

        with col_search:
            movie_title = st.selectbox(
                "Search or Select a Movie:",
                options=sorted(movies_df["title"].unique()),
                index=(
                    sorted(movies_df["title"].unique()).index("Inception")
                    if "Inception" in movies_df["title"].values
                    else 0
                ),
            )

        with col_algo:
            algo = st.selectbox(
                "Similarity Engine:",
                ["Dense Semantic Vectors (FAISS 384-d)", "Sparse Keyword Matching (TF-IDF)"],
            )

        with col_k:
            top_k = st.slider("Results:", 5, 20, 10, step=5)

        if st.button("🚀 Explore Recommendations", type="primary", use_container_width=True):
            with st.spinner("Traversing high-dimensional embedding space..."):
                if "Dense" in algo and "semantic" in resources:
                    recs_df = resources["semantic"].recommend(query=movie_title, top_k=top_k)
                elif "tfidf" in resources:
                    recs_df = resources["tfidf"].recommend(query=movie_title, top_k=top_k)
                else:
                    st.error("Selected model is not available.")
                    return

                source_row = movies_df[movies_df["title"].str.lower() == str(movie_title).lower()]
                source_id = int(source_row.iloc[0]["movie_id"]) if not source_row.empty else None

                st.markdown(f"#### ✨ Top Recommendations for *{movie_title}*")
                render_movie_grid(
                    movies=recs_df.to_dict(orient="records"),
                    num_cols=5,
                    show_explain=True,
                    explain_engine=explain_engine,
                    source_movie_id=source_id,
                )

    # =========================================================================
    # TAB 2: BUILD YOUR TASTE PROFILE (Cold-Start Guest Experience)
    # =========================================================================
    elif mode == "🔮 Build Your Taste Profile":
        st.markdown("### 🔮 Instant Taste Profile Generator (Guest Cold-Start)")
        st.caption(
            "Select 2–6 movies you love across different genres. Our AI computes your Centroid Taste Vector and diversifies recommendations with MMR."
        )

        curated_favorites = [
            "Inception",
            "Pulp Fiction",
            "The Dark Knight",
            "The Matrix",
            "Interstellar",
            "Fight Club",
            "The Godfather",
            "Spirited Away",
            "Toy Story",
            "The Shawshank Redemption",
            "GoodFellas",
            "Jurassic Park",
            "Gladiator",
            "Memento",
            "The Prestige",
        ]
        available_curated = [t for t in curated_favorites if t in movies_df["title"].values]

        selected_titles = st.multiselect(
            "Select Your Favorite Movies:",
            options=sorted(movies_df["title"].unique()),
            default=available_curated[:3] if len(available_curated) >= 3 else None,
        )

        col_mmr, col_k = st.columns([3, 1])
        with col_mmr:
            mmr_lambda = st.slider(
                "Relevance vs. Diversity Balance (MMR λ):",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher λ prioritizes pure relevance; lower λ promotes serendipity and multi-genre diversity.",
            )
        with col_k:
            top_k = st.slider("Results Count:", 5, 20, 10, step=5)

        if st.button(
            "🔮 Generate My Personalized Taste Recommendations",
            type="primary",
            use_container_width=True,
        ):
            if not selected_titles:
                st.warning("Please select at least 1 favorite movie to generate recommendations.")
                return

            with st.spinner("Synthesizing multi-vector centroid in latent semantic space..."):
                fav_rows = movies_df[movies_df["title"].isin(selected_titles)]
                fav_ids = fav_rows["movie_id"].tolist()

                hybrid_model: HybridRecommender = resources["hybrid"]
                hybrid_model.hybrid_cfg.mmr_lambda = mmr_lambda
                recs_df = hybrid_model.recommend(
                    favorite_movie_ids=fav_ids, top_k=top_k, apply_mmr=True
                )

                st.markdown(
                    f"#### 🎯 Recommended For Your Taste ({len(selected_titles)} Movies Selected)"
                )
                render_movie_grid(
                    movies=recs_df.to_dict(orient="records"),
                    num_cols=5,
                    show_explain=True,
                    explain_engine=explain_engine,
                    source_movie_id=fav_ids[0] if fav_ids else None,
                )

    # =========================================================================
    # TAB 3: PERSONALIZED USER FEED (User ID Deep Dive)
    # =========================================================================
    elif mode == "👤 Personalized User Feed":
        st.markdown("### 👤 Deep-Dive Personalized User Experience")
        st.caption(
            "Inspect a registered user's historical ratings and generate tailored recommendations using Collaborative & Hybrid models."
        )

        unique_users = sorted(ratings_df["user_id"].unique())
        col_user, col_algo, col_k = st.columns([2, 2, 1])

        with col_user:
            user_id = st.selectbox("Select User Profile ID:", options=unique_users, index=0)

        with col_algo:
            model_type = st.selectbox(
                "Recommendation Algorithm:",
                [
                    "Two-Stage Hybrid (+ MMR)",
                    "SVD Matrix Factorization",
                    "PyTorch Neural CF (NeuMF)",
                ],
            )

        with col_k:
            top_k = st.slider("Count:", 5, 20, 10, step=5)

        user_ratings = ratings_df[ratings_df["user_id"] == user_id].sort_values(
            by="rating", ascending=False
        )
        top_user_history = user_ratings.merge(
            movies_df[
                ["movie_id", "title", "release_year", "genres_str", "vote_average", "poster_path"]
            ],
            on="movie_id",
        ).head(5)

        with st.expander(
            f"📜 View User #{user_id}'s Top Rated History ({len(user_ratings)} Total Ratings)",
            expanded=False,
        ):
            render_movie_grid(movies=top_user_history.to_dict(orient="records"), num_cols=5)

        if st.button("🚀 Generate Personalized Feed", type="primary", use_container_width=True):
            with st.spinner("Scoring candidate catalog with collaborative latent vectors..."):
                if "Hybrid" in model_type:
                    recs_df = resources["hybrid"].recommend(
                        user_id=user_id, top_k=top_k, apply_mmr=True
                    )
                elif "SVD" in model_type and "svd" in resources:
                    recs_df = resources["svd"].recommend(
                        query=user_id, top_k=top_k, exclude_rated=True
                    )
                elif "Neural" in model_type and "neural_cf" in resources:
                    recs_df = resources["neural_cf"].recommend(
                        query=user_id, top_k=top_k, exclude_rated=True
                    )
                else:
                    st.error("Selected model is not loaded.")
                    return

                st.markdown(f"#### 🌟 Personalized Recommendations for User #{user_id}")
                render_movie_grid(movies=recs_df.to_dict(orient="records"), num_cols=5)

    # =========================================================================
    # TAB 4: BLOCKBUSTERS & HIDDEN GEMS (Demographic Explorer)
    # =========================================================================
    elif mode == "🏆 Blockbusters & Hidden Gems":
        st.markdown("### 🏆 Demographic Explorer & Bayesian Leaderboard")
        st.caption(
            "Explore critically acclaimed cinema sorted by IMDb Bayesian Weighted Rating (WR)."
        )

        all_genres = sorted(
            {g for row in movies_df["genres_str"].dropna() for g in str(row).split()}
        )
        genre_options = ["All Genres"] + [g.replace("_", " ").title() for g in all_genres]

        col_genre, col_year, col_k = st.columns([2, 2, 1])

        with col_genre:
            selected_genre = st.selectbox("Genre Filter:", genre_options)
        with col_year:
            min_year = st.slider("Released On or After Year:", 1930, 2020, 2000, step=5)
        with col_k:
            top_k = st.slider("Display Limit:", 5, 25, 10, step=5)

        genre_arg = (
            None if selected_genre == "All Genres" else selected_genre.lower().replace(" ", "_")
        )
        pop_model: PopularityRecommender = resources.get("popularity")

        if pop_model is not None:
            recs_df = pop_model.recommend(genre=genre_arg, min_year=min_year, top_k=top_k)
            st.markdown(f"#### 🏅 Top Rated {selected_genre} Films ({min_year}+)")
            render_movie_grid(movies=recs_df.to_dict(orient="records"), num_cols=5)

    # =========================================================================
    # TAB 5: AI THEMATIC & PLOT SEARCH (Natural Language Semantic Search)
    # =========================================================================
    elif mode == "🔍 AI Thematic & Plot Search":
        st.markdown("### 🔍 AI Natural Language & Thematic Plot Search")
        st.caption(
            "Type any descriptive narrative, aesthetic, or plot idea in plain English. Powered by Sentence-Transformers and FAISS."
        )

        default_prompts = [
            "mind-bending psychological thriller about dreams, memory loss, and reality",
            "heartwarming animated adventure about friendly ocean animals",
            "dark gritty crime noir detective solving a serial killer case in the rain",
            "epic space exploration through wormholes and time dilation",
            "historical martial arts epic with ancient dynasty politics",
        ]

        query_text = st.text_area(
            "Enter Plot, Theme, or Atmosphere Description:",
            value=default_prompts[0],
            height=90,
        )

        top_k = st.slider("Matches Count:", 5, 20, 10, step=5)

        if st.button(
            "🔍 Search Semantic Embedding Space", type="primary", use_container_width=True
        ):
            if not query_text.strip():
                st.warning("Please enter a search prompt.")
                return

            with st.spinner(
                "Computing 384-dimensional query vector and performing FAISS IndexFlatIP search..."
            ):
                semantic_model: SemanticVectorRecommender = resources.get("semantic")
                if semantic_model:
                    recs_df = semantic_model.recommend(query=query_text, top_k=top_k)
                    st.markdown("#### 🎯 Nearest Semantic Vector Matches")
                    render_movie_grid(
                        movies=recs_df.to_dict(orient="records"),
                        num_cols=5,
                        show_explain=True,
                        explain_engine=explain_engine,
                        query_text=query_text,
                    )


if __name__ == "__main__":
    main()
