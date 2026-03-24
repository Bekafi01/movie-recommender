import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from movie_recommender.model import GenreRecommender, load_artifacts

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")


def get_tmdb_api_key() -> str | None:
    api_key = st.secrets.get("TMDB_API_KEY") if hasattr(st, "secrets") else None
    if api_key:
        return str(api_key)
    return None


@st.cache_data
def load_tmdb_mapping() -> dict[int, int]:
    links_path = PROJECT_ROOT / "data" / "ml - 1m" / "links.csv"
    if not links_path.exists():
        return {}

    links_df = pd.read_csv(links_path)
    links_df = links_df.dropna(subset=["movieId", "tmdbId"])

    movie_to_tmdb: dict[int, int] = {}
    for _, row in links_df.iterrows():
        movie_to_tmdb[int(row["movieId"])] = int(row["tmdbId"])
    return movie_to_tmdb


@st.cache_data(ttl=60 * 60 * 24)
def fetch_poster_url(tmdb_id: int, api_key: str) -> str | None:
    endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
    try:
        response = requests.get(endpoint, params={"api_key": api_key}, timeout=8)
        if response.status_code != 200:
            return None

        poster_path = response.json().get("poster_path")
        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except requests.RequestException:
        return None

@st.cache_resource
def load_model():
    """Load model artifacts"""
    model_dir = PROJECT_ROOT / "artifacts"
    similarity, df = load_artifacts(model_dir)
    return GenreRecommender(similarity_matrix=similarity, df=df), df

# Load data
try:
    recommender, df = load_model()
    movies = df['title'].tolist()
    movie_to_tmdb = load_tmdb_mapping()
    tmdb_api_key = get_tmdb_api_key()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# UI
st.title("🎬 Movie Recommender")
st.markdown("Find similar movies based on genres")

selected = st.selectbox("Pick a movie you like:", movies)

if st.button("Recommend", type="primary"):
    with st.spinner("Finding matches..."):
        recs_df = recommender.get_recommendations(selected, top_n=5)
        
        st.subheader(f"Because you watched **{selected}**:")

        if not tmdb_api_key:
            st.caption("Poster preview is disabled. Set `TMDB_API_KEY` in Streamlit secrets to enable posters.")
        
        for _, rec in recs_df.iterrows():
            with st.container():
                col_poster, col_details, col_stats = st.columns([1.3, 3, 1])

                with col_poster:
                    poster_url = None
                    if tmdb_api_key:
                        tmdb_id = movie_to_tmdb.get(int(rec["movie_id"]))
                        if tmdb_id:
                            poster_url = fetch_poster_url(tmdb_id=tmdb_id, api_key=tmdb_api_key)

                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.caption("No poster")

                with col_details:
                    genres = rec["genres"]
                    if isinstance(genres, list):
                        genres = ", ".join(genres)
                    st.write(f"**{int(rec['rank'])}. {rec['title']}**")
                    st.caption(f"Genres: {genres}")

                with col_stats:
                    st.write(f"⭐ {rec['rating']}")
                    st.caption(f"Match: {rec['similarity_score']:.1%}")
                st.divider()
