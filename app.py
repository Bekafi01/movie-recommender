import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

@st.cache_resource
def load_model():
    """Load model artifacts"""
    model_dir = Path(__file__).parent / "artifacts"
    
    with open(model_dir / "similarity_matrix.npz", "rb") as f:
        similarity = np.load(f)['similarity']
    
    with open(model_dir / "movies_data.pkl", "rb") as f:
        df = pickle.load(f)
    
    return similarity, df

def get_recommendations(title, similarity, df, top_n=5):
    """Get similar movies"""
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    results = []
    for i, score in sim_scores:
        row = df.iloc[i]
        results.append({
            'title': row['title'],
            'genres': ', '.join(row['genres']),
            'rating': row['rating'],
            'similarity': f"{score:.1%}"
        })
    return results

# Load data
try:
    similarity, df = load_model()
    movies = df['title'].tolist()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# UI
st.title("🎬 Movie Recommender")
st.markdown("Find similar movies based on genres")

selected = st.selectbox("Pick a movie you like:", movies)

if st.button("Recommend", type="primary"):
    with st.spinner("Finding matches..."):
        recs = get_recommendations(selected, similarity, df)
        
        st.subheader(f"Because you watched **{selected}**:")
        
        for i, rec in enumerate(recs, 1):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {rec['title']}**")
                    st.caption(f"Genres: {rec['genres']}")
                with col2:
                    st.write(f"⭐ {rec['rating']}")
                    st.caption(f"Match: {rec['similarity']}")
                st.divider()