"""Unit tests for PyTorch Neural Collaborative Filtering (NeuMF) architecture and recommender wrapper."""

from pathlib import Path

import pandas as pd
import torch

from recsys.models.neural_cf import NeuMFNet, NeuralCollaborativeRecommender


def test_neumf_network_architecture() -> None:
    """Test NeuMFNet forward pass output shape and value properties."""
    num_users = 20
    num_items = 50
    model = NeuMFNet(
        num_users=num_users,
        num_items=num_items,
        latent_dim_gmf=16,
        latent_dim_mlp=16,
        mlp_layers=[32, 16],
        dropout=0.1,
    )

    batch_users = torch.tensor([0, 5, 12, 19], dtype=torch.long)
    batch_items = torch.tensor([10, 20, 30, 45], dtype=torch.long)

    logits = model(batch_users, batch_items)
    assert logits.shape == (4,)
    assert not torch.isnan(logits).any()

    # Test sigmoid prediction probability
    probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_neural_collaborative_recommender(tmp_path: Path) -> None:
    """Test fitting, scoring, saving, and loading NeuralCollaborativeRecommender."""
    movies_df = pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4],
            "tmdb_id": [101, 102, 103, 104],
            "title": ["Inception", "Interstellar", "Dark Knight", "Prestige"],
            "release_year": [2010, 2014, 2008, 2006],
            "genres_str": ["sci-fi", "sci-fi", "action", "drama"],
        }
    )

    ratings_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "movie_id": [1, 2, 1, 3, 2, 4],
            "rating": [5.0, 4.5, 4.0, 5.0, 4.5, 4.0],
            "timestamp": [100, 200, 300, 400, 500, 600],
        }
    )

    ncf = NeuralCollaborativeRecommender()
    ncf.fit(ratings_df=ratings_df, movies_df=movies_df, epochs=2, batch_size=4)

    recs = ncf.recommend(query=1, top_k=2, exclude_rated=True)
    assert len(recs) == 2
    assert 1 not in recs["movie_id"].values
    assert 2 not in recs["movie_id"].values

    # Test save and load
    save_path = tmp_path / "neumf_model.pt"
    ncf.save(save_path)

    loaded = NeuralCollaborativeRecommender.load(save_path)
    loaded_recs = loaded.recommend(query=1, top_k=2, exclude_rated=True)
    assert len(loaded_recs) == 2
