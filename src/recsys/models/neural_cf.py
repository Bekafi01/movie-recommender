"""Neural Collaborative Filtering (NeuMF) architecture in PyTorch."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import AppConfig, load_config
from ..features.sampler import NCFDataset, NegativeSampler
from ..utils.exceptions import UserNotFoundError
from ..utils.logger import get_logger
from ..utils.timer import timed
from .base import BaseRecommender

logger = get_logger("recsys.models.neural_cf")


class NeuMFNet(nn.Module):
    """Neural Matrix Factorization architecture combining GMF and Deep MLP."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        latent_dim_gmf: int = 32,
        latent_dim_mlp: int = 32,
        mlp_layers: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [64, 32, 16]

        # 1. Generalized Matrix Factorization (GMF) Embeddings
        self.user_embed_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embed_gmf = nn.Embedding(num_items, latent_dim_gmf)

        # 2. Multi-Layer Perceptron (MLP) Embeddings
        self.user_embed_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embed_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # 3. MLP Dense Layers
        mlp_modules: list[nn.Module] = []
        in_dim = latent_dim_mlp * 2
        for out_dim in mlp_layers:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        self.mlp_pipeline = nn.Sequential(*mlp_modules)

        # 4. Final Prediction / Fusion Layer
        final_in_dim = latent_dim_gmf + (mlp_layers[-1] if mlp_layers else in_dim)
        self.prediction_layer = nn.Linear(final_in_dim, 1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        # GMF Branch
        gmf_u = self.user_embed_gmf(user_indices)
        gmf_i = self.item_embed_gmf(item_indices)
        gmf_out = gmf_u * gmf_i

        # MLP Branch
        mlp_u = self.user_embed_mlp(user_indices)
        mlp_i = self.item_embed_mlp(item_indices)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=-1)
        mlp_out = self.mlp_pipeline(mlp_in)

        # Fusion
        fusion = torch.cat([gmf_out, mlp_out], dim=-1)
        logits = self.prediction_layer(fusion)
        return logits.squeeze(-1)


class NeuralCollaborativeRecommender(BaseRecommender):
    """Wrapper and trainer for PyTorch Neural Collaborative Filtering model."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.ncf_cfg = self.config.collaborative.neural_cf

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: NeuMFNet | None = None
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.user_ids: list[int] = []
        self.movie_ids: list[int] = []
        self.user_to_idx: dict[int, int] = {}
        self.movie_to_idx: dict[int, int] = {}
        self.user_rated_items: dict[int, set[int]] = {}
        self._fitted = False

    @timed("Training PyTorch Neural Collaborative Filtering (NeuMF)")
    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        epochs: int | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> NeuralCollaborativeRecommender:
        """Train NeuMF with negative sampling and BCEWithLogitsLoss."""
        self.movies_df = movies_df.copy()
        num_epochs = epochs or self.ncf_cfg.epochs
        b_size = batch_size or self.ncf_cfg.batch_size

        self.user_ids = sorted(ratings_df["user_id"].unique())
        self.movie_ids = sorted(movies_df["movie_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.movie_to_idx = {m: i for i, m in enumerate(self.movie_ids)}

        self.user_rated_items = ratings_df.groupby("user_id")["movie_id"].apply(set).to_dict()

        # 1. Negative Sampling
        sampler = NegativeSampler(
            num_negatives=self.ncf_cfg.negative_samples_ratio,
            positive_threshold=self.ncf_cfg.positive_rating_threshold,
            random_seed=self.config.project.random_seed,
        )
        users, items, labels = sampler.sample(ratings_df, self.user_to_idx, self.movie_to_idx)

        dataset = NCFDataset(users, items, labels)
        loader = DataLoader(dataset, batch_size=b_size, shuffle=True)

        # 2. Build Model
        self.model = NeuMFNet(
            num_users=len(self.user_ids),
            num_items=len(self.movie_ids),
            latent_dim_gmf=self.ncf_cfg.latent_dim_gmf,
            latent_dim_mlp=self.ncf_cfg.latent_dim_mlp,
            mlp_layers=self.ncf_cfg.mlp_layers,
            dropout=self.ncf_cfg.dropout,
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.ncf_cfg.learning_rate,
            weight_decay=self.ncf_cfg.weight_decay,
        )

        # 3. Training Loop
        self.model.train()
        for epoch in range(1, num_epochs + 1):
            total_loss = 0.0
            for batch_users, batch_items, batch_labels in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_users, batch_items)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / len(loader)
            if epoch % 2 == 0 or epoch == num_epochs:
                logger.info(f"NeuMF Epoch [{epoch}/{num_epochs}] - Loss: {avg_loss:.4f}")

        self._fitted = True
        return self

    def recommend(
        self,
        query: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Score candidate movies for a user and return Top-K recommendations."""
        if not self._fitted or self.model is None:
            raise RuntimeError("NeuralCollaborativeRecommender must be fitted before recommend().")

        user_id = int(query)
        if user_id not in self.user_to_idx:
            raise UserNotFoundError(user_id)

        u_idx = self.user_to_idx[user_id]
        all_item_indices = torch.arange(len(self.movie_ids), dtype=torch.long, device=self.device)
        u_tensor = torch.full((len(self.movie_ids),), u_idx, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(u_tensor, all_item_indices)
            scores = torch.sigmoid(logits).cpu().numpy()

        # Mask seen items
        if exclude_rated and user_id in self.user_rated_items:
            for rated_m in self.user_rated_items[user_id]:
                if rated_m in self.movie_to_idx:
                    scores[self.movie_to_idx[rated_m]] = -1.0

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        recommended_movie_ids = [self.movie_ids[i] for i in top_indices]

        recs_df = (
            self.movies_df[self.movies_df["movie_id"].isin(recommended_movie_ids)]
            .set_index("movie_id")
            .loc[recommended_movie_ids]
            .reset_index()
        )
        recs_df["rank"] = range(1, len(recs_df) + 1)
        recs_df["score"] = np.round(top_scores, 4)

        output_cols = [
            "rank",
            "movie_id",
            "tmdb_id",
            "title",
            "release_year",
            "genres_str",
            "vote_average",
            "score",
            "poster_path",
        ]
        available_cols = [c for c in output_cols if c in recs_df.columns]
        return recs_df[available_cols]

    def save(self, path: Path) -> None:
        """Save PyTorch weights and metadata."""
        path.parent.mkdir(parents=True, exist_ok=True)
        meta_path = path.parent / "neumf_meta.pkl"

        if self.model is not None:
            torch.save(self.model.state_dict(), path)

        with open(meta_path, "wb") as f:
            pickle.dump(
                {
                    "movies_df": self.movies_df,
                    "user_ids": self.user_ids,
                    "movie_ids": self.movie_ids,
                    "user_to_idx": self.user_to_idx,
                    "movie_to_idx": self.movie_to_idx,
                    "user_rated_items": self.user_rated_items,
                    "_fitted": self._fitted,
                },
                f,
            )
        logger.info(f"Saved NeuralCollaborativeRecommender to {path}")

    @classmethod
    def load(
        cls, path: Path, config: AppConfig | None = None, **kwargs: Any
    ) -> NeuralCollaborativeRecommender:
        """Load PyTorch model and metadata."""
        cfg = config or load_config()
        instance = cls(config=cfg)
        meta_path = path.parent / "neumf_meta.pkl"

        with open(meta_path, "rb") as f:
            data = pickle.load(f)
        instance.movies_df = data["movies_df"]
        instance.user_ids = data["user_ids"]
        instance.movie_ids = data["movie_ids"]
        instance.user_to_idx = data["user_to_idx"]
        instance.movie_to_idx = data["movie_to_idx"]
        instance.user_rated_items = data["user_rated_items"]
        instance._fitted = data["_fitted"]

        # Rebuild architecture and load weights
        instance.model = NeuMFNet(
            num_users=len(instance.user_ids),
            num_items=len(instance.movie_ids),
            latent_dim_gmf=cfg.collaborative.neural_cf.latent_dim_gmf,
            latent_dim_mlp=cfg.collaborative.neural_cf.latent_dim_mlp,
            mlp_layers=cfg.collaborative.neural_cf.mlp_layers,
            dropout=cfg.collaborative.neural_cf.dropout,
        ).to(instance.device)

        if path.exists():
            instance.model.load_state_dict(
                torch.load(path, map_location=instance.device, weights_only=True)
            )
            instance.model.eval()

        logger.info(f"Loaded NeuralCollaborativeRecommender from {path}")
        return instance
