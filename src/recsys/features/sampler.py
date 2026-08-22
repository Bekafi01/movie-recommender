"""Negative sampling utilities for implicit feedback training in Neural Collaborative Filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.logger import get_logger

logger = get_logger("recsys.features.sampler")


class NegativeSampler:
    """Generates negative user-item pairs (unobserved interactions) for implicit feedback training."""

    def __init__(
        self,
        num_negatives: int = 4,
        positive_threshold: float = 3.5,
        random_seed: int = 42,
    ):
        self.num_negatives = num_negatives
        self.positive_threshold = positive_threshold
        self.rng = np.random.default_rng(random_seed)

    def sample(
        self,
        ratings_df: pd.DataFrame,
        user_to_idx: dict[int, int],
        movie_to_idx: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate balanced user-item interaction pairs with negative sampling."""
        num_movies = len(movie_to_idx)
        all_movie_indices = np.arange(num_movies)

        # Find positive interactions
        positives = ratings_df[ratings_df["rating"] >= self.positive_threshold].copy()
        user_positive_items: dict[int, set[int]] = (
            ratings_df.groupby("user_id")["movie_id"]
            .apply(lambda ids: {movie_to_idx[m] for m in ids if m in movie_to_idx})
            .to_dict()
        )

        user_list: list[int] = []
        item_list: list[int] = []
        label_list: list[float] = []

        for _, row in positives.iterrows():
            u_id = int(row["user_id"])
            m_id = int(row["movie_id"])

            if u_id not in user_to_idx or m_id not in movie_to_idx:
                continue

            u_idx = user_to_idx[u_id]
            m_idx = movie_to_idx[m_id]

            # Positive sample (y = 1)
            user_list.append(u_idx)
            item_list.append(m_idx)
            label_list.append(1.0)

            # Sample N negative items (y = 0)
            seen_items = user_positive_items.get(u_id, set())
            neg_count = 0
            # Sample from candidates
            sampled_negatives = self.rng.choice(
                all_movie_indices, size=self.num_negatives * 3, replace=True
            )
            for neg_item in sampled_negatives:
                if neg_item not in seen_items:
                    user_list.append(u_idx)
                    item_list.append(int(neg_item))
                    label_list.append(0.0)
                    neg_count += 1
                    if neg_count >= self.num_negatives:
                        break

        logger.info(
            f"Negative sampling complete: {len(user_list):,} total samples "
            f"({sum(label_list):,} positive, {len(label_list) - sum(label_list):,} negative)."
        )
        return (
            np.array(user_list, dtype=np.int64),
            np.array(item_list, dtype=np.int64),
            np.array(label_list, dtype=np.float32),
        )


class NCFDataset(Dataset):
    """PyTorch Dataset for user-item interaction pairs."""

    def __init__(self, users: np.ndarray, items: np.ndarray, labels: np.ndarray):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.labels[idx]
