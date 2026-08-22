"""Temporal train/test evaluation split and ground-truth dataset creation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("recsys.evaluation.split")


@dataclass
class EvalDataset:
    """Evaluation dataset container."""

    user_histories: dict[int, list[int]]
    ground_truth: dict[int, list[int]]
    item_popularity_prob: dict[int, float]
    total_catalog_size: int
    eval_user_ids: list[int]


def create_evaluation_split(
    ratings_df: pd.DataFrame,
    total_catalog_size: int,
    leave_k: int = 2,
    min_interactions: int = 5,
    positive_threshold: float = 3.5,
) -> EvalDataset:
    """Create temporal leave-k split for offline ranking evaluation."""
    sorted_ratings = ratings_df.sort_values(by=["user_id", "timestamp"]).copy()

    # Calculate item popularity probabilities: p(i) = count(i) / total_interactions
    pop_counts = sorted_ratings["movie_id"].value_counts()
    total_ratings_count = float(len(sorted_ratings))
    item_popularity_prob = (pop_counts / total_ratings_count).to_dict()

    user_histories: dict[int, list[int]] = {}
    ground_truth: dict[int, list[int]] = {}
    eval_user_ids: list[int] = []

    for user_id, group in sorted_ratings.groupby("user_id"):
        if len(group) < min_interactions:
            continue

        # Split: history (all except last k) and test (last k)
        history_df = group.iloc[:-leave_k]
        test_df = group.iloc[-leave_k:]

        # Filter ground-truth to positive interactions (rating >= threshold)
        relevant_test = test_df[test_df["rating"] >= positive_threshold]["movie_id"].tolist()

        if relevant_test and not history_df.empty:
            user_histories[int(user_id)] = history_df["movie_id"].tolist()
            ground_truth[int(user_id)] = relevant_test
            eval_user_ids.append(int(user_id))

    logger.info(
        f"Evaluation split created: {len(eval_user_ids)} users qualified "
        f"(>= {min_interactions} interactions, >= 1 positive ground-truth test item)."
    )

    return EvalDataset(
        user_histories=user_histories,
        ground_truth=ground_truth,
        item_popularity_prob=item_popularity_prob,
        total_catalog_size=total_catalog_size,
        eval_user_ids=eval_user_ids,
    )
