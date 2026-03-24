from __future__ import annotations

import argparse
import json
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from .model import GenreRecommender, load_artifacts


def _safe_precision(hits: int, k: int, total_queries: int) -> float:
    if total_queries == 0 or k <= 0:
        return 0.0
    return hits / float(total_queries * k)


def evaluate_content_recommender(
    project_root: Path,
    top_k: int = 10,
    min_user_interactions: int = 5,
    report_name: str = "evaluation_report.json",
) -> dict[str, float | int]:
    data_dir = project_root / "data" / "ml - 1m"
    artifacts_dir = project_root / "artifacts"

    ratings = pd.read_csv(data_dir / "ratings.csv")

    similarity, model_df = load_artifacts(artifacts_dir)
    recommender = GenreRecommender(similarity_matrix=similarity, df=model_df)
    movie_id_to_title = (
        model_df[["movie_id", "title"]]
        .drop_duplicates(subset=["movie_id"])
        .set_index("movie_id")["title"]
        .to_dict()
    )

    ratings = ratings.sort_values(["userId", "timestamp"])
    user_groups = ratings.groupby("userId")

    train_popularity = ratings.groupby("movieId").size()
    popularity_total = float(train_popularity.sum()) if len(train_popularity) else 1.0

    total_queries = 0
    hits = 0
    recommended_movie_ids: set[int] = set()
    novelty_scores: list[float] = []

    for _, group in user_groups:
        if len(group) < min_user_interactions:
            continue

        history = group.iloc[:-1]
        target = group.iloc[-1]
        if history.empty:
            continue

        query_row = history.iloc[-1]
        query_title = movie_id_to_title.get(int(query_row["movieId"]))
        target_movie_id = int(target["movieId"])

        if not query_title:
            continue

        try:
            recs = recommender.get_recommendations(str(query_title), top_n=top_k)
        except ValueError:
            continue

        total_queries += 1
        rec_movie_ids = recs["movie_id"].astype(int).tolist()

        if target_movie_id in rec_movie_ids:
            hits += 1

        for movie_id in rec_movie_ids:
            recommended_movie_ids.add(movie_id)
            movie_count = int(train_popularity.get(movie_id, 1))
            prob = max(movie_count / popularity_total, 1e-12)
            novelty_scores.append(-log2(prob))

    coverage = (
        len(recommended_movie_ids) / float(model_df["movie_id"].nunique())
        if len(model_df)
        else 0.0
    )
    hit_rate_at_k = (hits / total_queries) if total_queries else 0.0
    precision_at_k = _safe_precision(hits=hits, k=top_k, total_queries=total_queries)
    recall_at_k = hit_rate_at_k
    novelty = float(np.mean(novelty_scores)) if novelty_scores else 0.0

    metrics: dict[str, float | int] = {
        "queries_evaluated": total_queries,
        "top_k": top_k,
        "min_user_interactions": min_user_interactions,
        "hit_rate_at_k": round(hit_rate_at_k, 4),
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "coverage": round(coverage, 4),
        "novelty": round(novelty, 4),
    }

    report_path = artifacts_dir / report_name
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate movie recommender quality.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics")
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Minimum interactions required for a user to be included",
    )
    args = parser.parse_args()

    metrics = evaluate_content_recommender(
        project_root=args.project_root,
        top_k=args.top_k,
        min_user_interactions=args.min_user_interactions,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
