"""Automated offline benchmarking suite comparing multi-paradigm recommendation models."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig, load_config
from ..data.db import DataRepository
from ..models.collaborative import SVDCollaborativeRecommender
from ..models.content_based import SemanticVectorRecommender, TFIDFRecommender
from ..models.hybrid import HybridRecommender
from ..models.neural_cf import NeuralCollaborativeRecommender
from ..models.popularity import PopularityRecommender
from ..utils.logger import get_logger
from ..utils.timer import timed
from .diversity import catalog_coverage, mean_intra_list_diversity, novelty_at_k
from .metrics import (
    average_precision_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .split import EvalDataset, create_evaluation_split

logger = get_logger("recsys.evaluation.benchmark")


class BenchmarkRunner:
    """Orchestrates comprehensive scientific evaluation of all recommendation engines."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.eval_cfg = self.config.evaluation
        self.repo = DataRepository(config=self.config)

    @timed("Executing Multi-Model Scientific Benchmark")
    def run_benchmark(
        self,
        top_k_list: list[int] | None = None,
        save_report: bool = True,
    ) -> pd.DataFrame:
        """Run evaluation across all models and return benchmark comparison DataFrame."""
        k_values = top_k_list or self.eval_cfg.top_k_list
        max_k = max(k_values)

        # 1. Load data and create evaluation split
        logger.info("Loading processed datasets for benchmarking...")
        movies_df = self.repo.load_movies()
        ratings_df = self.repo.load_ratings()

        eval_dataset = create_evaluation_split(
            ratings_df=ratings_df,
            total_catalog_size=len(movies_df),
            leave_k=self.eval_cfg.test_ratio_leave_k,
            min_interactions=self.eval_cfg.min_user_interactions,
            positive_threshold=self.eval_cfg.positive_rating_threshold,
        )

        # Load models
        models = self._load_models(movies_df=movies_df)

        # Load embeddings for diversity computation
        embeddings = models["semantic"].pipeline.embeddings
        id_to_idx = models["semantic"].id_to_idx

        # 2. Evaluate each model
        all_results: list[dict[str, Any]] = []

        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name.upper()}...")
            model_metrics = self._evaluate_single_model(
                model_name=model_name,
                model=model,
                eval_dataset=eval_dataset,
                k_values=k_values,
                max_k=max_k,
                embeddings=embeddings,
                id_to_idx=id_to_idx,
            )
            all_results.extend(model_metrics)

        results_df = pd.DataFrame(all_results)

        # 3. Save reports
        if save_report:
            self._save_benchmark_artifacts(results_df, eval_dataset)

        return results_df

    def _load_models(self, movies_df: pd.DataFrame) -> dict[str, Any]:
        """Load fitted models from artifacts."""
        cfg = self.config
        models: dict[str, Any] = {}

        # 1. Popularity
        pop_path = cfg.paths.models_path / "popularity_model.pkl"
        if pop_path.exists():
            models["popularity"] = PopularityRecommender.load(pop_path, config=cfg)

        # 2. TF-IDF
        tfidf_path = cfg.paths.models_path / "tfidf_model.pkl"
        if tfidf_path.exists():
            models["tfidf"] = TFIDFRecommender.load(tfidf_path, config=cfg)

        # 3. Semantic FAISS
        sem_path = cfg.paths.embeddings_path / "semantic_meta.pkl"
        if sem_path.exists():
            models["semantic"] = SemanticVectorRecommender.load(sem_path, config=cfg)

        # 4. SVD
        svd_path = cfg.paths.models_path / "svd_model.pkl"
        if svd_path.exists():
            models["svd"] = SVDCollaborativeRecommender.load(svd_path, config=cfg)

        # 5. Neural CF
        ncf_path = cfg.paths.models_path / "neumf_model.pt"
        if ncf_path.exists():
            models["neural_cf"] = NeuralCollaborativeRecommender.load(ncf_path, config=cfg)

        # 6. Hybrid Recommender
        models["hybrid"] = HybridRecommender(
            content_model=models.get("semantic"),
            collab_model=models.get("svd"),
            popularity_model=models.get("popularity"),
            config=cfg,
        )
        models["hybrid"].movies_df = movies_df
        models["hybrid"]._fitted = True

        return models

    def _evaluate_single_model(
        self,
        model_name: str,
        model: Any,
        eval_dataset: EvalDataset,
        k_values: list[int],
        max_k: int,
        embeddings: np.ndarray | None,
        id_to_idx: dict[int, int],
    ) -> list[dict[str, Any]]:
        """Compute metrics for a single model across users."""
        user_recommendations: dict[int, list[int]] = {}

        # Generate top max_k recommendations for each evaluation user
        for user_id in eval_dataset.eval_user_ids:
            history = eval_dataset.user_histories[user_id]
            history_set = set(history)
            recs_list: list[int] = []

            try:
                if model_name in ["svd", "neural_cf"]:
                    df_recs = model.recommend(
                        query=user_id, top_k=max_k, exclude_item_ids=history_set
                    )
                    recs_list = df_recs["movie_id"].tolist()
                elif model_name == "hybrid":
                    df_recs = model.recommend(
                        user_id=user_id, top_k=max_k, apply_mmr=True, exclude_item_ids=history_set
                    )
                    recs_list = df_recs["movie_id"].tolist()
                elif model_name in ["tfidf", "semantic"]:
                    # For content models, use user's most recent interaction as query
                    last_liked_movie = history[-1]
                    df_recs = model.recommend(
                        query=last_liked_movie, top_k=max_k, exclude_self=True
                    )
                    # Exclude seen history except candidate itself
                    recs_list = [m for m in df_recs["movie_id"].tolist() if m not in history_set][
                        :max_k
                    ]
                elif model_name == "popularity":
                    df_recs = model.recommend(top_k=max_k * 3)
                    recs_list = [m for m in df_recs["movie_id"].tolist() if m not in history_set][
                        :max_k
                    ]
            except Exception as e:
                logger.debug(f"Failed to generate recs for user {user_id} on {model_name}: {e}")
                recs_list = []

            user_recommendations[user_id] = recs_list

        # Compute metrics per K
        all_recs_lists = list(user_recommendations.values())
        results_per_k: list[dict[str, Any]] = []

        for k in k_values:
            ndcg_scores: list[float] = []
            map_scores: list[float] = []
            recall_scores: list[float] = []
            precision_scores: list[float] = []
            hit_scores: list[float] = []
            mrr_scores: list[float] = []

            for user_id in eval_dataset.eval_user_ids:
                actual = eval_dataset.ground_truth[user_id]
                pred = user_recommendations.get(user_id, [])

                ndcg_scores.append(ndcg_at_k(actual, pred, k=k))
                map_scores.append(average_precision_at_k(actual, pred, k=k))
                recall_scores.append(recall_at_k(actual, pred, k=k))
                precision_scores.append(precision_at_k(actual, pred, k=k))
                hit_scores.append(hit_rate_at_k(actual, pred, k=k))
                mrr_scores.append(mrr_at_k(actual, pred, k=k))

            # Beyond-accuracy metrics
            cov = catalog_coverage([r[:k] for r in all_recs_lists], eval_dataset.total_catalog_size)
            nov = novelty_at_k(all_recs_lists, eval_dataset.item_popularity_prob, k=k)
            ild = (
                mean_intra_list_diversity(all_recs_lists, embeddings, id_to_idx, k=k)
                if embeddings is not None
                else 0.0
            )

            row: dict[str, Any] = {
                "model": model_name.upper(),
                "top_k": k,
                "ndcg@k": round(float(np.mean(ndcg_scores)), 4),
                "map@k": round(float(np.mean(map_scores)), 4),
                "recall@k": round(float(np.mean(recall_scores)), 4),
                "precision@k": round(float(np.mean(precision_scores)), 4),
                "hit_rate@k": round(float(np.mean(hit_scores)), 4),
                "mrr@k": round(float(np.mean(mrr_scores)), 4),
                "coverage": round(cov, 4),
                "novelty": round(nov, 2),
                "diversity": round(ild, 4),
            }
            results_per_k.append(row)

        return results_per_k

    def _save_benchmark_artifacts(
        self, results_df: pd.DataFrame, eval_dataset: EvalDataset
    ) -> None:
        """Persist structured benchmark results to JSON and Markdown."""
        benchmarks_dir = self.config.paths.benchmarks_path
        benchmarks_dir.mkdir(parents=True, exist_ok=True)

        json_path = benchmarks_dir / "evaluation_report.json"
        md_path = benchmarks_dir / "benchmark_summary.md"

        # 1. Save JSON
        payload = {
            "evaluation_users_count": len(eval_dataset.eval_user_ids),
            "catalog_size": eval_dataset.total_catalog_size,
            "top_k_evaluated": self.eval_cfg.top_k_list,
            "results": results_df.to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # 2. Save Markdown (formatted table without external dependencies)
        def _df_to_md(df: pd.DataFrame) -> str:
            headers = [str(c) for c in df.columns]
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in df.values]
            return "\n".join([header_row, sep_row] + data_rows)

        summary_k10 = results_df[results_df["top_k"] == 10].copy()
        md_content = [
            "# 📊 Recommender Systems Offline Scientific Benchmark Summary\n",
            f"- **Evaluated Users**: {len(eval_dataset.eval_user_ids):,}",
            f"- **Catalog Size**: {eval_dataset.total_catalog_size:,} movies",
            r"- **Evaluation Protocol**: Temporal Leave-2 Out Split (Rating $\ge$ 3.5)" + "\n",
            "### Results at Top-10 Cutoff ($K = 10$):\n",
            _df_to_md(summary_k10),
            "\n### Complete Multi-Cutoff Comparison:\n",
            _df_to_md(results_df),
        ]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))

        logger.info(f"Saved benchmark reports to {benchmarks_dir}")


def run_benchmark_suite(top_k: list[int] | None = None) -> pd.DataFrame:
    """Convenience function to run benchmark suite."""
    runner = BenchmarkRunner()
    return runner.run_benchmark(top_k_list=top_k)


if __name__ == "__main__":
    run_benchmark_suite()
