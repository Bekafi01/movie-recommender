from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def space_tokenizer(text: str) -> list[str]:
    return text.split()


def prepare_movies_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["genres"] = prepared["genres"].fillna("(no genres listed)")
    prepared["genres"] = prepared["genres"].apply(
        lambda value: value if isinstance(value, list) else str(value).split("|")
    )
    prepared["genre_str"] = prepared["genres"].apply(lambda value: " ".join(value))
    return prepared


@dataclass
class TrainedArtifacts:
    vectorizer: CountVectorizer
    similarity: np.ndarray
    dataframe: pd.DataFrame


def train_content_model(df: pd.DataFrame) -> TrainedArtifacts:
    prepared = prepare_movies_dataframe(df)

    vectorizer = CountVectorizer(
        tokenizer=space_tokenizer,
        binary=True,
        lowercase=False,
    )
    genre_matrix = vectorizer.fit_transform(prepared["genre_str"])
    similarity = cosine_similarity(genre_matrix)

    return TrainedArtifacts(vectorizer=vectorizer, similarity=similarity, dataframe=prepared)


class GenreRecommender:
    def __init__(self, similarity_matrix: np.ndarray, df: pd.DataFrame):
        self.similarity_matrix = similarity_matrix
        self.df = df.copy()
        self.indices = pd.Series(df.index, index=df["title"]).to_dict()
        self.titles = df["title"].tolist()

    def get_recommendations(
        self,
        title: str,
        top_n: int = 5,
        exclude_self: bool = True,
    ) -> pd.DataFrame:
        if title not in self.indices:
            available = [movie for movie in self.titles if title.lower() in movie.lower()]
            if available:
                raise ValueError(f"Movie '{title}' not found. Did you mean: {available[:5]}?")
            raise ValueError(f"Movie '{title}' not found in database.")

        idx = self.indices[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda pair: pair[1], reverse=True)
        sim_scores = sim_scores[1 : top_n + 1] if exclude_self else sim_scores[:top_n]

        movie_indices = [pair[0] for pair in sim_scores]
        scores = [pair[1] for pair in sim_scores]

        recommendations = self.df.iloc[movie_indices][
            ["movie_id", "title", "genres", "rating"]
        ].copy()
        recommendations["similarity_score"] = scores
        recommendations["rank"] = range(1, len(recommendations) + 1)

        return recommendations[
            ["rank", "movie_id", "title", "genres", "rating", "similarity_score"]
        ]

    def get_similarity_between(self, title1: str, title2: str) -> dict[str, Any]:
        if title1 not in self.indices or title2 not in self.indices:
            raise ValueError("One or both movies not found.")

        idx1 = self.indices[title1]
        idx2 = self.indices[title2]
        score = float(self.similarity_matrix[idx1, idx2])

        return {
            "movie_1": title1,
            "movie_2": title2,
            "similarity_score": round(score, 4),
            "similarity_percentage": f"{score * 100:.1f}%",
        }


def save_artifacts(artifacts: TrainedArtifacts, artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with open(artifacts_dir / "count_vectorizer.pkl", "wb") as handle:
        pickle.dump(artifacts.vectorizer, handle)

    with open(artifacts_dir / "similarity_matrix.pkl", "wb") as handle:
        pickle.dump(artifacts.similarity, handle)

    with open(artifacts_dir / "movies_data.pkl", "wb") as handle:
        pickle.dump(artifacts.dataframe, handle)

    np.savez_compressed(artifacts_dir / "similarity_matrix.npz", similarity=artifacts.similarity)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "model_type": "Content-Based (CountVectorizer + Cosine Similarity)",
        "num_movies": len(artifacts.dataframe),
        "num_genres": len(artifacts.vectorizer.get_feature_names_out()),
        "genres": artifacts.vectorizer.get_feature_names_out().tolist(),
        "vectorizer_config": {
            "type": "CountVectorizer",
            "binary": True,
            "lowercase": False,
        },
        "similarity_metric": "cosine",
        "files": {
            "vectorizer": "count_vectorizer.pkl",
            "similarity_matrix": "similarity_matrix.pkl",
            "data": "movies_data.pkl",
            "compressed_similarity": "similarity_matrix.npz",
        },
    }

    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def load_artifacts(artifacts_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    with open(artifacts_dir / "similarity_matrix.npz", "rb") as handle:
        similarity = np.load(handle)["similarity"]

    with open(artifacts_dir / "movies_data.pkl", "rb") as handle:
        dataframe = pickle.load(handle)

    return similarity, dataframe
