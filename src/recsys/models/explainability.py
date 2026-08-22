"""Explainability engine providing human-readable reasons for movie recommendations."""

from __future__ import annotations

from typing import Any

import pandas as pd


class ExplainabilityEngine:
    """Generates feature overlap and collaborative reasoning explanations."""

    def __init__(self, movies_df: pd.DataFrame):
        self.movies_df = movies_df.set_index("movie_id").to_dict(orient="index")

    def explain(
        self, source_movie_id: int, recommended_movie_id: int, similarity_score: float = 0.0
    ) -> dict[str, Any]:
        """Generate detailed explanation for why recommended_movie was suggested for source_movie."""
        src = self.movies_df.get(source_movie_id, {})
        rec = self.movies_df.get(recommended_movie_id, {})

        if not src or not rec:
            return {
                "source_id": source_movie_id,
                "recommended_id": recommended_movie_id,
                "summary": "Recommended based on overall catalog similarity.",
                "shared_directors": [],
                "shared_cast": [],
                "shared_genres": [],
                "shared_keywords": [],
            }

        # Compare directors
        src_dirs = set(src.get("directors_list", []))
        rec_dirs = set(rec.get("directors_list", []))
        shared_directors = [d.replace("_", " ").title() for d in (src_dirs & rec_dirs)]

        # Compare cast
        src_cast = set(src.get("cast_list", []))
        rec_cast = set(rec.get("cast_list", []))
        shared_cast = [c.replace("_", " ").title() for c in (src_cast & rec_cast)]

        # Compare genres
        src_genres = set(src.get("genres_list", []))
        rec_genres = set(rec.get("genres_list", []))
        shared_genres = [g.replace("_", " ").title() for g in (src_genres & rec_genres)]

        # Compare keywords
        src_kws = set(src.get("keywords_list", []))
        rec_kws = set(rec.get("keywords_list", []))
        shared_kws = [k.replace("_", " ").title() for k in (src_kws & rec_kws)]

        reasons: list[str] = []
        if shared_directors:
            reasons.append(f"Directed by {', '.join(shared_directors)}")
        if shared_cast:
            reasons.append(f"Starring {', '.join(shared_cast)}")
        if shared_genres:
            reasons.append(f"Matching genres: {', '.join(shared_genres)}")
        if shared_kws:
            reasons.append(f"Shared themes: {', '.join(shared_kws[:3])}")

        if not reasons:
            reasons.append("Similar thematic tone and audience reception")

        summary = (
            f"Recommended because you liked '{src.get('title', 'this movie')}': "
            + "; ".join(reasons)
            + "."
        )

        return {
            "source_id": source_movie_id,
            "source_title": src.get("title", ""),
            "recommended_id": recommended_movie_id,
            "recommended_title": rec.get("title", ""),
            "similarity_score": round(similarity_score, 4),
            "match_percentage": f"{max(0.0, min(1.0, similarity_score)) * 100:.1f}%",
            "summary": summary,
            "shared_directors": shared_directors,
            "shared_cast": shared_cast,
            "shared_genres": shared_genres,
            "shared_keywords": shared_kws[:5],
        }
