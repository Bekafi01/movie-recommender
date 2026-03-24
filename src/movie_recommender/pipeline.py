from __future__ import annotations

import argparse
from pathlib import Path

from .data import build_movies_database, load_movies_from_db
from .model import save_artifacts, train_content_model


def train_pipeline(
    project_root: Path,
    rebuild_db: bool = False,
) -> None:
    data_dir = project_root / "data"
    db_path = data_dir / "movies.db"
    artifacts_dir = project_root / "artifacts"

    if rebuild_db or not db_path.exists():
        build_movies_database(data_dir=data_dir, db_path=db_path)

    movies_df = load_movies_from_db(db_path=db_path)
    trained = train_content_model(movies_df)
    save_artifacts(artifacts=trained, artifacts_dir=artifacts_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train movie recommender artifacts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root directory",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="Rebuild data/movies.db from ml - 1m CSV files before training",
    )
    args = parser.parse_args()

    train_pipeline(project_root=args.project_root, rebuild_db=args.rebuild_db)
    print("Training completed. Artifacts are available in artifacts/.")


if __name__ == "__main__":
    main()
