"""Main CLI entrypoint for the Movie Recommender System."""

from __future__ import annotations

import contextlib
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from recsys.data.db import DataRepository
from recsys.data.pipeline import run_data_pipeline
from recsys.evaluation.benchmark import run_benchmark_suite
from recsys.models.hybrid import HybridRecommender
from recsys.models.trainer import train_all_models

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

app = typer.Typer(
    name="recsys",
    help="Production-Grade Movie Recommendation Engine CLI",
    add_completion=False,
)
data_app = typer.Typer(help="Data ingestion and preprocessing commands")
train_app = typer.Typer(help="Model training commands")
rec_app = typer.Typer(help="Inference and recommendation commands")
eval_app = typer.Typer(help="Offline evaluation and benchmarking commands")

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(rec_app, name="recommend")
app.add_typer(eval_app, name="eval")

console = Console(highlight=False)


@data_app.command("preprocess")
def preprocess_data() -> None:
    """Ingest, clean, and persist the 5 raw CSVs into clean Parquet and SQLite."""
    console.print(Panel("[bold green]Starting RecSys Data Pipeline[/bold green]", expand=False))
    try:
        summary = run_data_pipeline()
        table = Table(title="Data Pipeline Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Clean Movies", f"{summary['num_movies']:,}")
        table.add_row("Clean Ratings", f"{summary['num_ratings']:,}")
        table.add_row("Unique Users", f"{summary['num_users']:,}")

        console.print(table)
        console.print("[bold green][OK] Pipeline completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Pipeline failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@data_app.command("search")
def search_movie(query: str = typer.Argument(..., help="Title substring to search")) -> None:
    """Search for movies by title substring in the clean catalog."""
    repo = DataRepository()
    try:
        results = repo.search_movies(query, limit=10)
        if not results:
            console.print(f"[yellow]No movies found matching '{query}'.[/yellow]")
            return

        table = Table(
            title=f"Search Results for '{query}'", show_header=True, header_style="bold blue"
        )
        table.add_column("Movie ID", style="dim")
        table.add_column("TMDB ID", style="dim")
        table.add_column("Title", style="bold white")
        table.add_column("Year", style="cyan")
        table.add_column("Genres", style="magenta")
        table.add_column("Rating", style="yellow")
        table.add_column("Weighted", style="green")

        for r in results:
            table.add_row(
                str(r.get("movie_id", "")),
                str(r.get("tmdb_id", "")),
                str(r.get("title", "")),
                str(r.get("release_year", "N/A")),
                str(r.get("genres_str", "")),
                f"{r.get('vote_average', 0.0):.1f}",
                f"{r.get('weighted_rating', 0.0):.2f}",
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Search failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@train_app.command("all")
def train_all(
    skip_neural: bool = typer.Option(
        False, "--skip-neural", help="Skip Neural CF training for faster run"
    ),
) -> None:
    """Train all multi-paradigm recommendation models (Popularity, TF-IDF, Semantic FAISS, SVD, NeuMF)."""
    console.print(
        Panel(
            "[bold green]Training Multi-Paradigm Recommendation Engines[/bold green]", expand=False
        )
    )
    try:
        results = train_all_models(train_neural_cf=not skip_neural)
        table = Table(
            title="Model Artifacts Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Model Engine", style="cyan")
        table.add_column("Artifact Path", style="green")

        for model_name, path in results["trained_models"].items():
            table.add_row(model_name.upper(), path)

        console.print(table)
        console.print(
            "[bold green][OK] All models successfully trained and persisted![/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Model training failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@rec_app.command("movie")
def recommend_movie(
    title: str = typer.Argument(..., help="Movie title (e.g., Inception, The Dark Knight)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of recommendations"),
) -> None:
    """Get recommendations similar to a movie title using Dense Semantic FAISS search."""
    try:
        model = HybridRecommender.load()
        recs = model.recommend(query=title, top_k=top_k)

        table = Table(
            title=f"Recommendations for '{title}'", show_header=True, header_style="bold green"
        )
        table.add_column("Rank", style="cyan")
        table.add_column("Title", style="bold white")
        table.add_column("Year", style="dim")
        table.add_column("Genres", style="magenta")
        table.add_column("Similarity Score", style="yellow")

        for _, row in recs.iterrows():
            table.add_row(
                str(int(row["rank"])),
                str(row["title"]),
                str(row.get("release_year", "N/A")),
                str(row.get("genres_str", "")),
                f"{row['score']:.1%}" if "score" in row else "N/A",
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Recommendation failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@rec_app.command("user")
def recommend_user(
    user_id: int = typer.Argument(..., help="User ID (e.g., 1, 42, 671)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of recommendations"),
) -> None:
    """Get personalized hybrid recommendations for a user ID."""
    try:
        model = HybridRecommender.load()
        recs = model.recommend(user_id=user_id, top_k=top_k)

        table = Table(
            title=f"Personalized Recommendations for User {user_id}",
            show_header=True,
            header_style="bold green",
        )
        table.add_column("Rank", style="cyan")
        table.add_column("Title", style="bold white")
        table.add_column("Year", style="dim")
        table.add_column("Genres", style="magenta")
        table.add_column("Predicted Score", style="yellow")

        for _, row in recs.iterrows():
            table.add_row(
                str(int(row["rank"])),
                str(row["title"]),
                str(row.get("release_year", "N/A")),
                str(row.get("genres_str", "")),
                f"{row.get('score', 0.0):.2f}",
            )
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Recommendation failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@eval_app.command("benchmark")
def evaluate_benchmark(
    k: int = typer.Option(10, "--top-k", "-k", help="Top-K cutoff for summary metrics"),
) -> None:
    """Execute scientific offline benchmark comparing all models on ranking and diversity."""
    console.print(
        Panel("[bold green]Executing Offline Scientific Benchmark[/bold green]", expand=False)
    )
    try:
        results_df = run_benchmark_suite(top_k=[5, k, 20])
        summary = results_df[results_df["top_k"] == k].copy()

        table = Table(
            title=f"Offline Scientific Evaluation Summary (Top-{k})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Model Engine", style="bold cyan")
        table.add_column(f"NDCG@{k}", style="green")
        table.add_column(f"MAP@{k}", style="green")
        table.add_column(f"Recall@{k}", style="yellow")
        table.add_column(f"Precision@{k}", style="yellow")
        table.add_column(f"Hit Rate@{k}", style="cyan")
        table.add_column(f"MRR@{k}", style="cyan")
        table.add_column("Coverage", style="magenta")
        table.add_column("Novelty", style="blue")
        table.add_column("Diversity", style="white")

        for _, r in summary.iterrows():
            table.add_row(
                str(r["model"]),
                f"{r['ndcg@k']:.4f}",
                f"{r['map@k']:.4f}",
                f"{r['recall@k']:.4f}",
                f"{r['precision@k']:.4f}",
                f"{r['hit_rate@k']:.4f}",
                f"{r['mrr@k']:.4f}",
                f"{r['coverage']:.2%}",
                f"{r['novelty']:.2f}",
                f"{r['diversity']:.4f}",
            )
        console.print(table)
        console.print(
            "[bold green][OK] Benchmark completed! Reports saved to artifacts/benchmarks/[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Benchmark failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("serve")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="API host interface"),
    port: int = typer.Option(8000, "--port", "-p", help="API server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Launch the FastAPI production inference microservice with Uvicorn."""
    import uvicorn

    console.print(
        Panel(
            f"[bold green]Starting RecSys FastAPI Server on http://{host}:{port}[/bold green]\n"
            f"[dim]Interactive Swagger UI: http://{host}:{port}/docs[/dim]",
            expand=False,
        )
    )
    uvicorn.run("recsys.serving.api:app", host=host, port=port, reload=reload)


@app.command("ui")
def launch_ui(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit web port"),
) -> None:
    """Launch the CineFlow AI interactive Streamlit Cinema Experience UI."""
    import subprocess

    console.print(
        Panel(
            f"[bold magenta]Launching CineFlow AI Cinema UI on http://localhost:{port}[/bold magenta]",
            expand=False,
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/recsys/ui/app.py",
            "--server.port",
            str(port),
        ]
    )


if __name__ == "__main__":
    app()
