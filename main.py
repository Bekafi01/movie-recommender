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
app.add_typer(data_app, name="data")

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

        table = Table(title=f"Search Results for '{query}'", show_header=True, header_style="bold blue")
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


if __name__ == "__main__":
    app()
