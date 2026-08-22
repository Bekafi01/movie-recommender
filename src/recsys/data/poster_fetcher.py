"""Official TMDB API v3 Poster Resolver with thread-safe persistent caching."""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from recsys.utils.logger import get_logger

# Load environment variables from .env
load_dotenv()

logger = get_logger(__name__)


class PosterResolver:
    """Resolves authentic theatrical movie posters via official TMDB API v3 with thread-safe persistent caching."""

    DEFAULT_FALLBACK = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?auto=format&fit=crop&w=500&q=80"
    TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

    def __init__(
        self,
        cache_file: Path | str = "artifacts/posters_cache.json",
        api_key: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_file)
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _load_cache(self) -> None:
        """Load persistent poster URL cache from JSON file."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f, self._lock:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached poster URLs from {self.cache_path}")
            except Exception as e:
                logger.warning(f"Failed to read poster cache file: {e}")
                with self._lock:
                    self.cache = {}

    def save_cache(self) -> None:
        """Save persistent poster URL cache to disk safely."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._lock:
                snapshot = dict(self.cache)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist poster cache to {self.cache_path}: {e}")

    def _query_tmdb_api(self, tmdb_id: int | str) -> str | None:
        """Query official TMDB API v3 for the current active poster hash."""
        if not self.api_key or not tmdb_id or str(tmdb_id) in ("None", "nan", "0"):
            return None
        url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={self.api_key}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CineFlowAI-RecSys/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.5) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    poster_path = data.get("poster_path")
                    if poster_path and isinstance(poster_path, str) and poster_path.strip():
                        clean_path = poster_path.strip()
                        if not clean_path.startswith("/"):
                            clean_path = f"/{clean_path}"
                        return f"{self.TMDB_IMAGE_BASE}{clean_path}"
        except Exception:
            return None
        return None

    def resolve(
        self,
        title: str,
        year: int | str | None = None,
        tmdb_id: int | str | None = None,
        tmdb_path: str | None = None,
    ) -> str:
        """Resolve movie poster URL: Cache -> TMDB API v3 -> TMDB Static Path -> Fallback."""
        clean_title = str(title).strip()
        cache_key = f"{clean_title}_{year}" if year else clean_title

        # 1. Check local persistent cache
        with self._lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

        # 2. Query official TMDB API v3
        if self.api_key and tmdb_id:
            tmdb_api_poster = self._query_tmdb_api(tmdb_id)
            if tmdb_api_poster:
                with self._lock:
                    self.cache[cache_key] = tmdb_api_poster
                self.save_cache()
                return tmdb_api_poster

        # 3. Try TMDB static path if available
        if tmdb_path and str(tmdb_path).strip() and str(tmdb_path) not in ("None", "nan"):
            clean_path = str(tmdb_path).strip()
            if not clean_path.startswith("/"):
                clean_path = f"/{clean_path}"
            tmdb_url = f"{self.TMDB_IMAGE_BASE}{clean_path}"
            with self._lock:
                self.cache[cache_key] = tmdb_url
            self.save_cache()
            return tmdb_url

        # 4. Fallback placeholder
        with self._lock:
            self.cache[cache_key] = self.DEFAULT_FALLBACK
        return self.DEFAULT_FALLBACK

    def bulk_enrich(
        self,
        movies: list[dict[str, Any]],
        max_workers: int = 16,
    ) -> dict[str, str]:
        """Bulk fetch posters in parallel using official TMDB API v3."""
        logger.info(
            f"Starting bulk TMDB poster resolution for {len(movies)} movies with {max_workers} threads..."
        )

        def _fetch_single(m: dict[str, Any]) -> tuple[str, str]:
            t = m.get("title", "")
            y = m.get("release_year")
            t_id = m.get("tmdb_id")
            p = m.get("poster_path")
            url = self.resolve(title=t, year=y, tmdb_id=t_id, tmdb_path=p)
            key = f"{t}_{y}" if y else t
            return key, url

        results: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_movie = {executor.submit(_fetch_single, m): m for m in movies}
            for future in as_completed(future_to_movie):
                try:
                    key, url = future.result()
                    results[key] = url
                except Exception as e:
                    logger.debug(f"Error fetching poster: {e}")

        self.save_cache()
        logger.info(f"Completed bulk TMDB poster resolution. Total cached: {len(self.cache)}")
        return results
