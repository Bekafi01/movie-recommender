"""Execution timing and latency profiling utilities."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from .logger import get_logger

logger = get_logger("recsys.timer")
F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """Context manager for measuring execution time."""

    def __init__(self, description: str = "Operation", log_output: bool = True):
        self.description = description
        self.log_output = log_output
        self.elapsed_sec: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> Timer:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.elapsed_sec = time.perf_counter() - self._start_time
        if self.log_output:
            if self.elapsed_sec < 1.0:
                logger.info(f"{self.description} completed in {self.elapsed_sec * 1000:.2f} ms")
            else:
                logger.info(f"{self.description} completed in {self.elapsed_sec:.3f} s")


def timed(description: str | None = None) -> Callable[[F], F]:
    """Decorator to measure and log function execution latency."""

    def decorator(func: F) -> F:
        desc = description or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(description=desc):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
