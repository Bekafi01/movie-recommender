"""Unit tests for utility functions, timers, and exceptions."""

import time

from recsys.utils.exceptions import (
    DataIngestionError,
    MovieNotFoundError,
    RecSysError,
    UserNotFoundError,
)
from recsys.utils.logger import get_logger
from recsys.utils.timer import Timer, timed


def test_exception_hierarchy() -> None:
    """Test custom domain exception formatting and inheritance."""
    err = DataIngestionError("File missing")
    assert isinstance(err, RecSysError)

    movie_err = MovieNotFoundError("Matrix", suggestions=["The Matrix", "Matrix Reloaded"])
    assert "The Matrix" in str(movie_err)
    assert movie_err.query == "Matrix"
    assert len(movie_err.suggestions) == 2

    user_err = UserNotFoundError(999)
    assert "999" in str(user_err)
    assert user_err.user_id == 999


def test_timer_context_manager() -> None:
    """Test Timer context manager measures positive elapsed time."""
    with Timer(description="Test sleep", log_output=False) as t:
        time.sleep(0.01)

    assert t.elapsed_sec >= 0.009


def test_timed_decorator() -> None:
    """Test @timed decorator wraps function and returns correct value."""

    @timed("Quick function")
    def compute(a: int, b: int) -> int:
        return a + b

    result = compute(3, 7)
    assert result == 10


def test_logger_creation() -> None:
    """Test get_logger returns a valid configured logger."""
    logger = get_logger("test_logger", level="DEBUG")
    assert logger.name == "test_logger"
    assert len(logger.handlers) > 0
