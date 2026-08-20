"""Custom exception hierarchy for the RecSys engine."""

from __future__ import annotations


class RecSysError(Exception):
    """Base exception for all RecSys errors."""


class DataIngestionError(RecSysError):
    """Raised when raw data files cannot be found or read."""


class DataProcessingError(RecSysError):
    """Raised when data transformation, ID mapping, or parsing fails."""


class ModelNotFoundError(RecSysError):
    """Raised when a requested model artifact cannot be found or loaded."""


class MovieNotFoundError(RecSysError):
    """Raised when a movie title or ID is not found in the catalog."""

    def __init__(self, query: str | int, suggestions: list[str] | None = None):
        self.query = query
        self.suggestions = suggestions or []
        msg = f"Movie '{query}' not found in catalog."
        if self.suggestions:
            msg += f" Did you mean: {', '.join(self.suggestions)}?"
        super().__init__(msg)


class UserNotFoundError(RecSysError):
    """Raised when a user ID does not exist in the training interaction matrix."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        super().__init__(f"User ID {user_id} not found in interaction history.")


class ModelTrainingError(RecSysError):
    """Raised when training a recommendation model fails."""


class EvaluationError(RecSysError):
    """Raised when computing offline evaluation metrics fails."""
