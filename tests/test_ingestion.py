"""Unit tests for raw data ingestion and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import AppConfig
from recsys.data.ingestion import RawDataIngestor
from recsys.utils.exceptions import DataIngestionError


def test_missing_raw_files_raises_error(tmp_path: Path) -> None:
    """Test that missing raw files raises DataIngestionError."""
    empty_raw_dir = tmp_path / "raw"
    empty_raw_dir.mkdir(parents=True)

    config = AppConfig()
    config.paths.data_raw_dir = str(empty_raw_dir)

    ingestor = RawDataIngestor(config=config)
    with pytest.raises(DataIngestionError, match="Missing required raw data files"):
        ingestor.verify_files_exist()


def test_schema_validation_failure() -> None:
    """Test that invalid dataframe schema raises DataIngestionError."""
    ingestor = RawDataIngestor()
    invalid_df = pd.DataFrame({"wrong_col": [1, 2]})

    with pytest.raises(DataIngestionError, match="missing required columns"):
        ingestor._validate_schema("links", invalid_df)
