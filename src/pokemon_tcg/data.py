"""Data acquisition and loading utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl

from . import config


def download_competition_data(
    competition: str = config.COMPETITION,
    dest: Path = config.RAW_DATA_DIR,
) -> None:
    """Download and unzip competition data via the Kaggle CLI.

    Requires Kaggle API credentials at ``~/.kaggle/kaggle.json``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)],
        check=True,
    )
    for zip_path in dest.glob("*.zip"):
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(dest)], check=True)


def load_csv(name: str, data_dir: Path = config.RAW_DATA_DIR) -> pl.DataFrame:
    """Load a CSV from a data directory into a Polars DataFrame."""
    return pl.read_csv(data_dir / name)
