"""Central configuration: project paths and shared constants.

Import these paths everywhere instead of hardcoding strings so notebooks and
scripts stay portable regardless of the current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels up from this file (src/pokemon_tcg/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories.
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"  # legacy; episodes now live in EPISODES_DIR

# Episodes (behavior-cloning training corpus).
# Layout: EPISODES_DIR/splits/<train-DATE|eval-DATE>/<episode_id>.json,
# keyed by EPISODES_DIR/splits/splits.json (see pokemon_tcg.il_dataset.resolve_split_dir,
# the actual source of truth -- these two constants previously pointed at
# data/episodes/{train,eval}/, which never existed; fixed to match reality.)
EPISODES_DIR = DATA_DIR / "episodes"
EPISODES_SPLITS_DIR = EPISODES_DIR / "splits"

# Packed corpus on Hugging Face Hub (ADR-001): private dataset repo holding
# zstd-Parquet episode shards, layout {split}/day=YYYY-MM-DD/shard-NNN.parquet.
# EPISODES_PACKED_DIR is the local staging/cache dir mirroring that layout.
# Env override mirrors PTCG_DEVICE's pattern (e.g. point at a fork/test repo).
HF_EPISODES_REPO = os.environ.get("PTCG_HF_EPISODES_REPO", "Rami/ptcg-episodes")
EPISODES_PACKED_DIR = DATA_DIR / "episodes_packed"

# Artifact directories.
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Kaggle competition slug.
COMPETITION = "pokemon-tcg-ai-battle"

# Reproducibility.
RANDOM_SEED = 42
