"""Central configuration: project paths and shared constants.

Import these paths everywhere instead of hardcoding strings so notebooks and
scripts stay portable regardless of the current working directory.
"""

from __future__ import annotations

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
# Layout: EPISODES_DIR/{train,eval}/<episode_id>.json  +  EPISODES_DIR/manifest.csv
EPISODES_DIR = DATA_DIR / "episodes"
EPISODES_TRAIN_DIR = EPISODES_DIR / "train"
EPISODES_EVAL_DIR = EPISODES_DIR / "eval"

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
