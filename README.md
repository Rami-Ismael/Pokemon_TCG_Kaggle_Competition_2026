# Pokemon TCG — Kaggle Competition 2026

Project scaffold managed with [uv](https://docs.astral.sh/uv/), using
[marimo](https://marimo.io/) notebooks.

## Setup

```bash
uv sync
```

Add Kaggle API credentials at `~/.kaggle/kaggle.json` (chmod 600), then download
the competition data:

```bash
uv run python -c "from pokemon_tcg import data; data.download_competition_data()"
```

## Project layout

```
├── configs/            # experiment / model configs
├── data/
│   ├── raw/            # original competition data (immutable, gitignored)
│   ├── interim/        # intermediate transformations
│   ├── processed/      # final feature sets for modeling
│   └── external/       # third-party / external data
├── models/             # trained model artifacts (gitignored)
├── notebooks/          # marimo notebooks (e.g. 01_eda.py)
├── reports/figures/    # generated plots
├── submissions/        # Kaggle submission files (gitignored)
└── src/pokemon_tcg/    # importable package
    ├── config.py       # central paths & constants
    └── data.py         # download / load helpers
```

## Usage

Launch a marimo notebook:

```bash
uv run marimo edit notebooks/01_eda.py
```

Lint / format:

```bash
uv run ruff check .
uv run ruff format .
```

Import shared paths and helpers from anywhere:

```python
from pokemon_tcg import config, data

df = data.load_csv("train.csv")
print(config.RAW_DATA_DIR)
```
