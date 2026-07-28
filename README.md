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

## Training data (episodes)

The behavior-cloning / imitation-learning corpus lives in `data/episodes/`: a flat
`manifest.csv` (one row per published day) plus a `splits/` directory with date-keyed
train/eval folders. Each split folder holds `<episode_id>.json` replay files in the
standard `kaggle-environments` replay format, and `splits/splits.json` records which
calendar day each split uses. The split enforces §F4's held-out-DAY rule (no
within-match leakage): train = 2026-07-26, eval = 2026-07-27. The full published corpus
spans 2026-06-16 → present (~862 GB / ~210k episodes over 41+ days); only the two split
days are downloaded locally.

## Project layout

```
├── configs/            # experiment / model configs
├── data/
│   ├── episodes/       # BC training corpus (gitignored)
│   │   ├── splits/
│   │   │   ├── train-2026-07-26/  # train-day replay JSONs
│   │   │   ├── eval-2026-07-27/   # held-out eval-day replay JSONs
│   │   │   └── splits.json        # split → date/episode metadata
│   │   └── manifest.csv  # one row per published day
│   ├── raw/            # original competition data (immutable, gitignored)
│   ├── interim/        # intermediate transformations
│   └── processed/      # final feature sets for modeling
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
