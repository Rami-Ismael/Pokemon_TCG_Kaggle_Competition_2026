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

## Behavior-cloning agent (`il_agent`)

Full design rationale lives in `notes/`: `phase0_discovery_report.md` (data/
obs-schema discovery, measured on the real corpus), `phase1_decisions.md`
(architecture/param/masking/deck decisions), `phase6_projection.md`
(time/cost projection). Summary:

**Obs schema** (`agent(obs_dict)`'s argument): `select` / `current` / `logs`
/ `remainingOverageTime` / `search_begin_input` / `step`. `current` hides
the opponent's hand (`None`) and both decks' order (no `deck` key at all)
by construction -- verified empirically, no leakage into the encoder.
`current.result` is never populated in this dataset's replay format (always
`-1`); use the episode-level `rewards` field for outcomes instead.
`select is None` marks the deck-submission step.

**Encoder** (`src/pokemon_tcg/il_dataset.py`, v1): Pattern-B option scoring
-- every legal move becomes one token, scored by a shared head, masked to
`-inf` when illegal/padding. A synthetic DECLINE slot is added whenever
`select.minCount == 0`. Multi-select (`maxCount > 1`) decisions are
unrolled autoregressively (one training row per pick, re-masking prior
picks). Card ids are shared between the state encoder and the option-
reference features via one embedding table, with ids clamped into vocab
range (`_clamp_id`) so an unseen card can never crash inference.

**Model** (`src/pokemon_tcg/il_model.py`): `transformers.BertModel` fed
`inputs_embeds` only (no NLP vocabulary involved), hidden=192/layers=6/
heads=6 by default (~3.3M params).

**Device**: `src/pokemon_tcg/device.py::resolve_device()` is the only place
that picks a device (MPS if available, else CPU -- no CUDA branch). Force
CPU with `--device cpu` / `PTCG_DEVICE=cpu` to reproduce what the Kaggle
evaluator does.

Reproduce a run:

```bash
uv run python scripts/train_il.py --epochs 1          # ~20 min, full train day, see phase6_projection.md
uv run python scripts/eval_rung1.py --max-episodes 500  # Rung 1: offline accuracy vs. majority baseline
uv run python scripts/benchmark_agents.py --agents il_agent,rule_baseline --games 15  # Rung 2
uv run python scripts/eval_rung3_sanity.py --games 5     # Rung 3: play + inspect transcripts
uv run python tests/test_il_pipeline.py                  # smoke tests
```

TensorBoard logs land under `runs/<timestamp>/` (`uv run tensorboard --logdir runs/`);
each run directory also records `device_info.json`, `run_config.json`, and
the git SHA it was trained at. Checkpoints save to `models/il_agent/`
(state dict moved to CPU before saving, so an MPS-trained checkpoint loads
correctly on a CPU-only evaluator) with a `train_metadata.json` recording
what it was trained on, for how long, and its eval accuracy.
