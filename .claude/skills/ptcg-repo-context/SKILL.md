---
name: ptcg-repo-context
description: Load the Pokémon TCG AI Battle Challenge repo's hard constraints, data layout, and agent/benchmark interfaces. Use at the start of ANY task in this repo that touches training, evaluation, agents, decks, episodes, or submissions — before writing code, not after. Also use when about to hardcode a device, a path, an episode directory, or a deck.
---

# PTCG repo context

Load this before writing code in this repo. Everything below is verified against
the source tree, not inferred. Where something is unverified it says so.

## The constraints that invalidate work if ignored

**Device is never hardcoded.** `src/pokemon_tcg/device.py::resolve_device(override=None)`
returns `'mps'` or `'cpu'` — arg, then `PTCG_DEVICE` env var, then auto-detect. There
is deliberately **no CUDA branch**: training is Apple Silicon, the Kaggle evaluator is
CPU-only. Writing `torch.device("cuda")` or `.cuda()` is always wrong here. To rehearse
what the evaluator will do, force `PTCG_DEVICE=cpu` on the laptop.

**The evaluator envelope: ~1.6 vCPU, ~197.7 MiB, effectively single-threaded.**
⚠️ This came from Phase 0 discovery notes and is **not enforced anywhere in the repo** —
no test asserts it. Treat it as a design constraint you must respect and cannot verify
by running the test suite. Two consequences that have bitten before:
- `torch.set_num_threads(1)` must run **before** any heavy torch work; set late it
  silently does nothing.
- A model that fits in memory on the laptop can still OOM or time out on the ladder.
  Bundle size is checked at build time by `scripts/build_improved_submission.py`, which
  prints the tarball MiB — read that number, don't assume.

**Everything runs under `uv`.** `uv run python scripts/<name>.py`. Not bare `python`.

**Paths come from `src/pokemon_tcg/config.py`.** Import the constants; never hardcode
strings. `PROJECT_ROOT`, `DATA_DIR`, `EPISODES_DIR`, `EPISODES_SPLITS_DIR`, `MODELS_DIR`,
`SUBMISSIONS_DIR`, `REPORTS_DIR`, `FIGURES_DIR`, `CONFIGS_DIR`, `RANDOM_SEED = 42`,
`COMPETITION = "pokemon-tcg-ai-battle"`.

## Data layout

Episodes live at `data/episodes/splits/<split-folder>/<episode_id>.json`, keyed by
`data/episodes/splits/splits.json`:

| split | date | episodes | folder |
|---|---|---:|---|
| train | 2026-07-26 | 4554 | `splits/train-2026-07-26` |
| eval | 2026-07-27 | 4430 | `splits/eval-2026-07-27` |

Split methodology is the **held-out-DAY rule** (§F4): train and eval are distinct
calendar days, to avoid within-match leakage. Do not re-split randomly.

⚠️ `config.EPISODES_DIR` and `EPISODES_SPLITS_DIR` are path constants, but the **actual
source of truth for resolving a split is `il_dataset.resolve_split_dir(split)`** — an
earlier version of config pointed at `data/episodes/{train,eval}/`, which never existed.
Use the function.

`data/episodes/manifest.csv` carries per-episode
`episode_id, create_time, avg_score, min_score, sum_score, agent_count, size_bytes`.
**`avg_score`/`min_score` are the player-rating field** — relevant to any experiment
about filtering the training corpus by skill. ⚠️ Coverage measured 2026-08-02
(`scripts/check_weight_plumbing.py`): the manifest holds only **2026-07-01 (5,266
eps) and 2026-07-27 (the eval day, 100%)** — the **train day 2026-07-26 has ZERO
manifest rows**, so skill-filtering the training corpus is blocked until a
train-day manifest is fetched from the Kaggle episodes dataset.
`il_dataset.load_manifest_scores()` is the loader; missing episodes get a −1.0
sentinel in `ILDataset(with_meta=True)`.

`data/episodes/local/` holds 2 hand-captured battles (`local_battle.json`,
`local_battle_envrun.json`) for smoke tests — not a corpus.

## Agents and the benchmark harness

`scripts/benchmark_agents.py` is the round-robin. 13 agents in `AGENT_FILES`:

- **Ours:** `rule_baseline`, `improved_prob_main`, `agent_core_improved`, `proto`, `il_agent`
- **Floor, not a competitor:** `random_legal` — uniform random over legal moves. If a
  trained policy does not beat this decisively, its offline accuracy is not evidence.
- **Public opponent pool** (from Kaggle's Code tab, individually safety-reviewed):
  `kiyotah_dragapult`, `kiyotah_iono`, `kiyotah_abomasnow`, `dedquoc_rule_engine`,
  `ryotasueyoshi_alakazam`, `makthanithin_improved_prob`, `mechi22_alakazam`

Key interfaces:

- `load_agent(name)` → the `agent` callable. Prefers `AGENT_MAIN[name]` (the real
  submission `main.py`, deck already wired) and falls back to the bare module.
- **How a deck reaches an agent:** module-level `my_deck`. If the bare module lacks it,
  `load_agent` injects `[int(x) for x in (path.parent/"deck.csv").read_text().splitlines()][:60]`.
  9 of 13 agent dirs ship a `deck.csv`; the repo root also has one.
- `play_match(agent_a, agent_b, env_factory, pairs)` plays **mirrored pairs** — each pair
  is 2 games with seats swapped, cancelling first-player advantage. `--games N` means N
  *mirrored pairs* per ordered pair, i.e. 2N games.
- Env is `kaggle_environments.make("cabt")`.
- Glicko-1 (`scripts/glicko1.py`) persists to `reports/glicko_ratings.json` and
  **compounds across runs**. Self-play is excluded from rating periods. Every rating has
  an RD — quote it, and never call two agents different when their intervals overlap.

⚠️ **Deck is bound to agent identity in this harness — there is no deck axis.** Any task
that wants to vary deck while holding policy fixed has to add that mechanism first. Say
so instead of silently benchmarking a confound.

Other scripts: `train_il.py`, `eval_rung1.py` (offline top-1/top-3 action match vs a
majority-class baseline on the same rows), `eval_rung3_sanity.py`, `replay_episode.py`,
`verify_improved_agent.py`, `build_*_submission.{sh,py}`.

## Model / dataset code

`src/pokemon_tcg/`: `config.py`, `data.py`, `device.py`, `il_dataset.py`, `il_model.py`,
`logging_utils.py`. The encoder entry points are `il_dataset.encode_observation(...)` and
`il_dataset.iter_decisions(...)`; the policy is `il_model.PTCGImitationPolicy`, loaded via
`.from_pretrained(model_dir)`. Read the actual signatures before calling — do not guess
argument names.

## Standing rules for this repo

1. **Action masking is mandatory.** Illegal actions must never be sampleable.
2. **No opponent-private information in the encoder.** The replay log contains it; leaking
   it is invisible offline and fatal on the ladder. Check this explicitly when touching
   `il_dataset.py`.
3. **Report uncertainty with every comparison.** ≥3 seeds, and Glicko RD or a win-rate σ
   on every number. A point estimate alone is not a result here.
4. **Compare at equal steps, not equal epochs.** Fewer episodes makes an epoch smaller, so
   equal-epoch comparisons secretly under-train the smaller arm.
5. **Say when a number is unverified.** This project's notes already distinguish sourced
   from assumed claims; keep that discipline in code comments and reports.
