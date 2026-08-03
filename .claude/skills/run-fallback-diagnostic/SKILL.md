---
name: run-fallback-diagnostic
description: Run, measure, or gate the Pokémon TCG agent's silent-fallback rate. Use when asked to run the fallback diagnostic, measure how often il_agent (or an s2 arm) falls back to _safe_choice instead of the model, answer "is the bad score a bug or the model?", check action-masking correctness (model_action_illegal), verify a checkpoint actually loaded, or smoke-test the agent before building a submission. Also the map of every silent fallback site in our agent code.
---

# Run the fallback diagnostic

A fallback that fires silently is worse than a crash: the agent "runs",
scores badly, and nothing says whether the model is weak or `_safe_choice`
answered 40% of the decisions. The driver is
`scripts/fallback_diagnostic.py`: it plays a target agent against chosen
opponents with `PTCG_FALLBACK_TRACK=1`, then reports fallbacks **by reason,
as a rate over `decisions`** — the denominator is the whole story. Counters
live in `agents/il_agent/agent_core.py` (`diag_snapshot`/`diag_first`,
re-exported by every `agents/s2_arms/*` wrapper); the full site map is
[notes/fallback_inventory.md](../../../notes/fallback_inventory.md).

All paths below are relative to the repo root; every command was run and
worked on 2026-08-03 (macOS ARM64, native cg engine).

## Prerequisites

Everything runs under `uv` (no separate install step). **In a git worktree**
first symlink the untracked model/episode dirs from the main checkout, or
every decision silently counts as `model_unavailable`:

```bash
MAIN=$(dirname "$(git rev-parse --git-common-dir)"); [ -e models/il_agent ] || ln -s "$MAIN/models/il_agent" models/il_agent; [ -e data/episodes ] || ln -s "$MAIN/data/episodes" data/episodes
```

## Run (agent path)

Quick smoke — il_agent vs the rule baseline, 2 games, ~15 s:

```bash
uv run python scripts/fallback_diagnostic.py --pairs 1
```

Reads e.g. `decisions=70 fallbacks=0 fallback_rate=0.00%` (decision counts
vary run to run — cabt exposes no RNG seed) plus a by-reason
table and, for each reason that fired, occurrence #1 with full state /
traceback (complete copy in `reports/fallback_diagnostic.json`).

Against the public benchmark pool (the "bug or model?" run — do this
before spending a submission):

```bash
uv run python scripts/fallback_diagnostic.py --opponents rung2 --pairs 1 --out reports/fallback_diag_pool.json
```

Masking gate (exit 2 unless `model_action_illegal == 0` and decisions > 0):

```bash
uv run python scripts/fallback_diagnostic.py --opponents random_legal --pairs 1 --assert-clean --no-tb
```

Other knobs: `--agent s2_e0_s42` (any `AGENT_FILES` key, `@deck-tag` ok),
`--tb-dir` (TensorBoard scalars, default `runs/fallback_diag`), `--no-tb`.

Reason key: `model_unavailable:*`, `policy_exception:<ExcClass>`,
`step_timeout`, `encode_none`, `model_action_illegal`, `min_count_unmet`,
`no_legal_actions`, `too_many_options` are fallbacks (counted in the rate);
`unknown_card`, `unknown_attack`, `nan_logits`, `model_load_error`,
`model_dir_redirect` are silent-degradation signals (reported, never in
the rate — the model still chose).

## Verify the tracker itself (negative control)

A zero from a broken tracker is the exact bug class this hunts. Force a
fallback storm and confirm ~100%:

```bash
mkdir -p /tmp/empty_model_dir && IL_MODEL_DIR=/tmp/empty_model_dir uv run python scripts/fallback_diagnostic.py --pairs 1 --no-tb --out reports/fallback_diag_negcontrol.json
```

Expect `fallback_rate=100.00%`, `model_unavailable:load_failed`, and the
`model.safetensors`-missing traceback under `model_load_error`.

## Full benchmark integration

`scripts/benchmark_agents.py` sets the flag itself and, for every agent
exposing `diag_snapshot`, prints the rate table (stdout + a
`[fallback-diag]` stderr summary that survives redirection), stores
`fallback_diag`/`fallback_first` in the result JSON, and writes TensorBoard
scalars under `fallback/<agent>/…` to `--tb-dir` (default
`runs/benchmark_fallbacks`; `--no-tb` to skip). Nothing extra to pass.

## Test

```bash
uv run python tests/test_fallback_tracking.py
```

(Also pytest-compatible, but pytest is not in this venv — run it directly.)

## Under Kaggle submission

The bundle never sets `PTCG_FALLBACK_TRACK`, so every hook is one falsy
branch — a genuine no-op inside the evaluator's ~1.6 vCPU / ~198 MiB
budget. Nothing to strip at build time.

## Gotchas (all hit for real)

- **The flag is read at import.** Setting `PTCG_FALLBACK_TRACK=1` after an
  agent module is loaded does nothing. The driver and benchmark set it
  before `load_agent`; do the same in any new harness.
- **A nonexistent `IL_MODEL_DIR` does not fail** — it silently redirects to
  `models/il_agent` and benchmarks the wrong checkpoint. This defeated the
  first negative control this session. It now surfaces as
  `model_dir_redirect` in the snapshot; to force a load failure use an
  *existing but empty* dir (see negative control above).
- **`diag_reset()` wipes import-time events**, which is why
  `model_dir_redirect` is a snapshot property, not a counter. Don't convert
  it back.
- **`decisions` counts every select-carrying `agent()` call**, including
  the cabt interpreter's stale-echo re-asks of the inactive player. Rates
  are comparable across runs; don't equate the denominator with "turns".
- **Extending the tracker: never add a new `try/except`.** It observes the
  existing fallback sites (inventory above); manufacturing fallbacks to
  count defeats the diagnostic. `_diag()` itself must never raise — it runs
  inside except handlers where a raise becomes an INVALID instant loss.
- **Interpreting results is leaderboard-gated**: a clean local fallback
  report says the model made the decisions locally — it says nothing about
  ladder strength. Load the `leaderboard-check` skill before writing
  "better/worse" anywhere.

## Troubleshooting

- `No module named pytest` → run tests directly:
  `uv run python tests/test_fallback_tracking.py`.
- `[fallback-diag] TensorBoard scalars skipped (...)` on stderr → torch's
  SummaryWriter backend missing in that env; the JSON report still has
  everything. Not an error.
- `<agent> exposes no diag_snapshot(); nothing to measure` → target isn't
  instrumented. Instrumented: `il_agent`, all `s2_*` arms; coarser rule-
  agent `_DIAG`: `rule_baseline` family, `improved_prob_main`,
  `mechi22_alakazam`.
- Every decision is `model_unavailable:no_ml_stack` → torch/pokemon_tcg
  import failed (check `uv run python -c "import torch"`); `:load_failed`
  → read the `model_load_error` traceback in `diag_first` / the JSON.
- `OpenSpiel environments: 41` INFO lines at startup → kaggle_environments
  noise, harmless.
