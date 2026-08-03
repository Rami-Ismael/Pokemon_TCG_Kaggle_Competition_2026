---
name: run-ptcg-battle
description: Build, run, and drive the Pokémon TCG agent — launch real cabt battles, smoke-test the checkpoint, render a battle replay you can watch. Use when asked to run the app or agent, play a game, verify a change works in a live battle (not just tests), screenshot or visualize a replay, or set up a fresh clone or worktree so battles run at all.
---

The "app" of this repo is a Pokémon TCG battle: two agent callables inside
`kaggle_environments.make("cabt")` (native `cg` engine, git-tracked at
`data/external/cg-lib`). Drive it with
`.claude/skills/run-ptcg-battle/driver.py` — a three-subcommand smoke
harness that reuses the proven `scripts/benchmark_agents.py::load_agent`
path. All paths below are relative to the repo (or worktree) root.

## Prerequisites

Verified on macOS arm64 (Apple Silicon) with [uv](https://docs.astral.sh/uv/)
installed. The engine is a compiled native lib; `cg-lib` ships
`libcg.dylib` (macOS arm64, the one exercised here) plus Linux `.so`
builds that this skill has not verified. No GPU needed — smokes run CPU.

## Setup

`uv run` auto-syncs the venv on first use (~1 min in a fresh worktree:
builds the `pokemon_tcg` package + installs deps; battles afterwards are
seconds). The only real setup is worktree data wiring — **run doctor
first, it prints the exact fix commands**:

```bash
uv run python .claude/skills/run-ptcg-battle/driver.py doctor
```

In a fresh git worktree two symlinks are missing (gitignored artifacts
that live only in the main checkout); doctor emits these lines with
resolved absolute paths:

```bash
ln -sfn <main-checkout>/models/il_agent models/il_agent
ln -sfn <main-checkout>/data/episodes data/episodes
```

Without `models/il_agent` battles still "run" — il_agent silently plays
`_safe_choice` every move and the results are junk. Doctor treats it as
battle-critical for that reason.

## Run (agent path)

```bash
uv run python .claude/skills/run-ptcg-battle/driver.py doctor    # ~5s
uv run python .claude/skills/run-ptcg-battle/driver.py battle    # ~10s
uv run python .claude/skills/run-ptcg-battle/driver.py forward   # ~7s
```

| command | what it does |
|---|---|
| `doctor` | verifies cg engine import, `make("cabt")`, torch device, checkpoint, episodes; prints exact fixes; warns if a training run is live (keep smokes on cpu). Exit 1 on battle-critical failure. |
| `battle` | plays real games. `--a/--b` any `benchmark_agents.py` name (incl. `il_agent@<deck>` arms), `--games N`, `--device cpu\|mps\|auto` (default cpu = evaluator parity + safe while MPS trains), `--save-json` / `--save-html` for the last game. Exit 1 if any game errors. Never touches the Glicko file. |
| `forward` | direct invocation, no env: encoder + `PTCGImitationPolicy.from_pretrained` forward pass over the first eval-split episode's decisions, prints top-1 match. The layer most model/encoder PRs touch. |

Typical output (numbers vary — no seed control exists, see Gotchas):

```
loaded il_agent vs rule_baseline in 3.0s (PTCG_DEVICE=cpu)
game 0: rewards=(1,-1) status=DONE steps=137 time=0.6s
il_agent vs rule_baseline: 1W-0L-0D, 0 errored / 1 games
```

## Watch a battle (visual replay)

The env ships a full replay player. Save it, serve it (`file://` can be
blocked), open in a browser — a scrubbable "Card Battle Visualizer"
showing hands, decks, prizes, actives per step:

```bash
uv run python .claude/skills/run-ptcg-battle/driver.py battle --save-html /tmp/replay.html
```

```bash
cd /tmp && python3 -m http.server 8642   # then open http://localhost:8642/replay.html
```

The first ~2 steps render a black board (deck submission has nothing to
draw) — scrub the timeline forward before concluding it's broken.

## Test

```bash
uv run python tests/test_il_pipeline.py   # ~8s → "all smoke tests passed"
```

## Gotchas

- **Missing checkpoint fails silently, not loudly.** il_agent's contract
  is never-crash: no `models/il_agent` → every decision falls back to
  `_safe_choice`, games complete, win rates are garbage. Run `doctor`
  after creating any worktree; measure with the `run-fallback-diagnostic`
  skill if a result looks off.
- **MPS is often busy.** Concurrent sessions train on it (doctor lists
  live `train_il`/`train_ppo` processes). `battle` defaults to cpu for
  this reason and for Kaggle-evaluator parity — don't "fix" it to mps.
- **`data/episodes/local/*.json` are not parseable episodes.** They're
  single-POV captures (steps are dicts with `obs`), good for
  `replay_episode.py --local` only. `iter_decisions` on them raises
  `KeyError: 0` — use split episodes via `resolve_split_dir("eval")`.
- **No RNG control.** The compiled engine exposes no seed (verified in
  `benchmark_agents.py`'s header) — identical commands produce different
  shuffles/outcomes. A 1-game smoke proves plumbing, never strength; for
  claims use `scripts/benchmark_agents.py` + the `leaderboard-check`
  skill.
- **`reports/glicko_ratings.json` compounds across runs.** The driver
  never writes it, but `benchmark_agents.py` does unless you pass
  `--no-glicko-persist` — don't let a throwaway comparison pollute the
  standing ratings.
- **Import-time log noise is normal.** `open_spiel_env ... Successfully
  loaded OpenSpiel environments: 41` INFO lines appear on every run;
  filter with `grep -v "INFO:"`.

## Troubleshooting

- **`KeyError: 0` in `iter_episode_decisions`**: you fed it a
  `data/episodes/local/` file. Those are single-POV; point at a split
  episode instead (the driver's `forward` already does).
- **Replay page loads but the board is black**: you're on step 0–2
  (deck submission). Click the timeline or step-forward; the board
  appears once play starts.

## When you outgrow the smoke

`scripts/benchmark_agents.py` (mirrored-pair round-robin, Glicko, the
real harness — `--games N` means 2N games), `scripts/eval_rung3_sanity.py`
(behavioral stats: declines/retreats/attacks), `scripts/fallback_diagnostic.py`
(is the bad score a bug or the model?), `scripts/eval_rung1.py` (offline
accuracy), `scripts/replay_episode.py` (step through a recorded episode).
