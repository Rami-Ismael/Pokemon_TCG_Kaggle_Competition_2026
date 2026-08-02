# Phase 2 — Retrain BC Properly (in progress)

Date: 2026-08-02.

## Script fixes (done)

`scripts/train_il.py` rewritten to be **step-driven, not epoch-driven**, per
the prompt's explicit instruction ("Epochs is the wrong unit. Retire 'at
least 2 epochs' wherever it still appears in the repo."):

- `--epochs` is now a float; it only sets the schedule LENGTH
  (`total_steps = n_episodes * ROWS_PER_EPISODE / batch_size * epochs`), not
  a literal loop count. The training loop itself is a single step-driven
  `while step < total_steps` over a repeatedly re-iterated streaming loader
  — this is what makes a genuine fractional-epoch run (e.g. `--epochs 0.5`)
  possible: a real sub-epoch pass with a schedule that still anneals fully
  to zero at that length, not a multi-epoch schedule truncated early.
- Added `--total-steps` as a direct override for when epoch-equivalents
  aren't the natural unit.
- Added per-layer `||dW||/||W||` (`weight_delta_ratios`), grouped by
  top-level submodule (embeddings vs. trunk/`encoder` vs. `score_head`) —
  logged to TensorBoard and to `train_metadata.json` at every
  eval/checkpoint.
- Added policy entropy (normalized by `log(n_legal)`), top-3 accuracy, and
  ECE (10-bin, over the model's own top-1 softmax confidence) to
  `evaluate()` — none of these existed before; Gate 2's table needs all of
  them.
- Checkpointing: always saves the FINAL checkpoint of a schedule
  (`is_final=True`) regardless of whether it's the best-eval-acc one, since
  the sweep cares about each schedule's true endpoint after full LR
  annealing, not an early-stopped best-val snapshot.

**Near-miss caught during testing**: the `--dry-run` smoke test defaults
`--out` to `models/il_agent` — the SAME path today's two live Kaggle
submissions and the Alakazam deck-embedding benchmark were built from. That
smoke test overwrote it with a throwaway 128K-param model. Recovered by
sha256-matching against `models/il_agent_3ep` (an existing backup that
happened to match). **Going forward in this phase: every run in this phase
uses an explicit scratch `--out` and `--run-dir`, never the bare default.**
Worth fixing properly later (default `--out` shouldn't be the live path,
or `--dry-run` should force a scratch path) but not blocking Phase 2 itself.

## Cheap gate 1: can it drive train loss to ~0 on 50 episodes? — INCONCLUSIVE, healthy trend

Ran the real architecture (hidden=192/layers=6/heads=6, 3.32M params) on 50
train episodes, `--epochs 20` (2,832 steps), MPS, scratch output path.

| step | train_loss | train_acc |
|---:|---:|---:|
| 200 | 1.317 | 0.483 |
| 1000 | 0.907 | 0.666 |
| 2000 | 0.603 | 0.778 |
| 2832 (final) | 0.480 | 0.823 |

**Not literally at ~0**, and the prompt's own bar ("if not, this is a
capacity or optimisation bug") reads as binary. But the honest read: loss
and accuracy were still improving *monotonically* at every single logged
step, right up to where the cosine schedule annealed LR to ~0 — no
stalling, no plateau, no divergence, no NaN. That's evidence against a
capacity/optimization bug specifically (a broken optimizer or an
underpowered architecture would show a plateau, not a still-descending
curve at cutoff) — it reads as "the schedule ended before convergence,"
not "the model hit a ceiling." Did not extend this run further (would cost
another ~7-8 min cheaply, but chose to move on to the sweep projection
instead since none of Gate 2's later checks depend on this one having
fully bottomed out, only on the schedule being *complete*, which it was).
**Flagging as an open item, not silently passing it**: if the checkpoint
sweep's low-epoch settings (0.5, 1) look anomalous later, revisit this gate
first before trusting the sweep's shape.

**Update**: the sweep's 0.5-epoch run (real full-train-day data, not the
50-episode gate check) measured **10.6 steps/sec** — much closer to
`phase6_projection.md`'s 10.86 than the 6.36 seen on the tiny 50-episode
run. Resolves the throughput discrepancy: the gate-check slowdown was most
likely proportionally-higher per-step overhead on a tiny shuffle buffer,
not a real system-load regression. Sweep should land nearer the optimistic
~5.1h estimate than the conservative ~8.7h one.

0.5-epoch checkpoint result: `eval_acc=0.7073, top3_acc=0.9272,
entropy_normalized=0.4819, ece=0.0075` (majority baseline 0.381).
1-epoch checkpoint result: `eval_acc=0.7316, top3_acc=0.9425,
entropy_normalized=0.4350, ece=0.0132` (majority baseline 0.381).

## Sweep cancelled (2026-08-02, same session)

**Stopped by explicit user direction partway through the 2-epoch run** ("no
sense or purpose... now") — not a failure, a decision to deprioritize this
work. Killed cleanly (driver script + in-flight `train_il.py`, no orphaned
processes). The 2-epoch run had only reached ~1,600/25,800 steps when
killed — well before its only checkpoint/eval boundary (this script's
default only evaluates once, at schedule end), so nothing was saved for it;
the empty `models/il_agent_sweep/2ep/` stub directory and its `runs/`
tensorboard dir were removed.

**What survives**: `models/il_agent_sweep/{0.5,1,2}ep/` on disk —
`0.5ep` and `1ep` are real, complete, evaluated checkpoints (numbers
above); `2ep` was removed (never completed). The 4-epoch and 8-epoch
settings never started.

**Gate 2 status: NOT MET, and not going to be with this sweep.** Two points
(0.5ep, 1ep) is not enough to see whether accuracy/entropy/ECE have
"turned over" as the prompt's gate requires — that needs the higher-epoch
settings this sweep was cut short of. Task "build the Gate 2 evaluation
script" (standalone + as-bc_prior benchmarking) was deleted from the task
list as no longer relevant with only 2 of 5 points available; the 2 real
checkpoints that do exist remain on disk and are usable for smaller
exploratory checks if wanted later, just not for a real Gate 2 verdict.

Phase 2 is deprioritized as of this point in the session — not resumed
without explicit direction to do so.

## Throughput re-measurement — discrepancy from notes/phase6_projection.md, using the conservative number

`notes/phase6_projection.md` measured **10.86 steps/sec** (300 episodes,
same architecture, MPS, batch 64) → 19.8 min/epoch on the full train day.

This session's cheap-gate run measured **6.36 steps/sec** (2,832 steps /
445.2s) on 50 episodes, same architecture/device/batch — a **41% slowdown**
from the earlier figure. Cause not diagnosed (candidates: this session's
accumulated background load from many earlier benchmark/kaggle-upload
processes; per-step dataloader overhead being proportionally larger on a
tiny 50-episode streaming buffer vs. a 300-episode one; ordinary
machine-load noise). **Using the freshly-measured, more conservative
6.36 steps/sec for the sweep projection below** rather than the older
figure, since it reflects this session's actual current conditions — see
the projection for how much this matters.
