# Phase 6 — Time & Cost Projection

Measured, not estimated: 300 train episodes, hidden=192/layers=6/heads=6
(3.32M params, the Phase 1 target config), batch=64, MPS, on this machine.
850 steps in 78.3s → **10.86 steps/sec**, 54,400 rows processed → **181.3
rows/episode** (this build's row count per episode is higher than Phase 0's
718,122-row figure implies, because it now also includes decline rows and
the multi-select autoregressive unroll rows added in this pass).

Projected to the full train day (4,554 episodes):
- Rows: 4,554 × 181.3 ≈ **825,900**
- Steps/epoch at batch=64: ≈ **12,900**
- **Wall-clock per epoch: 12,900 / 10.86 ≈ 1,188s ≈ 19.8 minutes**

This is a projection from a 300-episode sample to the full 4,554, on the
same machine, same config, same batch size — not a different-hardware
extrapolation.

**Decision, per this project's own stop condition ("any training run
projected to exceed 1 hour" requires approval before launching):**
- **1 epoch over the full train day (~20 min) is safely under the 1-hour
  bar** — running this now as the capped validation run for Phase 4, with
  eval capped at 100 batches (6,400 rows) rather than the full eval day, to
  keep the eval pass itself fast (seconds, not minutes).
- **The default `--epochs 3` (≈59–65 min once eval overhead is included)
  sits right at or over the 1-hour line — not launching that without your
  explicit go-ahead.** If 1 epoch's Rung 1 numbers look promising, the next
  ask would be for approval to run 3+ epochs.

Anchor from your own prompt: a comparable IL attempt did 21k games in 3–4h
on one H200. This corpus is ~9k games (4,554 + 4,430) on an M4 Pro MPS
backend — roughly half the games, on consumer-grade (not datacenter)
hardware, and the measured 20 min/epoch here is consistent with that being
a reasonable, not-bottlenecked number rather than a dataloader-bound one.
