# 2026-08-05 — recovering the s2v2 (and critic) checkpoints

Written because the recovery path is not reproducible from the repo: `models/**`
is gitignored, so **none of what follows is carried by git**. If this laptop's
`models/` is lost, this note is the map, not a backup.

## What was actually wrong

`tests/test_agent_benchmark.py::test_agents_load_and_play` failed on this branch
with `IL_MODEL_DIR=models/s2v2/e0_s42 does not exist`. The first read of that was
wrong twice over, and both errors are worth recording:

1. **"It fails on main too."** It does not — main passes. Main's checkout is
   *behind* this branch (`main` is an ancestor of HEAD) and predates the commits
   that added `agents/s2v2_arms/*`, so its registry has no s2v2 agents and never
   looks for the checkpoints.
2. **"The wrappers are stale, delete them."** The precedent (`768550e`) removed
   *never-trained* e1/e2b arms. These nine are not that: `e0_s43` is the Stage-2
   control arm that was submitted as `55246108` and settled at 320.4, and
   `models/s2v2/e0_s43` is the league opponent in all three self-play launchers
   (`run_selfplay_g{1,2,3}.sh`). Deleting the wrappers would have destroyed
   reachable, load-bearing work to make a test go green.

The checkpoints were never stale. They were **unreachable**: they existed only
inside other worktrees' local `models/` dirs and had never been placed in the
main checkout that every worktree symlinks from.

## Where they were found

| Artifact | Recovered from |
|---|---|
| `models/s2v2/{e0,e3,efb}_s{42,43,44}` (9 dirs, 114 MB) | `.claude/worktrees/rewrite-kaggle-pokemon-tcg-prompt-c69997/models/s2v2/` |
| `models/critic_search_prior` (13 MB) | `.claude/worktrees/ptcg-mcts-pipeline-port-a37b68/models/critic_search_prior/` |

Both were **copied** (not moved) into `<main>/models/`, leaving the source
worktrees untouched, then symlinked into this worktree.

## Verification

- **Provenance, not just presence.** `models/s2v2/e0_s43/model.safetensors`
  hashes to `bb0517c49a2c7ee5a4a878fd`, which matches the `checkpoint_sha256`
  the ledger recorded for submission `55246108` at submit time. The restored
  file is bit-identical to the artifact that scored 320.4 on the ladder.
- **Loadability.** All 9 arms load via `PTCGImitationPolicy.from_pretrained`
  at 3,318,721 params each — the documented 3.32M.
- `critic_search_prior` was the quieter problem: `mcts_il_agent` loads it
  *optionally* (`if Path(_CRITIC_DIR).exists()`), so its absence produced no
  error, just an mcts agent silently running with no critic.

## The root cause, and the guard added

`driver.py doctor` printed all-green throughout, because it only ever checked
`models/il_agent`. Eleven other checkpoint dirs the agent registry resolves went
unvalidated, and the load-all-agents test — which takes ~84 minutes — was the
only thing that noticed.

The doctor now statically extracts every `_REPO / "models" / ...` path from
`agents/**/agent_core.py` and reports any that are missing, with a runnable
`ln -sfn` fix naming the wrapper that needs it. It **warns** rather than failing
the exit code, preserving the doctor's existing "exit 1 = battle-critical"
contract — a worktree that only wants `battle` does not need the full roster.
Verified by removing a symlink and confirming the warning fires with the correct
fix line, then restoring it. It now reports 31 of 31 present.

## Not done, and worth a decision

These checkpoints exist on exactly one disk, in one un-backed-up directory. The
episode corpus has Hugging Face; submitted bundles have Kaggle. `e0_s43` is
recoverable from `55246108` if it comes to that, but the `e3_*` and `efb_*` arms
were never submitted and appear to have no copy off this machine. Pushing them
anywhere is an external action and was not taken.

---

## Update — the e3/efb arms now have an offsite copy

`Rami/ptcg-s2v2-arms` on Hugging Face, **private** (the ladder is live until
08-16 and these are trained competition agents; it must not be made public
before the competition closes).

Contents: `{efb,e3}_s{42,43,44}`, 76 MB, each with `model.safetensors`,
`config.json`, `train_metadata.json`. The README carries the recipe, the
per-seed offline numbers, and the sha256 of every checkpoint.

**Verified by restoring, not by uploading.** `snapshot_download` into a scratch
dir, then sha256 against the local files: all six bit-identical. A backup that
has never been restored from is a hypothesis.

What the pair is worth keeping for: `e3` is `efb` plus one flag
(`--skill-min-score 1189.0`), so it is a clean controlled comparison of the
skill gate. Offline, on the held-out day, 3 seeds each:

| Arm | top-1 (mean ± sd) | top-3 | ECE |
|---|---|---|---|
| `efb` | 0.7475 ± 0.0023 | 0.9545 | 0.018 |
| `e3` | 0.7263 ± 0.0043 | 0.9461 | 0.054 |

The gate costs ~2.1 points of top-1 and triples calibration error, ~6 pooled SD
— reproducing the standing `skill-filter-negative-result`. Neither arm has a
ladder read, so neither is "better" in the sense this project reserves that word
for.

**Still unbacked:** `e0_s42` and `e0_s44`. `e0_s43` is recoverable from its
Kaggle submission (`55246108`); the other two were never submitted and carry the
same single-disk exposure these six had before today.
