# Run prompt — execute RL Pipeline v4

Paste this into a fresh Claude Code session. It assumes nothing from the design
conversation.

---

Execute `prompts/rl_pipeline_v4.md`. Read it in full before you touch anything —
it is the spec, and it already contains the measurements, so do not re-derive
what it settles.

Load the `ptcg-repo-context` and `leaderboard-check` skills first. Then
`git fetch && gh pr list` — concurrent sessions merge work mid-task, and stale
main has already cost an hour of duplicated effort in this repo.

## Order of operations

**1. Standing daily item, before anything else, every session.**

```bash
uv run python scripts/submission_ledger.py refresh && uv run python scripts/check_leaderboard.py
```

Update the §9.7 results table in `rl_pipeline_v4.md` with anything that moved.
This is not optional and it is not a formality — the ladder has contradicted a
local claim four separate times, and §9.7 is where that record lives.

**2. Read the §0 kill-gate.** The gate is the settled read of `55253900`
(`selfplay_g1_ref430k`), due ~2026-08-08. Everything in Phases 2–3 is
conditional on it. Do not start a generation-4 training run before the gate
resolves — g2 and g3 already burned 2.5M steps producing zero promotions.

- **≥ 550 → PASS.** Execute work items 2→5 in §12, in order.
- **< 450 → FAIL.** Stop Phases 2–3. Execute §0.1 instead, and write up the
  negative result in `notes/experiments/`. A clean negative result is the
  deliverable in that branch, not a consolation prize.
- **450–550 → ambiguous.** One more read on 08-10, then decide. Do not run a
  third generation on an ambiguous gate.

**3. Before the gate resolves**, only these are unblocked — do them now:

- **Work item 7 (§12), free and parallel:** pre-register the local pool's
  predicted ladder ordering for the next five submissions into the ledger's
  `expects` field. This is the only thing that can turn the pool's in-sample
  ρ = +0.929 into a defensible gate, and it costs nothing.
- **The logging half of work item 2:** add a second frozen reference so
  `kl_to_prior` is logged against **both** the current promoted best and the
  original IL prior. Today only the former is logged, which makes "has it
  forgotten the human prior" literally unanswerable from the logs after the
  first retarget. Pure instrumentation, correct under either gate branch.
- Nothing else. Do not tune, sweep, or launch training against an unresolved
  gate.

## Rules that override your defaults

- **Do not submit to Kaggle without asking.** Report what you would submit and
  why, then wait. Slots are not scarce (43 readable remain); settled reads are.
- **Never submit two experiments back-to-back.** Team score is the MAX of the
  latest 2 submissions, so an experiment is free only while the other active
  slot holds a strong build. Two in a row is how this team fell 804.0 → 395.0
  and ~4,500 ranks on 08-03.
- **No "better", "best", "beats", or "improves" about any agent without a
  ladder read that supports it.** Local Glicko is a hypothesis about the ladder,
  not a result. If the ladder contradicts a local number, report the
  contradiction — do not restate the local claim.
- **Never point a trainer at the local raw split dirs.** They are stubs;
  `iter_decisions` returns near-nothing *silently*. Stream from Hugging Face:
  `--data-source auto --num-workers 4`.
- **Never delete or overwrite raw episode data without asking.** The HF dataset
  is the only copy of the corpus.
- `resolve_device()` only, no CUDA branch. `torch.set_num_threads(1)` first.
  Everything under `uv run`. Paths from `pokemon_tcg.config`. Seed 42.
- `num_envs == num_workers`, always — the cg engine is a per-process singleton.
- Long runs do not stop for approval. Report the projection, chain runs so MPS
  is never contended, nice everything, cap workers, keep the laptop responsive.
- Every comparison: ≥3 seeds, equal **steps** (never equal epochs), an RD or σ
  on every number, a chart in `reports/figures/`, the control beside the claim.

## Stop conditions — halt and report, no approval needed

- Opponent-private information reaching the encoder from the rollout path.
- `kl_to_prior` > 0.6 sustained over 5 updates (the anchor stopped binding).
- Any SelectContext with n ≥ 100 and entropy < 0.05. **This already tripped in
  g1 and went unread** — fix the entropy coefficient before restarting, don't
  log it and continue.
- Two consecutive retargets with the gate diagnostic below 0.55.
- Any design that wants to write an episode corpus to disk.
- Bundle over the ~197.7 MiB envelope.

## What "done" looks like for a work item

A chart in `reports/figures/`, the control beside the claim, a σ or RD on every
number, and — if it produced a checkpoint — a ladder read recorded in §9.7
before the next item starts. An item with a local number and no ladder read is
not finished; it is a hypothesis.

If a measurement contradicts `rl_pipeline_v4.md`, say so plainly with the
numbers and update the doc. The doc is a plan, not a result.
