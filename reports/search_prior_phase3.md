# Phase 3: our IL policy as the MCTS prior — experiment report

Date: 2026-08-04. Design: piKL-shaped search (Jacob et al., arXiv:2112.07544)
— the engine's official Search API with `PTCGImitationPolicy` (models/il_agent,
3.32M params) as child prior; leaf values from an S2-E4-style offline critic.
Code: `src/pokemon_tcg/search_prior_mcts.py`, agent `agents/mcts_il_agent/`.
Deck held fixed (il_agent's Grimmsnarl list) — every comparison is
policy vs policy+search, no deck confound. SEARCH_COUNT=30 (Phase-2a safe
point). All benchmark games via `scripts/benchmark_agents.py` mirrored pairs,
`--no-glicko-persist`, Wilson 95% CIs.

## Arms and results

Head-to-head vs plain `il_agent` (the base policy itself):

| arm | result (mcts side) | win rate [95% CI] | fallbacks |
|---|---|---|---|
| prior-only, run A | 29–11 (n=40) | 72.5% [57.2, 83.9] | 0/5,359 |
| prior-only, run B (replication) | 51–28, 1 draw (n=79 decided) | 64.6% [53.6, 74.2] | 0/10,411 |
| **prior-only, pooled** | **80–39** (n=119) | **67.2% [58.4, 75.0]** | 0 |
| prior + critic (seed-42 critic) | 21–19 (n=40) | 52.5% [37.5, 67.1] | 0/5,285 |

Self-play controls (il vs il, mcts vs mcts) stayed seat-balanced (8–12,
9–11, 12–8, 10–10, 24–16) — no harness asymmetry.

**Finding 1: search with our IL prior beats the IL policy it is anchored to,
67.2% [58.4, 75.0] pooled, CI excluding 50% in both independent runs.**
With no useful leaf value (see below), the gain must come from terminal-node
lookahead (forced win/loss detection within 30 sims) plus the combo-level
treatment of multi-selects.

**Finding 2: the critic ERASES the gain instead of adding to it.** The
seed-42 critic failed its own calibration gate before the benchmark: eval
MSE 0.2660 vs 0.2500 for the constant-0.5 predictor (held-out day, 2,171
rows) — worse than a coin flip, despite train MSE 0.17 (it memorized its
single 1,490-episode training day). A leaf evaluator that is noise pulls
search away from the real terminal signals. The arm behaved exactly as the
failed gate predicted. Per the ≥3-seeds rule this critic result is a single
seed — but the calibration failure, not the benchmark, is the load-bearing
evidence, and it makes more seeds of the SAME configuration pointless: the
configuration (one day of data, tiny eval set) is what failed.

## Bug history (both runs before the fix are VOID)

The first benchmark pair silently answered 12.76% of mcts decisions with
`_safe_choice`: decisions with ≥ MAX_OPTIONS(48) options have no DECLINE
slot, and the prior mapping read `logits[n_real]` out of bounds
(IndexError → never-crash fallback). Fixed (neutral prior for the empty
selection when the slot is unavailable), verified 0 fallbacks under the
reproducing seed, and both arms rerun clean. Lesson recorded: the fallback
instrumentation (run-fallback-diagnostic) is what caught this — a
plausible-looking 42.5% result would otherwise have been reported as "critic
doesn't help" for the wrong reason.

## Timing rehearsal (forced-CPU, evaluator envelope)

Quiet-machine, `PTCG_DEVICE`-independent (agent hard-forces cpu +
`torch.set_num_threads(1)` at import): prior-only at N=30 runs
**209 ms/decision mean, 358 ms p95, 441 ms max**; benchmark games averaged
7.8–9.2 s/game for the mcts side (~35–70 own decisions/game).

Budget: cabt has NO per-step timeout (`actTimeout: 0`); each agent draws
from a **600 s per-match overage bank** (verified in real ladder episodes;
typical agents consume 13–47 s). Worst case ~100 decisions × 0.44 s ≈ 44 s
per match, ×3 evaluator CPU-speed safety factor ≈ **132 s ≪ 600 s. PASS.**
RAM: one 3.3M-param float32 policy (~13 MiB weights) inside the ~197.7 MiB
envelope — the prior-only configuration ships no critic.

## Gate verdict and recommendation

The task's gate — beats plain il_agent locally AND survives the forced-CPU
timing rehearsal — is MET by the prior-only configuration. It is technically
a submission candidate. **Recommended action: do NOT submit it now.**

Ladder statement (read 2026-08-04, `scripts/check_leaderboard.py`): we are
rank **2380/6266** at team score **696.0** (active set: 55228113 same-build
resubmit of the 804.0 baseline, reading 688.6, + 55246108 at 406.7;
best-ever 804.0 is already displaced). Locally, mcts_il_agent beats il_agent
at 67.2% [58.4, 75.0] — but il_agent itself reads ~397–400 on this ladder,
far below the 688.6 active anchor, and a new submission would displace that
anchor from the 2-slot active set. A ~170-Elo-equivalent local gain on a
~400-rated base does not clear 688. The 2026-08-02/03 inversions also say
pool-internal gains routinely fail to transfer. Submitting this would most
likely repeat the 804→395 displacement incident.

## What this unlocks instead (next experiments, in value order)

1. **IL-prior search on the 804-family base**: the active-set anchor
   (agent_core_improved + BC prior + UCB1 search) already searches, but NOT
   through the official Search API with determinized lookahead. Porting its
   stronger prior into this scaffold attacks the team score directly.
2. **A critic that passes calibration**: multi-day training data (needs
   outcome/meta support in ShardILDataset — currently local-only), a real
   eval set (the 12-episode local eval day is too small to trust), 3 seeds.
   Only then rerun the prior+critic arm.
3. **Stage-3 rollouts via the direct battle API** (Phase-2b: ~10–15× serial
   speedup over env.run) — orthogonal to search, biggest training-loop win.
