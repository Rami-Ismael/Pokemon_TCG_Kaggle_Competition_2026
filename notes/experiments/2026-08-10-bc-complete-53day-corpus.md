# BC on the complete 53-day corpus — fourth point on the data-scaling curve

Design: **controlled comparison + scaling-study point** (same shape as the
2026-08-04 card — we are extending the same curve, not starting a new one).
Proposed by Rami 2026-08-10: "train a new powerful behavior cloning model
because now we have more data than the last time … the coverage will be much
higher and better usage for training RL agents."

**Premise correction (recorded, not a criticism):** the latest BC model is NOT
the tiny-corpus one. `il_agent_v3_alldays` (trained 08-07→08-08) already used
**210,512 episodes / 48 days**. What is new since then: the backfill finished
(commit bb4a57f) — the hub now holds **264,495 episodes / 53 days
(2026-06-16 → 2026-08-07)**. The real delta this experiment buys is
**~+49k train episodes (+23%)** and, more importantly, the days
**07-24, 08-05, 08-06, 08-07** — the three most recent meta days. The 08-04
experiment's only offline gain on this architecture coincided with adding
recent days (recency mechanism), so the delta is small in dex but targeted.

- **Hypothesis:** fresh BC on all 52 train days (~259k episodes) matches or
  beats `il_agent_v3_alldays` offline and beats it against the external field,
  because the added days cover the meta the field currently plays
  (recency), on top of +23% coverage.
- **Independent variable:** training corpus only — 210,512 eps / 48 days →
  ~259k eps / 52 days. Held fixed: architecture 192/6/6 (3.32M params), fresh
  init, 1.0 epoch-equivalent (same samples-to-convergence protocol as the
  baseline; equal-step matching rejected per standing feedback), lr 3e-4,
  batch 64, warmup 200, grad clip 1.0, seed 42, eval-every-steps 10000,
  eval day 2026-07-27, same encoder, no feature groups.
- **Baseline:** `models/il_agent_v3_alldays` — eval acc **0.7342** / top-3
  0.9539 / ECE 0.0487 on held-out 07-27 (eval_batches=100 protocol, 6,400
  rows). Never pool-benchmarked or submitted (0 ledger entries) — the pool
  comparison below is new for BOTH models.
- **Metric & protocol:**
  1. Offline acc on 07-27, identical protocol. NOTE: 07-27 is now 2 weeks
     stale relative to the meta, so it CANNOT see the recency mechanism —
     recorded, not a decision metric. Expect ≈flat.
  2. Fallback diagnostic must be clean (hard rule) before any comparison.
  3. Decision metric: pool benchmark (ordered by Rami 2026-08-10, chained via
     scripts/run_bc_alldays52_pool_benchmark.sh): star mode vs the FULL
     92-agent pool (`--agents all --focus`), 8 games/opponent (~736 games),
     then the IDENTICAL star for `il_agent_v3_best` as the paired baseline.
     Checkpoint selection mirrors v3: best-eval snapshot if it beats the
     annealed final (v3's best was +1.9pp over its final). CPU inference,
     and the benchmark waits for an idle machine so search-based pool
     members get honest think time.
  4. Ladder only after (3), per the benchmark-pool-before-submitting rule.
- **Pre-registered decision (Claude's proposal — Rami may amend):** adopt as
  the BC base for RL pipeline v5 if offline acc ≥ baseline − 0.005 AND pool
  win-rate ≥ baseline's (intervals overlapping counts as adopt, since the
  corpus superset costs nothing). Drop only if it is clearly worse on the
  pool. Registered prediction: offline ≈flat (the stale eval day is blind to
  recency); any real gain shows only against the field or on the ladder.
- **Cost estimate (launch census, measured):** 260,065 train episodes /
  52 days → 736,715 steps ≈ **15 h MPS** (v3 measured 13.6 steps/s);
  ~13 MB checkpoint; disk 45 GB free vs ~28 GB gate, HF cache at 11 GB;
  MPS uncontended at launch (verified — only orphaned niced PPO worker
  processes from 08-05/06, ~0% CPU). Single seed 42 — provisional until a
  second seed is justified by the result.
- **Scale confound note:** this is a POSITIVE-direction data-scaling arm, so
  the confound argues FOR it, not against it. If the result is null, the
  claim is "the last +0.1 dex of THIS corpus bought nothing", not "data
  doesn't help" — the corpus ceiling is Kaggle's publishing, not our choice.
- **Prior work checked:** 2026-08-04-il-full-corpus-ladder.md (its result is
  now in: 55248985 settled **418**, i.e. real-but-modest per its own bars,
  not the ≥500 "dramatic" bar); runs/il-v3-alldays.log (throughput, corpus
  census); scaling points so far — 4,554 eps→0.7534 · 9,820→0.7527 ·
  15,032→0.7583 (127k-step 3-epoch protocol, not comparable to the 1-epoch
  points) · 210,512→0.7342 (1-epoch protocol); model-size memory (log-linear
  +1.7pt/doubling → a 6.6M arm on this same corpus is the natural NEXT
  experiment, kept out of this one: one variable).

## Result (2026-08-10)
- **Observed:** training finished cleanly: 736,715 steps / 16.3 h. Offline on
  held-out 07-27: best ckpt (step 620k) **0.7619**, final 0.7606 — above the
  registered bar (v3_best 0.7528, v3_final 0.7342) and the highest offline
  point on the curve. Fallback diagnostic clean (0/392, 0 illegal). Pool
  stars (91-agent bed, 8 games/opponent, 720 games each, single run):
  · bc_alldays52: field win **72.1%** [68.7, 75.2], Glicko-this-run 1727 RD 30
  · il_agent_v3_best: field win **70.3%** [66.8, 73.5], Glicko 1619 RD 30
  Wilson intervals OVERLAP → not separable on field win rate; Glicko
  intervals barely overlap (1667–1787 vs 1559–1679) → still not called.
  Head-to-head across both stars: new model 9/16 — noise-level sample.
  Commands: scripts/run_bc_alldays52_pool_benchmark{,_resume}.sh.
- **Decision:** **ADOPTED** as the BC base per the pre-registered criterion
  (offline ≥ baseline AND pool ≥ baseline with overlap counting as adopt —
  the corpus superset costs nothing). NOT a "beats v3" claim: local pool
  numbers, overlapping intervals, single seed, and the local pool has
  inverted against the ladder before. Ladder adjudication only if/when
  Rami orders a submission.
- **What we learned:** the fourth data point moved offline accuracy UP
  (+0.9pp over v3_best at same protocol) — the second time recent-day data
  coincided with an offline gain — but field strength stayed within noise
  of the 48-day model: at this corpus size, +23% more episodes buys offline
  fit faster than it buys wins. Transferable lesson: pre-registering
  "overlap counts as adopt" avoided both overclaiming a win and wasting a
  free upgrade.

### Raw pool-benchmark record (auto-appended 2026-08-10 20:50)
- wired checkpoint: best step 620000, eval acc 0.761875 (final was 0.760625)
- fallback diagnostic: clean, 0/392 fallbacks, 0 illegal (reports/fallback_diag_bc_alldays52.json)
- bed: 91 agents; EXCLUDED grid_medium_comb + il_alldays_equalsteps (checkpoints lost -- searched worktrees + HF backup, 0 hits). Any number quoted from this run carries that caveat.
- star results: reports/pool_star_bc_alldays52.json vs reports/pool_star_il_agent_v3_best.json
  (full log: runs/bc_alldays52_pool_benchmark.log)
