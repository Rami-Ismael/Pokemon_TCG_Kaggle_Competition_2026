# RL Pipeline v1 — PRIOR → REWEIGHT → SELFPLAY

The successor plan to `prompts/bc_pipeline_v2.md`. That prompt built Stage 1;
this one names and specifies the full three-stage progression, modeled on
**"Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning
with Transformers"** (Grigsby et al., RLC 2025, arXiv:2504.04395 — the
"Metamon" paper). Their recipe, adapted to this repo:

1. Behavior-clone a policy from a large corpus of human replays (**the prior**).
2. Improve it with offline RL on the *same* data — advantage/outcome-weighted
   and filtered BC-style objectives, with a critic used at train time only
   (**reweighting**).
3. Improve it further with self-play against itself, past versions, and the
   public opponent pool (**self-play fine-tuning**).

⚠️ Provenance discipline (repo rule 5): steps 1–2 follow the paper. Step 3
**deliberately deviates from it**. The paper's third step aggregates self-play
trajectories into an ever-growing offline corpus and retrains offline RL on
the union — that requires corpus storage, and this laptop has **~16 GB of free
disk** against a human dump that is already ~40 GB. Storing generations of
self-play episodes is not possible here. Stage 3 is therefore **on-policy PPO**:
rollouts live in RAM, are consumed by the update, and are discarded; nothing
persistent is written except checkpoints and logs. This is also what the
original roadmap (bc_pipeline_v2 header: "Stage 3 self-play / synthetic
fine-tuning (PPO)") and the PufferLib-not-Ray decision (2026-08-01, memory
`stage3-selfplay-pufferlib-not-ray`) were already scoped for. What the
deviation costs, stated honestly: offline aggregation lets every retrain see
all past data, so the human prior cannot be forgotten; PPO sees only the
current policy's states, so forgetting the prior is a live failure mode — the
KL anchor in §3.2 is the guard, and per-context KL/entropy logging is how the
guard is watched, not assumed.

---

## Stage map

| Stage | Name | Input | Output | Gate to pass |
|---|---|---|---|---|
| 1 | **PRIOR** | human episodes (train-2026-07-26) | frozen BC checkpoint `models/il_agent/` | DONE — Rung 1 vs 0.381 majority baseline; Rung 2 vs pool |
| 2 | **REWEIGHT** | same human episodes + `manifest.csv` ratings | weighted/offline-RL checkpoint `models/rw_agent/` | beats PRIOR at Rung 2 (non-overlapping Glicko intervals, ≥3 seeds) AND on the real Kaggle ladder |
| 3 | **SELFPLAY** | Stage-2 winner + live cabt rollouts (in-memory, nothing stored) | PPO checkpoints `models/ppo/<step>/` | each promoted candidate beats the current best by the same two-part gate; loop until it stalls |

Standing rules inherited from `bc_pipeline_v2.md` and CLAUDE-level repo rules —
these are not restated per stage, they apply everywhere:

- Device via `resolve_device()` only; no CUDA branch exists. Training on MPS,
  the evaluator is CPU-only (~1.6 vCPU, ~197.7 MiB — unverified envelope,
  design to it anyway). `torch.set_num_threads(1)` before heavy torch work.
- Everything under `uv run`. Paths from `pokemon_tcg.config`. Seed 42.
- Action masking is structural (Pattern-B option-scoring head) — never a path
  that can emit an out-of-range index.
- No opponent-private information reaches the encoder. Stage-3 rollouts feed
  the encoder **the acting agent's `obs_dict` as served by the env** — the
  same view the ladder serves — never the engine's omniscient state.
- Every comparison: ≥3 seeds, equal **steps** not equal epochs, an RD or σ on
  every number, a chart in `reports/figures/`, and the control beside the claim.
- Local Glicko and the real leaderboard have already diverged once
  (2026-08-02). No checkpoint is called "better" on local numbers alone; the
  gate always includes a real submission with a detailed submit message and a
  leaderboard read after scoring.
- Any run projected over 1 hour stops for approval first (Phase-6 rule).
- **Disk budget: ~16 GB free.** No stage may write an episode corpus. The
  allowed persistent artifacts are checkpoints (~13 MB fp32 at 3.32M params),
  TensorBoard logs, figures, and a handful of Rung-3 transcript episodes
  (MBs). Anything else that wants disk is a design error — redesign, don't
  compress.
- Ladder closes **2026-08-16**; ratings normalize only over hundreds of games
  (~48 matches/day). A checkpoint submitted with fewer than ~3 days of ladder
  time left cannot be confidently evaluated — plan the last submission slot
  backward from that.

---

## Stage 1 — PRIOR (built; what it owes the later stages)

Status: complete. `scripts/train_il.py` → `PTCGImitationPolicy` at
`models/il_agent/`, hidden=192/layers=6/heads=6, 3.32M params, 2.85 ms/decision
CPU fp32 (`notes/phase1_decisions.md`).

What Stages 2–3 consume from it — verify each before starting Stage 2:

1. **A frozen, independently loadable checkpoint** via `.from_pretrained()`,
   never mutated again. It is (a) the warm-start for Stage 2, (b) the KL
   anchor for Stage 3, (c) the control arm in every experiment below.
2. **Calibrated probabilities over the option set**, not just argmax — the
   weighted objectives and the KL penalty both need the full distribution.
   If per-context entropy was never logged, log it once for the frozen model
   as the reference line.
3. **The encoder version.** Stages 2 and 3 use the *same*
   `encode_observation()` unchanged. Any encoder change forks the pipeline
   version and invalidates cross-stage comparisons.

---

## Stage 2 — REWEIGHT (offline RL / weighted BC on the human corpus)

The paper's finding, in one line: the same replay data trains a meaningfully
stronger policy when rows are weighted or filtered by outcome instead of
cloned uniformly. This is E2 from the bc_pipeline_v2 register, promoted to a
full stage.

### 2.0 Data plumbing (blocking, do first)

`iter_decisions()` yields only `(obs, chosen_index, exclude)` — no outcome, no
seat, no rating. Extend it (or add a parallel `iter_decisions_weighted()`) so
every row also carries:

- `outcome` — the acting seat's terminal result from the episode's `rewards`
  pair. ⚠️ Remap before use: cabt ships −1/0/1 and bc_pipeline_v2 §8.4 records
  why negative terminal rewards are a known pathology magnet; use a {0, 0.5, 1}
  loss/draw/win mapping and keep the raw value out of every objective.
- `seat` (agent_idx) — first-player advantage is real; it is already an
  encoder scalar question, here it is needed for per-seat diagnostics.
- `episode_id` — join key to `data/episodes/manifest.csv`.
- `avg_score` / `min_score` from the manifest — the player-rating field, the
  skill-filter lever. Load the manifest once into a dict, not per row.
- `turn_index` — enables late-game upweighting diagnostics; cheap to carry now.

`ILDataset` passes these through as tensors; `collate` unchanged otherwise.
Acceptance: a dry-run epoch prints the outcome distribution (should be ~50/50
by construction) and the avg_score histogram, saved as a figure.

**STATUS 2026-08-02: built** — `DecisionMeta` in `il_dataset.py`,
`ILDataset(with_meta=True)`, `load_manifest_scores()`, acceptance script
`scripts/check_weight_plumbing.py`, tests green. A hard `winner_only` filter
(S2-E1's mechanism) also exists: `iter_decisions(..., winner_only=True)` /
`train_il.py --winner-only`. Two measured findings:

- 500-episode check: seat outcomes perfectly complementary (500 win / 500
  loss seats); winner seats contribute ~52.9% of rows — winners take slightly
  more decisions per game, so even "unweighted" BC has a mild winner tilt.
- ⚠️ **S2-E3 is blocked on the train day**: `manifest.csv` covers only
  2026-07-01 and the eval day 2026-07-27 — **zero rows for train-2026-07-26**,
  so `avg_score` is the −1.0 sentinel for the entire training corpus. To
  unblock E3, fetch the train-day manifest from the Kaggle episodes dataset
  (small CSV; needs `kaggle auth login`). E0/E1/E2 are unaffected.

### 2.1 The objective ladder — register as experiments, pick by measurement

All arms warm-start from the PRIOR checkpoint, train the same step count, same
LR schedule, 3 seeds each. Loss stays cross-entropy over the legal option set;
the arms differ only in the per-row weight `w`:

| ID | Arm | Weight `w` | Notes |
|---|---|---|---|
| S2-E0 | control | 1 (plain BC, continued) | separates "more steps" from "better objective" |
| S2-E1 | filtered BC (winners-only) | 1 if won else 0 | the paper-adjacent simple baseline; halves the data — equal-steps rule matters here |
| S2-E2 | outcome-weighted BC | `exp(β·(outcome − b))`, β tuned over {0.5, 1, 2}, b = 0.5 | AWR-shaped, keeps losing data at reduced weight |
| S2-E3 | skill × outcome | S2-E2 weight × 1[avg_score ≥ Q75] | tests whether the manifest rating field pays; Q75 threshold is a starting point, chart the sensitivity |
| S2-E4 | critic-advantage weighted | `exp(A(s)/β)`, A = outcome − V(s) | only if E1–E3 plateau; adds a value head trained on outcome, **train-time only — the shipped bundle contains the actor alone** (CPU + 197.7 MiB envelope). If built, this value head is the natural critic init for Stage 3 |

Run order: E0, E1, E2 first (the two most-different arms plus control, per the
old E2 rule). E3 and E4 are gated on E1/E2 showing signal. Do not build all
five before measuring the first three.

Known trap, stated up front: winners-only filtering can *lower* Rung-1
accuracy while raising Rung-2 win-rate — offline action-match is a pipeline
check, not a selection metric (Orbit Wars 49th, already accepted in this
repo). Selection is on Rung-2, confirmed on the ladder.

**STATUS 2026-08-02: E0/E1/E2 trained (3 seeds each, equal 12,900 steps, warm
LR 1e-4) and Rung-2 benchmarked.** Rung 1: three-way tie (75.8% ± 0.1 every
arm, PRIOR 75.3%) — as predicted, offline accuracy didn't separate them.
Rung 2 (15-agent round-robin, 12 mirrored pairs/pairing, seeds pooled,
charts `s2_e0e1e2_rung1.png` + `s2_rung2_arms_vs_prior.png`):

- Head-to-head vs PRIOR: E0 62.0% [50.3,72.4], E1 62.5% [51.0,72.8],
  E2 56.9% [45.4,67.7] — **every arm beats PRIOR**; E0/E1 significantly.
- Arms do not separate from each other (overall 47.9/49.0/45.0, overlapping
  CIs). E1 nominally first with half the training rows.
- Reality check: all IL checkpoints still lose ~92% to the strong public trio
  (plamen06_steel 86.7%, mechi22_alakazam 83.1%, kiyotah_dragapult 80.8%
  overall). Stage 2 moved the IL line, not the league table.

Ladder: `s2_e1_s43` (E1's best checkpoint, user-selected) submitted 2026-08-02
as Kaggle submission 55196434 (14 MiB bundle, forced-CPU rehearsal 4.4 ms/dec)
— **first read 516.7 vs the Stage-1 il_agent's settled 400.0** (submitted
2026-08-01). Provisional until ~3 days of ladder games settle it; if it holds
above 400, the Stage-2 gate is met and `models/s2/e1_seed43` is the Stage-3
(SELFPLAY PPO) initialization. Reference: the heuristic `agent_core_improved`
lineage sits at 719–804 on the same ladder — the IL line is climbing but not
leading.

### 2.2 Stage-2 gate

1. Rung 1 per-context accuracy vs the 0.381 majority line (pipeline check only).
2. Rung 2 round-robin via `scripts/benchmark_agents.py` — the winning arm vs
   PRIOR and the public pool, enough mirrored pairs that the Glicko intervals
   separate; quote RD.
3. Build the bundle (`scripts/build_improved_submission.py` pattern — read the
   printed tarball MiB), submit, detailed message, read the leaderboard after
   scoring. The Stage-2 winner becomes the **Stage-3 initialization** only
   after the ladder confirms it.

---

## Stage 3 — SELFPLAY (on-policy PPO fine-tuning, nothing stored)

On-policy replaces the paper's offline aggregation for the disk reason in the
header. The learner improves by playing live games; every trajectory is used
once by the PPO update and freed. Disk sees only checkpoints and logs.

### 3.1 Rollout engine

- **Parallelism: game-level stdlib multiprocessing** (`pokemon_tcg/selfplay.py`).
  **DEVIATION, 2026-08-02, stated loudly:** the recorded decision said
  "PufferLib emulation + multiprocessing backend." The substance (multiprocessing,
  never Ray) is kept; the PufferLib emulation wrapper is not, because cabt is
  callback-driven (`env.run([a, b])`, the tested benchmark path), a game is
  ~0.9 s with ~3 ms policy calls (env-bound, not inference-bound), and forcing
  a two-player turn-based env into a Gym step() mold buys batched inference we
  don't need. PufferLib remains the fallback if step-level vectorization ever
  becomes the bottleneck.
- **First measurement (done 2026-08-02, `scripts/probe_selfplay_throughput.py`):**
  8 workers on the M4 Pro: **6.2 games/sec mirror, 425 rollout rows/sec
  (~1.5M rows/hour)**; league mix 5.4 games/sec. Mean game ≈ 68 rows/seat,
  0 fallback decisions in 176 probe games. A 4096-row PPO update collects in
  ~10-15 s. Gotcha recorded in code: kaggle_environments counts a bound
  method's `self` in `co_argcount` — every callable handed to `env.run` goes
  through `as_env_agent()` or it errors instantly and scores as a crash-loss.
- **Rollout buffer sized to RAM, not disk** — a config knob (e.g. 2048–8192
  decisions per update), tuned to keep the collect/learn cycle busy, never
  spilled to disk.
- **Opponent sampling per episode** (league play, the collapse guard): the
  learner side is always the current policy; the opponent is drawn ~50%
  current weights (mirror), ~30% frozen past checkpoints (league), ~20%
  public pool agents from `AGENT_FILES`. Pure mirror self-play overfits to
  its own blind spots. A league of 10 frozen checkpoints costs ~130 MB —
  cheap under the disk budget. The mix is a knob, not a finding — log it.
- **Exploration at rollout**: sample from the policy (temperature ~1.0), do
  not argmax — argmax self-play produces near-duplicate games. Record the
  temperature in the run manifest.
- **Masking**: structural Pattern-B masking stays in the rollout *and* in the
  loss. Huang & Ontanon (arXiv:2006.14171): masking changes the gradient — it
  is part of the objective, not a filter bolted on after.
- **Deck confound warning (repo-level, standing):** deck is bound to agent
  identity in this harness. All Stage-3 games use the frozen control deck
  unless a deck axis is deliberately added first. Stage 3 improves *this
  deck's* policy; say so in reports rather than implying general strength.

### 3.2 PPO specifics — the bc_pipeline_v2 §8 RL-readiness notes, now due

- **Init**: actor from the Stage-2 winner. Critic head fresh (or from S2-E4's
  value head if that arm was built), trained on the remapped terminal reward.
- **Reward**: terminal only, win = 1, draw = 0.5, loss = 0 — the §8.4 rule.
  Never −1 for a loss: negative terminal rewards under γ < 1 pay the agent to
  delay losing, which shows up as pathologically long games and a rollout
  throughput collapse (all four Orbit Wars RL solutions hit a version of it).
  No reward shaping in v1; prize-differential shaping is a registered
  follow-up experiment (S3-E2), not a default.
- **KL penalty to the frozen prior** (PRIOR or the Stage-2 winner — pick one
  and log which) instead of, or alongside, a naive entropy bonus. This is the
  concrete stage-3 use of the calibrated BC prior promised in §8.2, and the
  named guard against on-policy forgetting. Log KL-to-prior and entropy per
  SelectContext every update; a KL blowup or an entropy collapse is a stop
  signal, not a curiosity.
- **Critic is train-time only.** The shipped bundle contains the actor alone
  (CPU + ~197.7 MiB envelope, same as every prior stage).
- **Hyperparameters**: no manual knob-twiddling marathons. If a sweep is
  needed, the tool is Protein (PufferLib's cost-aware sweep, §8.3) — the
  flat-dict config from work item 2 is its interface, and the sweep objective
  is the Rung-2 proxy signal, not val loss.

### 3.3 Promotion gate (per candidate checkpoint)

Snapshot a candidate every N updates (N set from the measured steps/sec so
candidates arrive a few times per day, not per minute). A candidate is
promoted to current-best only if **both**:

1. Rung 2: beats the current best with non-overlapping Glicko intervals
   (self-play excluded from rating periods, as the harness already does), and
   still beats the public pool — a candidate that gains in the mirror but
   loses to `kiyotah_dragapult` has collapsed, not improved.
2. Ladder: submitted, scored, and reads above the current best's settled score.

Promoted candidates join the frozen-opponent league. Two consecutive
non-promoted candidates = the loop has stalled; stop and write up rather than
tuning knobs past the deadline. Rung 3 (read five full transcripts) runs once
per candidate regardless — the failure mode self-play invites is a policy that
exploits its own blind spots in ways transcripts show and aggregates hide.

---

## Work items in order

1. **[S2] Data plumbing** — extend `iter_decisions`/`ILDataset` with outcome,
   seat, episode_id, manifest ratings (2.0). Smoke test + distribution figure.
2. **[S2] Weighted trainer** — per-row weight hook in the train loop
   (`w · CE`), config-driven arm selection, flat-dict config (Protein-ready,
   bc_pipeline_v2 §8.3 — and Stage 3's sweep interface).
3. **[S2] Run E0/E1/E2** (3 seeds each, equal steps) → Rung 1 + Rung 2 →
   chart → pick arm → optionally E3/E4 → submit → **Stage-3 init declared**.
4. **[S3] Rollout engine** — PufferLib emulation wrapper for cabt (flatten
   variable-length obs/options, mask passthrough) + multiprocessing
   vectorization + the steps/sec measurement report (3.1).
5. **[S3] PPO trainer** — `scripts/train_ppo.py`: masked PPO loss,
   KL-to-prior, fresh critic, opponent sampler, in-memory rollout buffer;
   persists checkpoints and TB logs only (3.2).
6. **[S3] Promotion loop** — snapshot → Rung 2 → ladder → league (3.3);
   repeat until two consecutive fails or the calendar (2026-08-16 minus
   rating-settle time) ends it.

Stop conditions (any → halt and report): a projected run > 1 hour without
approval; any opponent-private field found reaching the encoder from the
rollout path; measured vectorized steps/sec too low for a meaningful update
budget (report the number and the arithmetic); Stage-2 gate unmet by every arm
(then Stage 3 initializes from PRIOR and the writeup says why); KL-to-prior
blowup or per-context entropy collapse during PPO; any design that wants to
write episodes to disk beyond Rung-3 transcript samples; bundle MiB over the
envelope.
