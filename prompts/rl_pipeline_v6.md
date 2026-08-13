# RL Pipeline v6 — imitation → binary-advantage weighting → lineage-only self-play

Revision of `prompts/rl_pipeline_v5.md` (which lives untracked in the
`pokemon-tcg-ppo-pipeline-0e0618` worktree). Written 2026-08-13 on Rami's
direction. Ladder closes 2026-08-16 — three days.

The three steps still map onto Grigsby et al., RLC 2025 (arXiv:2504.04395, "the
Metamon paper") — **imitation → offline RL → self-play fine-tuning**. v6 exists
because v5 deviated from the paper in two places, and both were mistakes:

1. **v5 picked the wrong weighting scheme for the offline RL step.** It ran the
   exponential-advantage weighting (`w = exp(β·(outcome − V(s)))`), the one
   variant from the paper's menu that our runs had already shown does not beat
   the plain imitation checkpoint. v6 runs the recipe the paper's strongest
   model (SynRL-V2) actually uses: **Binary** weighting (`w = 1[advantage >
   0]`) with the calibrated critic, and — as the one follow-on — the paper's
   **two-hot critic** upgrade (§1.1).
2. **v5's self-play was not real OSFP.** OSFP (optimistic smooth fictitious
   play, ByteRL, arXiv:2303.04096) is by definition played against the
   learner's *own* population — its lineage. v5 put external agents in the
   league (`--field-pool`, the public pool, PFSP weights over the field) and
   ran on-policy PPO. v6 self-play contains **no external agents anywhere in
   the process**: the model plays only its own past checkpoints, opponents
   sampled OSFP-style, and fine-tunes **offline** on the aggregated self-play
   episodes — the paper's actual third step.

**v4 is not discussed in this document.** Its failure was implementation
defects (hardcoded opponent deck, mirror-only league, wrong learning rate); it
carries no evidence about any method and nothing here needs defending against
it.

Everything v5 settled about *process* stays settled: no halting conditions
anywhere in the pipeline, the correctness invariants that are not halting
conditions, and the standing rules (v5 §2). Cited, not re-derived.

---

## §1 — The three steps

| Step | What runs | Init | Output | Status |
|---|---|---|---|---|
| **`imitation`** | nothing — reuse the finished 53-day checkpoint | — | `bc_alldays52_jun16_aug07_seed42` (main repo, `agents/`) | done; do not retrain |
| **`binary_advantage`** | `train_il.py --weight-arm adv-binary --critic-dir <calibrated critic>` on the streamed HF corpus | `imitation` | `models/binaryadv_alldays_jun16-aug07_seed42` | flag already exists; runnable today |
| **`lineage_selfplay`** | `scripts/run_lineage_selfplay.py`: play games against lineage checkpoints via the direct battle API, write episodes in corpus format, resume `train_il.py` over human ∪ self-play episodes with the same binary-advantage loss | `binary_advantage` | `models/lineage_selfplay_<mint-rule>_gen<N>_seed42` | **built + smoke-tested 2026-08-13** (driver, episode writer, `--selfplay-shards` union in `train_il.py`); waiting on the `binary_advantage` checkpoint to launch |

No step waits on a performance read. The next step starts when the previous
process exits 0. Halting conditions stay removed (v5 §0); the diagnostics in §5
are charted, never enforced.

With three days left, budget one `binary_advantage` run and **one generation**
of lineage self-play as the realistic scope.

### §1.1 — Step `binary_advantage`: what replaces the exponential weighting

The paper's actor loss (its Eq. 2) is

```
L_actor = E[ −w(h,a)·log π(a|h) − λ·E_{a~π}[Q(h,a)] ]
```

with a menu of weightings: `w = 1` (imitation), exponential advantage (what we
ran as `adv-exp` — the AWR/MARWIL corner), **Binary** `w = 1[A > 0]`, and
**Binary+MaxQ** (binary weighting with `λ > 0`). The critic behind `A` is
trained with one-step TD.

**Verified against the paper, 2026-08-13.** The paper's own reads: "RL updates
significantly outperform the pure-BC Transformers, but there is little
difference between the many RL variants considered" — and its strongest model,
**SynRL-V2, uses binary-weighted BC with the value head switched to two-hot
classification targets, no MaxQ term**. So the paper's most powerful offline
recipe is: binary weighting, plus a *more accurate critic* (the two-hot head
tightens the filter — the paper credits it with "entirely new levels of
pessimism in the binary BC filter"). The differentiator is critic quality, not
extra loss terms. v6's step is exactly that recipe at our scale: `adv-binary`
with the calibrated critic now, the two-hot critic as the one follow-on that
the paper's results actually rank above it.

**How win/loss becomes a per-decision label.** The raw label is one bit per
*game*; used directly (the `outcome` arm) it boosts every blunder in a won
game and buries every good move in a lost one — measured null. The critic is
the credit-assignment device that localizes it: `V(s)` is the expected
win probability at each state, so `A = outcome − V(s)` grades each decision
against expectation — a win from a losing position marks those moves as
better-than-expected; coasting on a won position carries no signal. The
mandatory `calibration.json` maps the critic's raw output onto a real
probability scale so the *sign* of `A` is trustworthy, and binary weighting
consumes only that sign. That is the whole chain: one bit per game → calibrated
per-state expectation → per-decision keep/drop.

Why binary:

- **The exponential scheme is measured-dead in our hands.** The exp-weighted
  runs reached 395.0 and 320.4 on the ladder against the imitation
  checkpoint's 418.0, and the earlier outcome-weighting comparison was null
  across 3 seeds. None of those runs ever tested the paper's binary variants.
- **Binary asks less of the critic.** Exponential weighting trains on every
  row, softly re-scaled — a badly calibrated `V(s)` bends every weight. Binary
  weighting *drops* the rows the critic thinks were losing moves, so it only
  needs the critic's **sign** to be right, not its scale. Our measured critic
  (AUC 0.76) is a usable sign-classifier and a proven-bad scale — exactly the
  profile binary tolerates and exponential does not.

Concretely:

- `--weight-arm adv-binary` already exists (`scripts/train_il.py`,
  `src/pokemon_tcg/offline_critic.py`: `w = 1[outcome − V(s) > 0]`).
- Critic: `critic_outcome_day_2026-07-26_seed42` **with its mandatory
  `calibration.json`** — the uncalibrated shipped critic is worse than a
  constant.
- **Binary+MaxQ is dropped, not deferred.** The paper's strongest model does
  not use the MaxQ term, and the `λ·E_{a~π}[Q]` term needs a per-action Q head
  our critic doesn't have. If the plain Binary run finishes with schedule
  left, the follow-on the paper's results rank first is the **two-hot
  critic**: retrain the outcome critic with two-hot classification targets,
  re-fit `calibration.json`, rerun `adv-binary` with it. Better sign accuracy
  feeds the filter directly; no trainer change needed.
- Variant choice verified against the paper 2026-08-13 (see §1.1 above). The
  standing rule stands: if the run scores badly, audit our implementation
  against the paper line by line before any verdict about the method.
- Corpus: `--train-split train_combined_v4` (all 53 days, includes 07-27),
  streamed with `--data-source auto --num-workers 4`. Holdout day is
  **2026-08-09**; never evaluate a v4-corpus checkpoint on 07-27.

### §1.2 — Step `lineage_selfplay`: the paper's actual third step

What the paper does (confirmed from the paper text, 2026-08-13): recent
checkpoints of the model battle **each other** — no external agents — until the
offline dataset reaches its target size, then offline training **resumes over
the aggregated dataset**. The defense against overfitting-to-self is not
opponent diversity; it is **team diversity** (their procedurally generated
"Variety Team Set"). Reward is dominated by binary win/loss with light shaping
for damage/health.

The v6 design, mapped to our stack:

- **Opponent sampling — OSFP over the lineage.** With probability `p_opt` the
  opponent is the *current* policy (the optimism, implemented exactly as ByteRL
  does it: a sampling probability on the learner, not gradient extrapolation);
  otherwise sample uniformly over the saved lineage checkpoints (the smooth
  fictitious-play mixture). Start `p_opt = 0.5`; log it per game. Lineage =
  `binary_advantage` + every generation checkpoint this run produces. The
  imitation checkpoint is in the lineage (it is this model's ancestor).
- **When a generation checkpoint joins the lineage — open question, now an
  experiment (2026-08-13).** Rami's design: a copy is minted into the lineage
  (and becomes the tryout reference) only when the live policy wins >70% of
  decisive tryout games against the newest lineage member; the v4/v5
  precedent is that this trigger fired 0/31 under PPO+KL, but the v6 offline
  loop is a different regime. The driver carries both rules behind one flag
  (`--mint-rule tryout|cadence`); either way the new checkpoint becomes the
  current policy — the rules differ only in lineage membership. Arms,
  metrics, and the committed decision rule live in
  `notes/experiments/2026-08-13-lineage-minting-rule.md`: adopt tryout only
  if its final agent beats the cadence arm's on the anchored 8-agent pool
  with non-overlapping Glicko intervals; overlap, or a tryout that never
  fires, ships cadence.
- **Deck diversity carries the paper's team-diversity role.** `--opp-deck
  sample` stays, and the *learner's* deck is sampled too — both sides draw from
  the full legal deck pool every game. This is the analogue of the Variety Team
  Set and the load-bearing defense against self-play collapse; it is not
  optional.
- **Offline aggregation, not PPO.** Games run through the direct battle API
  (~10–15× faster than the gym path; per-move inference on CPU, not MPS).
  Each game is written as an episode in the same format the HF corpus uses, so
  `il_dataset.py` consumes it unchanged. Training then resumes with the same
  binary-advantage loss over **human episodes ∪ self-play episodes**. The
  human prior cannot be forgotten because every batch still samples it — the
  forgetting problem earlier versions could only monitor with KL charts is
  dissolved structurally.
- **Reward for the critic on self-play episodes:** terminal binary win/loss
  only. The paper's light damage/health shaping is flagged, not adopted —
  consistent with our terminal-only rule to date; adopting it would be a
  separate experiment.
- **Built 2026-08-13:** `scripts/run_lineage_selfplay.py` (driver: lineage
  registry, OSFP sampler, deck sampler for both seats, mint rules, resume
  call), `src/pokemon_tcg/lineage_selfplay.py` (recording game workers +
  corpus-format episode assembly), `--selfplay-shards` in `train_il.py` /
  `extra_local_files` in `ShardILDataset` (the human ∪ self-play union).
  Episodes stream straight into zstd parquet shards — no raw JSON on disk
  unless `--keep-raw`. Smoke-verified: episodes round-trip exactly through
  `iter_episode_decisions` (the round-trip caught and fixed a reward-
  convention bug: loser must be −1, not 0 — 0 reads as a draw), and the full
  generate → resume → tryout loop runs clean. Illegal pool decks are dropped
  loudly on first engine rejection.
- Disk: one episode day is ~20 GiB raw and the laptop has ~45–49 GB free.
  Self-play episodes are ours to regenerate, so they may live locally and be
  deleted after training — but **never** delete or overwrite the human raw
  episode data, and say what was deleted in the run report.

### §1.3 — Zero external agents anywhere in the self-play process

The self-play process — opponent selection, game generation, the episode
dataset, and the training that consumes it — contains **no external agent at
any point**. Removed entirely:

- `--field-pool` and every external agent in the league
- PFSP weighting over public-pool opponents
- Heuristic anchors as opponents
- Any mixed-league mode where a league seat is filled by an agent that is not
  a checkpoint of this run's model

And the invariant that keeps them out for good, not just at launch:

- **No episode from a game involving an external agent may ever be written
  into the self-play corpus.** The episode writer runs only inside the lineage
  driver; evaluation games (below) are played through the benchmark path,
  which writes results, never episodes. If an external agent's game ends up in
  the dataset, the run is invalid — not weaker, invalid.

Still standing, *outside* the self-play process:

- **Evaluation.** The holdout battery and the local tournament against the
  anchored 8-agent pool run on every generation checkpoint after training, and
  the benchmark-the-pool-before-submitting rule stands. Measuring against the
  field is not part of self-play and feeds nothing back into it.
- Deck variety (§1.2 — it is the substitute for opponent variety).
- The correctness invariants: no opponent-private info in the encoder, bundle
  ≤ ~197.7 MiB, `num_envs == num_workers` where the gym path is still used,
  action masking on both sides of every objective, `resolve_device()` only,
  everything under `uv run`, seed 42.

---

## §2 — What v5 got wrong, stated plainly

1. **Wrong weighting scheme.** Of the paper's actor-loss variants, v5 ran the
   exponential one — the variant our own ladder reads had already measured as
   not beating plain imitation. The binary variants were sitting untried in
   the same equation of the same paper.
2. **Not real OSFP.** Fictitious self-play means best-responding to a mixture
   over your *own* past policies. The moment `--field-pool` put external
   agents in the league, the step stopped being fictitious self-play and
   became league training against a hand-picked field — a different method
   with a different convergence story. v6 restores the actual mechanism:
   population = lineage, optimism = sampling probability on the current
   learner, last-iterate checkpoint is the ship candidate.

If a v6 run scores badly, the first suspect is our implementation of the
paper — audit it against the paper line by line before any verdict about the
method. That rule stands regardless of version number.

---

## §3 — Decision rules (committed before the run; they interpret, never halt)

No rule below stops the pipeline. They decide what the write-up may claim and
what may be submitted.

| Comparison | Metric | Read |
|---|---|---|
| `binary_advantage` vs `imitation` | local tournament vs the anchored 8-agent pool, ≥50 games each, Glicko with RD | overlap = "no measurable gain", write it that way |
| `lineage_selfplay_gen1` vs `binary_advantage` | same battery, paired decks | same |
| mint rule: `tryout` vs `cadence` | anchored 8-agent pool, ≥50 games, Glicko with RD (card: `notes/experiments/2026-08-13-lineage-minting-rule.md`) | adopt tryout only on non-overlapping intervals in its favor; overlap or a never-firing tryout ships cadence |
| Any "better/beats/improves" claim | settled ladder read via the `leaderboard-check` skill | local numbers alone never support the claim — five inversions on record |

Submission rules unchanged from v5 §2: never submit without asking first;
never two experiments back-to-back (team score is the max of the latest 2
slots); refresh the submission ledger before any submit; report the current
rank of the latest submission, never a stale best-ever.

---

## §4 — Rules carried forward unchanged (v5 §2, abbreviated)

Stream the corpus (`Rami/ptcg-episodes` is the only copy of the human data);
never delete raw episode data without asking; chain runs and never overlap MPS
jobs, `nice` everything, keep the laptop responsive; preflight disk/RAM before
every training run; ≥3 seeds and an RD or σ on every compared number where the
schedule allows — with three days left, a single-seed result is reportable
only if labeled single-seed; plain checkpoint names (method, data window, seed
spelled out); back up each step's checkpoint to the HF backup repo
(`Rami/ptcg-s2v2-arms`) and verify the upload by listing the remote files and
reporting both counts.

## §5 — Diagnostics to chart at the end (none can halt anything)

- Entropy in nats with `exp(H)` on a secondary axis.
- Critic: explained variance and the calibration curve on the 2026-08-09
  holdout day, plus a shuffled-label control.
- **Fraction of rows the binary weight drops**, per epoch — this is the
  binary weighting's entire mechanism; if it drops ~0% or ~100% of rows the
  loss degenerated to imitation or to nothing, and the chart must show it.
- KL to the never-updated imitation reference on a fixed eval batch, per
  generation — cheap, and it answers "has it drifted from the human prior"
  even though forgetting is now structurally prevented.
- Lineage win-rate matrix: every generation vs every lineage member,
  per-opponent, not aggregate — self-play farming of a weak ancestor looks
  exactly like weak-opponent farming did.
- Dataset composition per generation: human vs self-play episode counts.
- `p_opt` and the realized opponent-sampling histogram.

The driver already writes the raw material for these: per-game rows in
`<out>/selfplay_log.jsonl`, per-generation summaries (opponent histogram,
per-opponent win rates, fallback rate, deck rejections) in
`<out>/generation_log.jsonl`, every mint decision in `<out>/mint_log.jsonl`,
and the human-vs-self-play episode counts in the resume step's stdout.
