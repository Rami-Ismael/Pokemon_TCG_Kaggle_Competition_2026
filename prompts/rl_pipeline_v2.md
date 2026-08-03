# RL Pipeline v2 — IL → Offline RL → On-Policy PPO Self-Play

Successor to `prompts/rl_pipeline_v1.md` (left unchanged for diffing), revised
against the 2026-08-03 review. The plan follows
[[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
(Grigsby et al., RLC 2025, arXiv:2504.04395 — "the Metamon paper" hereafter,
always cited by that wikilink). Phase names now match the authors' own
terminology: their released checkpoints are literally named `SmallIL`,
`SmallRL`, and `SyntheticRLV0/V1/V2`, and their abstract describes "a
progression from imitation learning to offline RL and offline fine-tuning on
self-play data." So:

| This repo's phase | Paper's name | Old v1 name |
|---|---|---|
| Phase 1 | **IL** (imitation learning) | PRIOR |
| Phase 2 | **Offline RL** (on the human corpus; the paper's "RL" checkpoints) | REWEIGHT |
| Phase 3 | **On-Policy PPO Self-Play** (our adaptation of the paper's "Synthetic RL" / self-play fine-tuning) | SELFPLAY |

Experiment IDs keep their `S2-`/`S3-` prefixes — trained artifacts
(`models/s2/e1_seed43`, submission 55196434) already carry them.

## What changed from v1 (review of 2026-08-03)

1. Phases renamed to the paper's terminology (table above).
2. **Training corpus doubled 2026-08-02**: new split `train-2026-07-01`
   (5,266 episodes, median avg_score 1180.3) joins `train-2026-07-26`
   (4,554) → **9,820 train episodes, ~2.16×**. Details in §Data.
3. Every paper citation now carries its title in `[[wikilink]]` form.
4. New §Hyperparameters: Protein-driven sweeps, with the specific PPO knobs
   named and prioritized.
5. The "frozen control deck" sentence is now spelled out (§3.1).
6. §2 now states exactly which Metamon training Phase 2 corresponds to.
7. PufferLib 4.0 risk front-loaded: install and stress-test it as **work
   item 0**, not when Phase-3 plumbing needs it (§Work items).
8. AWR is defined where first used (§2.1).
9. "Metamon" always resolves to the full-title wikilink.
10. Ladder submission after every phase is an explicit standing ritual, not an
    implicit gate clause.
11. Weighted BC situated as offline RL, with reference implementations linked.
12. New §Model scale & architecture: size grid (Option B, rescaled) and the
    causal-transformer question.
13. New §Data: our corpus vs the paper's, like-for-like.
14. The literal 15M/50M/200M grid is **rejected**; Option B (3.32M / ~10–12M /
    50M-probe) adopted. Rationale in §Model scale.
15. **All approval gates removed.** Long runs launch immediately; the only
    stop conditions left are technical ones.

---

## Data — ours vs the paper's

Train split is now **two held-out days** (§F4 held-out-DAY rule preserved —
train and eval remain distinct calendar days):

| Split | Day | Episodes | Manifest ratings? |
|---|---|---:|---|
| train | 2026-07-26 | 4,554 | ⚠️ no — `manifest.csv` has zero rows for this day; `avg_score` is the −1.0 sentinel |
| train (new 2026-08-02) | 2026-07-01 | 5,266 | **yes** — median avg_score 1180.3; this day was selected for high median rating |
| eval | 2026-07-27 | 4,430 | yes |

Consequences of the doubling:

- Rows/epoch: ~826K measured on the old single train day → **~1.78M
  estimated** at 9,820 episodes (unverified until the next dry-run epoch
  prints it — repo rule 5).
- All new training runs consume **both** train days. The equal-steps rule
  (never equal-epochs) matters even more now: an epoch is ~2.16× larger, so
  any comparison against the 12,900-step E0/E1/E2 runs is at equal *steps*.
- S2-E3 (skill × outcome) is **half-unblocked**: the new day carries real
  ratings, the old day still needs its manifest fetched from the Kaggle
  episodes dataset (small CSV; `kaggle auth login` first). Until then E3 can
  run on the 2026-07-01 day alone, or gate rows without ratings to weight 1.

Scale comparison with [[Human-Level Competitive Pokémon via Scalable Offline
Reinforcement Learning with Transformers]] (their numbers from metamon.tech
and the repo README):

| | Paper (publication era) | Paper (maintained dataset, 2026) | This repo |
|---|---|---|---|
| Human battles | 475k+ reconstructed | ~2.7M | **9,820** train episodes |
| Trajectories (one side of one battle) | ~1M | ~5.3M | **~19.6K** (2 seats × 9,820) |
| Self-play data | grows corpus by 4M–11M trajectories | 20M+ | none stored (disk budget — §3) |

We sit **~50× below their battle count and ~2 orders of magnitude below their
trajectory count**, even after doubling. Every scale decision below (model
grid, no 200M, on-policy Phase 3) follows from this row of arithmetic.

---

## Model scale & architecture

### Size grid — Option B (adopted), literal paper grid (rejected)

The paper swept 15M / 50M / 200M. Copying that grid here is rejected:
**"200M params against ~826K rows/epoch is a 2-order-of-magnitude smaller
ratio than the source paper ever ran"** — and even at ~1.78M rows the ratio
is still ~50× worse than their smallest model's. A 200M model quantized to
fit the evaluator would be a new, untested artifact on an envelope
(~197.7 MiB, ~1.6 vCPU) with no headroom. Instead, anchor at the deployed
size and treat the upper tiers as ceiling probes:

| Tier | Params | Rationale |
|---|---|---|
| Small | **3.32M** (current) | Already deployed, already measured (2.85 ms/decision CPU fp32, ~20 min/epoch) — the true baseline, not a new run |
| Medium | **~10–12M** | Inside the repo's own 2–15M soft band; large enough to test whether scaling helps at all before spending a ceiling probe |
| Large | **~50M** (the repo's hard ceiling) | A probe, run once objective selection is settled on Small/Medium — not a tier to sweep all objectives against. ⚠️ 50M fp32 ≈ 200 MB weights alone, over the ~197.7 MiB envelope: the probe ships int8-quantized or not at all, and the quantized artifact re-runs the forced-CPU latency rehearsal before any submission |

Scaling runs compare at equal steps, 3 seeds, same objective — size is an
axis orthogonal to the S2 objective ladder, explored only after an objective
wins on Small.

### Architecture axis

The paper's models are **causal transformers over the full battle trajectory**
— the sequence dimension is turns, so the policy conditions on everything both
players have revealed and can adapt to the opponent within one game. Our
`PTCGImitationPolicy` (hidden 192 / 6 layers / 6 heads) is a **per-decision
set transformer**: attention runs over the entities of the *current*
observation; there is no cross-turn memory. That is a real representational
gap — opponent modeling and long-game planning live in exactly the history our
architecture cannot see.

Register (do not build yet) **S-ARCH: causal-over-history variant** — same
encoder, plus a turn-level causal transformer with a bounded context window
(memory scales with window; the envelope must be re-measured). Run it only
after the size grid answers whether *any* added capacity pays at our data
scale; a history model at 9.8K episodes may simply overfit — which is itself
the measurable answer.

---

## Standing rules (all phases)

- Device via `resolve_device()` only; no CUDA branch exists. Training on MPS,
  the evaluator is CPU-only (~1.6 vCPU, ~197.7 MiB — unverified envelope,
  design to it anyway). `torch.set_num_threads(1)` before heavy torch work.
- Everything under `uv run`. Paths from `pokemon_tcg.config`. Seed 42.
- Action masking is structural (Pattern-B option-scoring head) — never a path
  that can emit an out-of-range index. In policy-gradient training the mask is
  part of the objective, not a filter:
  [[A Closer Look at Invalid Action Masking in Policy Gradient Algorithms]]
  (Huang & Ontañón, arXiv:2006.14171) shows masking changes the gradient.
- No opponent-private information reaches the encoder. Phase-3 rollouts feed
  the encoder the acting agent's `obs_dict` as served by the env — the same
  view the ladder serves — never the engine's omniscient state.
- Every comparison: ≥3 seeds, equal **steps** not equal epochs, an RD or σ on
  every number, a chart in `reports/figures/`, and the control beside the claim.
- **After every phase (and every promoted Phase-3 candidate): submit to the
  real ladder.** Local Glicko and the leaderboard have already diverged once
  (2026-08-02). The ritual is fixed: build the bundle
  (`scripts/build_improved_submission.py` pattern — read the printed tarball
  MiB), forced-CPU latency rehearsal, submit with a detailed message (what
  changed, from what baseline, expected effect), then **read the leaderboard
  after scoring** and record the number next to the local claim. No checkpoint
  is called "better" on local numbers alone.
- **No approval gates.** Long runs launch immediately; log the projected
  duration and checkpoint often enough that any run can be killed and resumed
  without loss. The stop conditions at the end of this file are technical
  triggers, not requests for permission.
- **Disk budget: ~16 GB free.** No phase may write an episode corpus. Allowed
  persistent artifacts: checkpoints (~13 MB fp32 at 3.32M), TB logs, figures,
  a handful of Rung-3 transcripts. Anything else that wants disk is a design
  error — redesign, don't compress.
- Ladder closes **2026-08-16**; ratings settle only over hundreds of games
  (~48 matches/day). A checkpoint submitted with fewer than ~3 days left
  cannot be confidently evaluated — plan the last submission slot backward.

---

## Phase 1 — IL (built; what it owes the later phases)

Status: complete. `scripts/train_il.py` → `PTCGImitationPolicy` at
`models/il_agent/`, 3.32M params, 2.85 ms/decision CPU fp32
(`notes/phase1_decisions.md`). This is the paper's IL stage: plain behavior
cloning of human replays, the `SmallIL`-equivalent.

What Phases 2–3 consume from it:

1. **A frozen, independently loadable checkpoint** via `.from_pretrained()`,
   never mutated: (a) warm-start for Phase 2, (b) KL anchor for Phase 3,
   (c) the control arm in every experiment.
2. **Calibrated probabilities over the option set** — the weighted objectives
   and the KL penalty both need the full distribution, not argmax.
3. **The encoder version.** Phases 2 and 3 use the same
   `encode_observation()` unchanged. Any encoder change forks the pipeline
   version and invalidates cross-stage comparisons.

⚠️ Note: the deployed IL checkpoint was trained on the single old train day.
It stays frozen as-is (it is the measured baseline); the doubled corpus enters
through Phase-2 retraining, not by retraining the anchor.

---

## Phase 2 — Offline RL (weighted BC on the human corpus)

**Which Metamon training this is, exactly:** the paper's second stage — the
step that turns `SmallIL` into `SmallRL`. Same replay dataset, no new games;
only the objective changes, from uniform cloning to value-aware training
(actor-critic offline RL in their case). Their finding, in one line: the same
human data trains a meaningfully stronger policy when rows are weighted or
filtered by outcome/advantage instead of cloned uniformly.

**Is weighted BC an offline RL approach? Yes.** Filtered and
advantage-weighted BC are the "one-step" corner of offline RL — policy
improvement by regression against reweighted logged actions, no bootstrapped
Q-learning required. The lineage:
[[Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning]]
(Peng et al. 2019, arXiv:1910.00177),
[[Exponentially Weighted Imitation Learning for Batched Historical Data]]
(Wang et al. 2018 — MARWIL, shipped in RLlib), and
[[Critic Regularized Regression]] (Wang et al. 2020). Reference
implementations to crib from: the official
[xbpeng/awr](https://github.com/xbpeng/awr), the compact
[omardrwch/advantage-weighted-regression](https://github.com/omardrwch/advantage-weighted-regression),
the index at
[hanjuku-kaso/awesome-offline-rl](https://github.com/hanjuku-kaso/awesome-offline-rl),
and Metamon's own training code in
[UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon) (AMAGO
trainer).

**What AWR is**, since S2-E2 is shaped by it: advantage-weighted regression
trains the policy by maximizing `w · log π(a|s)` over logged actions with
`w = exp(A(s,a)/β)` — a soft, exponential preference for actions that did
better than expected. `A` is return minus a baseline (in E2 the baseline is
the constant b = 0.5, making it outcome-weighted BC; in E4 it is a learned
`V(s)`, full AWR). β is the temperature: large β → weights flatten toward
plain BC; small β → approaches hard winners-only filtering. E1, E2, E4 are
three points on that same dial.

### 2.0 Data plumbing — STATUS 2026-08-02: built

`DecisionMeta` in `il_dataset.py`, `ILDataset(with_meta=True)`,
`load_manifest_scores()`, acceptance `scripts/check_weight_plumbing.py`,
tests green. Rows carry `outcome` (remapped {0, 0.5, 1} — raw −1/0/1 never
enters an objective), `seat`, `episode_id`, `avg_score`/`min_score`,
`turn_index`. Hard `winner_only` filter exists
(`train_il.py --winner-only`). Measured: winner seats contribute ~52.9% of
rows — even "unweighted" BC has a mild winner tilt.

**New task (from the corpus doubling):** point the loaders at both train
folders via `resolve_split_dir` and re-run `check_weight_plumbing.py` on the
union — acceptance is the printed outcome distribution (~50/50) and the
avg_score histogram, which now has a real (non-sentinel) mode from the
2026-07-01 day. Fetch the 2026-07-26 manifest rows when convenient to fully
unblock E3.

### 2.1 The objective ladder

All arms warm-start from the IL checkpoint, same step count, same LR
schedule, 3 seeds, per-row weight `w` is the only difference:

| ID | Arm | Weight `w` | Notes |
|---|---|---|---|
| S2-E0 | control | 1 (plain BC, continued) | separates "more steps" from "better objective" |
| S2-E1 | filtered BC (winners-only) | 1 if won else 0 | the paper-adjacent simple baseline; halves the data — equal-steps rule matters |
| S2-E2 | outcome-weighted BC | `exp(β·(outcome − 0.5))`, β ∈ {0.5, 1, 2} | AWR-shaped (see above); keeps losing data at reduced weight |
| S2-E3 | skill × outcome | E2 weight × 1[avg_score ≥ Q75] | the manifest-rating lever; runnable now on the 2026-07-01 day |
| S2-E4 | critic-advantage weighted | `exp(A(s)/β)`, A = outcome − V(s) | closest to the paper's actual actor-critic RL stage; value head is train-time only — the shipped bundle contains the actor alone. If built, it is the natural critic init for Phase 3 |

**STATUS 2026-08-02: E0/E1/E2 trained** (3 seeds, equal 12,900 steps, warm LR
1e-4, on the *old single train day*) and Rung-2 benchmarked. Rung 1:
three-way tie (75.8% ± 0.1, IL 75.3%) — offline accuracy didn't separate
them, as predicted. Rung 2: every arm beats IL head-to-head (E0 62.0%
[50.3,72.4], E1 62.5% [51.0,72.8], E2 56.9% [45.4,67.7]); arms don't separate
from each other; all still lose ~92% to the strong public trio. Ladder:
`s2_e1_s43` submitted 2026-08-02 (submission 55196434) — first read **516.7
vs the IL agent's settled 400.0**, provisional until ~3 days settle it.

**Next runs use the doubled corpus.** The E0/E1/E2 results above stand as
old-corpus measurements; the doubled-corpus rerun (same three arms first, then
E3 now that ratings exist) is the cheapest expected win in the whole plan —
the paper's central axis is data scale, and this is our only 2× data lever.
Compare at equal steps against the 12,900-step runs, and also extend steps to
match the larger epoch — both comparisons, charted.

Known trap, restated: winners-only filtering can lower Rung-1 accuracy while
raising Rung-2 win-rate — offline action-match is a pipeline check, never a
selection metric. Selection is Rung 2, confirmed on the ladder.

### 2.2 Phase-2 gate (technical, no approvals)

1. Rung 1 vs the 0.381 majority line (pipeline check only).
2. Rung 2 round-robin (`scripts/benchmark_agents.py`) — winning arm vs IL and
   the public pool, mirrored pairs until Glicko intervals separate; quote RD.
3. The ladder ritual from Standing rules. The Phase-2 winner becomes the
   **Phase-3 initialization** only after the leaderboard confirms it.

---

## Phase 3 — On-Policy PPO Self-Play (adapted from the paper's Synthetic RL)

**The deviation, stated loudly (repo rule 5):** in
[[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
this stage is *offline*: self-play battles are appended to an ever-growing
corpus (their `SyntheticRLV0→V2` line, +4M–11M trajectories per generation)
and offline RL retrains on the union — so the human prior can never be
forgotten, because every retrain still sees it. That requires corpus storage;
this laptop has ~16 GB free against a human dump already ~40 GB. Phase 3 here
is therefore **on-policy PPO**
([[Proximal Policy Optimization Algorithms]], Schulman et al. 2017): rollouts
live in RAM, are consumed by the update, and are freed. What the deviation
costs: PPO sees only the current policy's states, so forgetting the prior is
a live failure mode — the KL anchor in §3.2 is the guard, and per-context
KL/entropy logging is how the guard is watched, not assumed.

### 3.1 Rollout engine

- **Parallelism: game-level stdlib multiprocessing** (`pokemon_tcg/selfplay.py`).
  PufferLib emulation is not the v1 path (cabt is callback-driven,
  `env.run([a, b])`; games are env-bound at ~0.9 s with ~3 ms policy calls),
  but PufferLib is the named fallback if step-level vectorization ever becomes
  the bottleneck **and** the Protein sweep dependency regardless — which is
  why work item 0 stress-tests it now, not later (see §Hyperparameters and
  Work items).
- **Measured 2026-08-02** (`scripts/probe_selfplay_throughput.py`): 8 workers
  on the M4 Pro → 6.2 games/s mirror, 425 rollout rows/s (~1.5M rows/hour);
  league mix 5.4 games/s; 0 fallback decisions in 176 games. A 4096-row PPO
  update collects in ~10–15 s. Gotcha recorded: every callable handed to
  `env.run` goes through `as_env_agent()` (bound-method `co_argcount` trap).
- **Rollout buffer sized to RAM, not disk** — 2048–8192 decisions per update,
  a config knob, never spilled to disk.
- **Opponent sampling per episode** (league play, the collapse guard):
  learner side always current policy; opponent ~50% current weights (mirror),
  ~30% frozen past checkpoints (league), ~20% public pool from `AGENT_FILES`.
  A league of 10 frozen checkpoints ≈ 130 MB. The mix is a knob — log it.
- **Exploration at rollout**: sample at temperature ~1.0, never argmax
  (argmax self-play produces near-duplicate games). Temperature goes in the
  run manifest.
- **Masking**: structural Pattern-B masking in rollout *and* loss
  ([[A Closer Look at Invalid Action Masking in Policy Gradient Algorithms]]).
- **The deck constraint, spelled out.** In this repo's harness a deck is not
  an independent variable: each agent directory ships one fixed 60-card
  `deck.csv`, and `load_agent` binds that list to the agent — there is no code
  path that assigns decks per-game. "The frozen control deck" means the one
  60-card list our agent lineage ships (the same list Phase 1 and Phase 2
  trained against and the ladder submissions carry). So in every Phase-3
  game, our learner, our frozen league checkpoints, and the mirror opponent
  all pilot that identical list; only the public-pool opponents differ (their
  own shipped decks). "Adding a deck axis" would mean building harness
  support for varying the deck independently of the policy — new plumbing plus
  cg-legality checks per list — and is out of scope for v2. Consequence for
  reporting: Phase 3 improves *this deck's* policy; nothing here measures
  general strength across decks, and reports must say so.

### 3.2 PPO specifics

- **Init**: actor from the Phase-2 winner. Critic head fresh (or from
  S2-E4's value head if built), trained on the remapped terminal reward.
- **Reward**: terminal only, win = 1, draw = 0.5, loss = 0. Never −1 for a
  loss: negative terminal rewards under γ < 1 pay the agent to delay losing —
  pathologically long games and rollout-throughput collapse (all four Orbit
  Wars RL solutions hit a version of it). No reward shaping in v1;
  prize-differential shaping is registered follow-up S3-E2, not a default.
- **KL penalty to the frozen prior** (IL checkpoint or Phase-2 winner — pick
  one and log which) instead of, or alongside, a naive entropy bonus. This is
  the guard against on-policy forgetting. Log KL-to-prior and entropy per
  SelectContext every update; KL blowup or entropy collapse is a stop signal.
- **Critic is train-time only.** The shipped bundle contains the actor alone.

### 3.3 Promotion gate (per candidate checkpoint)

Snapshot every N updates (N from measured steps/s so candidates arrive a few
times per day). Promote to current-best only if **both**:

1. Rung 2: beats current best with non-overlapping Glicko intervals AND still
   beats the public pool — a candidate that gains in the mirror but loses to
   `kiyotah_dragapult` has collapsed, not improved.
2. Ladder: submitted (the standing ritual), scored, reads above the current
   best's settled score.

Promoted candidates join the frozen league. Two consecutive non-promotions =
stalled; stop and write up. Rung 3 (read five full transcripts) runs once per
candidate regardless — self-play's failure mode is exploiting its own blind
spots in ways transcripts show and aggregates hide.

---

## Hyperparameters — what gets tuned, and how

No manual knob-twiddling marathons. The tool is **Protein**, PufferLib's
cost-aware Bayesian sweep (the headline results in
[puffer.ai blog post 12](https://puffer.ai/blog.html#post-12) — note those
landed on **PufferLib 3.0**; our risk is 4.0, see work item 0). The flat-dict
config from work item 2 is Protein's interface. The sweep objective is the
**Rung-2 proxy** (win-rate vs a fixed opponent set at a fixed game budget),
never val loss — offline metrics already failed to separate arms once.

Phase-2 sweep space (cheap, run first):
- AWR temperature **β** (E2/E4) — the one knob v1 already swept by hand
  ({0.5, 1, 2}); Protein owns it now.
- Skill threshold quantile (E3): Q50–Q90.
- LR and warm-start LR schedule.

Phase-3 (PPO) sweep space, in priority order — the top three are where PPO
lives or dies, the rest are refinements:

| Priority | Knob | Range to search | Why |
|---|---|---|---|
| 1 | learning rate | 1e-5 – 3e-4, log | dominant sensitivity in every PPO study |
| 2 | KL-to-prior coefficient | 0.01 – 1.0, log | *our* critical knob: too low → forgetting, too high → frozen at the prior |
| 3 | clip ε | 0.1 – 0.3 | update aggressiveness |
| 4 | GAE λ | 0.9 – 0.98 | credit over ~68-decision games |
| 5 | γ | 0.99 – 1.0 | terminal-only reward argues for ≈1; sweep confirms |
| 6 | update epochs × minibatch | 1–4 × {256, 512, 1024} | sample reuse vs staleness |
| 7 | rollout buffer size | 2048–8192 | collect/learn balance (measured: 4096 ≈ 10–15 s) |
| 8 | rollout temperature | 0.8 – 1.2 | exploration breadth |
| — | entropy bonus | 0 or small | secondary to the KL anchor; include only if KL alone under-explores |

Fixed, not swept: opponent-mix ratios (a design choice, logged), seed policy,
reward mapping, masking (structural, non-negotiable). Sweep budget honesty:
each Protein trial costs a real Rung-2 evaluation — size the trial count from
measured games/s and the calendar, and log what the sweep did *not* cover.

---

## Work items in order

0. **[NOW] PufferLib 4.0 stress test.** `uv add pufferlib` (4.x) immediately
   and try hard to break it on this machine *before* any phase depends on it:
   import under uv on ARM64 macOS, run its bundled demo envs, run a toy
   Protein sweep end-to-end, and attempt the cabt emulation wrapper as a
   smoke test (even though v1's rollout path doesn't need it). If 4.0 breaks,
   pin the newest working 3.x (the post-12 gains are 3.0-era anyway) and
   record the pin + failure mode in `notes/`. The point is to discover today,
   not during Phase-3 integration, which PufferLib we actually have.
1. **[S2] Corpus-doubling plumbing** — loaders read both train days; re-run
   `check_weight_plumbing.py` on the union; avg_score histogram figure;
   fetch the 2026-07-26 manifest rows to fully unblock E3.
2. **[S2] Weighted trainer** — per-row weight hook (`w · CE`), config-driven
   arm selection, flat-dict config (the Protein interface).
3. **[S2] Doubled-corpus runs** — E0/E1/E2 (3 seeds, equal steps vs the
   12,900-step baselines *and* an extended-steps arm), then E3 with real
   ratings; Protein on β and the E3 quantile → Rung 1 + Rung 2 → chart →
   pick arm → **ladder ritual** → Phase-3 init declared.
4. **[S3] Rollout engine** — multiprocessing vectorization per §3.1 +
   steps/s report (probe exists; productionize).
5. **[S3] PPO trainer** — `scripts/train_ppo.py`: masked PPO loss,
   KL-to-prior, fresh critic, opponent sampler, in-memory buffer; checkpoints
   + TB logs only. Protein sweep per §Hyperparameters.
6. **[S3] Promotion loop** — snapshot → Rung 2 → ladder → league; repeat
   until two consecutive fails or the calendar (2026-08-16 minus
   rating-settle time) ends it.
7. **[Parallel, gated on item 3's winner] Size grid** — Medium (~10–12M) with
   the winning objective, 3 seeds, equal steps; Large 50M int8 probe only if
   Medium beats Small on Rung 2. S-ARCH (causal-over-history) only after the
   grid answers whether capacity pays at all.

Stop conditions (any → halt and report; none require anyone's approval):
opponent-private field reaching the encoder from the rollout path; measured
steps/s too low for a meaningful update budget (report the number and the
arithmetic); Phase-2 gate unmet by every arm (then Phase 3 initializes from
IL and the writeup says why); KL-to-prior blowup or per-context entropy
collapse during PPO; any design that wants to write episodes to disk beyond
Rung-3 transcripts; bundle MiB over the envelope.
