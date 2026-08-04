# RL Pipeline v2 — PRIOR → REWEIGHT (restarted) → SELFPLAY

Successor to `prompts/rl_pipeline_v1.md` (kept unchanged for diffing). Same
three-stage skeleton, modeled on **"Human-Level Competitive Pokémon via
Scalable Offline Reinforcement Learning with Transformers"** (Grigsby et al.,
RLC 2025, arXiv:2504.04395 — "the Metamon paper"): BC prior → reweighted /
offline RL on the same data → self-play fine-tuning. Everything in v1 not
amended here carries forward unchanged — in particular the reward remap
{0, 0.5, 1} (never −1), masking in rollout AND loss, the 50/30/20 opponent
mix, the critic being train-time only, no opponent-private information in the
encoder, and the deck-confound warning.

Terminology (per Rami, 2026-08-04 — the old "Rung 1/2/3" ladder jargon
is retired in prose): **offline accuracy check** = held-out action-match vs
the majority baseline (`eval_rung1.py`); **local tournament** = the
`benchmark_agents.py` round-robin of real games with Glicko±RD; **replay
review** = reading full game transcripts (`eval_rung3_sanity.py`). Script
filenames keep their historical names.

**Why v2 exists — Stage 2 is being restarted from scratch (user direction,
2026-08-03).** The previously-run Stage 2 has two strikes against it:

1. **Documented suspected-error source:** `manifest.csv` had ZERO coverage
   for the train day 2026-07-26, so `avg_score` was the −1.0 sentinel across
   the entire training corpus (v1 §2.0). Any arm that touched ratings was
   silently unratable, and the plumbing has been heavily rewritten since the
   arms were trained.
2. **Ladder refutation (read 2026-08-04 03:36 UTC, ledger
   `reports/submission_ledger.jsonl`):** every recorded Stage-2/3 "win"
   failed to settle above its own baseline:

   | Checkpoint | Submission | First read | Settled/latest | Local claim at submit time |
   |---|---|---:|---:|---|
   | il_agent (PRIOR) | 55149903 (08-01) | — | **400.0** | baseline |
   | s2_e1_s43 (old E1 winner) | 55196434 (08-02) | 516.7 | **395.0** | beat PRIOR 62.5% [51.0, 72.8] |
   | ppo_u120832 (old Stage-3 gen 1) | 55215267 (08-03) | 532.2 | **275.1** | beat s2 87.5% [69.0, 95.7]; Glicko 1478±71 vs 1300±71, non-overlapping |
   | heuristic AdvancedPolicy (Mega Lucario) | 55162376 (08-01) | — | 804.0 (peak 827.8) | the ladder incumbent |

   The old Stage-2 gate is retroactively FAILED (395.0 < 400.0), and the old
   Stage-3 gen-1 promotion — despite non-overlapping local Glicko — was a
   ladder collapse. **No recorded Stage-2/3 result is trusted.** The restart
   is empirical necessity, not procedural hygiene.

Rule of precedence used throughout: where v1's text and the code's recorded
decisions disagree, the code wins (its decisions were made later, with
measurements); where the code looks wrong, it is flagged in the §B1 audit,
never silently "fixed".

---

## Corrections to stale v1 facts — each verified at runtime 2026-08-03/04

| Fact | v1 said | Measured now (this session) |
|---|---|---|
| Free disk | ~16 GB | **81.9 GB** (`shutil.disk_usage`, 2026-08-03 ~23:00 local). Fluctuates with concurrent sessions; the "no episode corpus on disk" rule stands regardless — the corpus lives on the Hub (ADR-001). |
| >1-hour runs stop for approval | Phase-6 rule, in force | **RETIRED 2026-08-03.** Report projections, do not stop for approval, chain runs so MPS is never idle and never contended (preflight is built into `train_ppo_puffer.py`; `--skip-preflight` to override). |
| Local-vs-ladder divergence | "diverged once (2026-08-02)" | **Four data points now:** 08-02 (#1), 08-03 (#2, memory), and the two settled refutations in the table above — s2_e1_s43 516.7→395.0 and ppo_u120832 532.2→275.1. Additionally, byte-identical resubmits of the 804.0 bundle read 512.0–670.4 on early reads (55224682, 55228113): **first reads carry ~±150-point noise; only ~3-day-settled scores mean anything.** |
| Raw episode corpus on disk | train 4,554 + eval 4,430 files | **Deleted in the 08-03 cleanup** (user-approved, after Hub verification). Resolvable files counted today: `train-2026-07-26/` **24**, `eval-2026-07-27/` **12**, `train-combined-0701-0726/` 24 of 9,820 entries (**9,796 symlinks dangle**), `train-2026-08-01/` 1,490 (new raw day, not yet on the Hub), `train-2026-08-03/` 2. **`splits.json` episode counts are stale — count resolvable files, never trust the JSON.** |
| Manifest coverage | 07-01 + 07-27 only; train day zero | Still true for the train day: **2026-07-26 has ZERO manifest rows** (12,884 total: 07-01 5,266 all-scored, 07-27 4,430 all-scored, 08-02 2,683 with only 339 scored, 08-03 505 with 86 scored). Fixing 07-26 is §B0. |
| Rollout throughput | 425 rows/s (game-level probe) | The probe number stands for `selfplay.py`, but the **integrated PufferLib-PPO path measured ~61 agent-steps/s** at 8 envs × 8 workers with MPS learner and league mix (v1 §3.1 STATUS). PPO batch arithmetic below uses 61/s, not 425/s. |

Also corrected: the training corpus is now **two train days** (2026-07-01,
5,266 episodes + 2026-07-26, 4,554 = 9,820 episodes ≈ 1.78M decision rows),
authoritative copy on the private HF repo `Rami/ptcg-episodes` (zstd Parquet,
all days hub-verified; local HF cache ≈ 977 MB so streaming reads at SSD
speed). Days 08-02/08-03 also exist on the Hub and 08-01 exists raw-locally;
they are a **registered extension** (more data, approximated ratings), not
part of the restart corpus — the restart trains on 07-01 + 07-26 so results
are comparable and the rating fields are (after §B0) fully real.

---

## Ladder context the plan must respect (new in v2)

- Team account: hashhakim. As of 2026-08-04 03:27 UTC: rank 2,378/6,221,
  team_score 692.7, top score 1259.4, top-8 cutoff 1135.8.
- The **active submission set is the displacement budget**: newest
  submissions displace oldest actives. The 08-02/08-03 experimental
  submissions (s2, ppo) dropped the team ~4,500 ranks until same-build
  resubmits of the 804.0-lineage heuristic bundle restored the active set
  (55224682 → 670.4, 55228113 → 512→600, both 08-04, still early-read).
- Consequence, binding: **an experimental submission costs an active slot
  for days.** A Stage-2/3 candidate is submitted only when (a) its local
  gate is decisively met (non-overlapping intervals, all three local checks) and
  (b) the calendar leaves ≥3 days of settle time. Ladder closes
  **2026-08-16**; ratings settle over hundreds of games (~48/day) — the
  **last confident submission slot is ~2026-08-13**, computed backward.
- The IL/RL lineage's real bar: beat **PRIOR's settled 400.0** (and the old
  s2's 395.0). The heuristic incumbent at ~700–800 is the team's active set,
  not this pipeline's gate — but it is the reason experimental slots are
  expensive.

---

## Standing rules (v1's list, updated)

- Device via `resolve_device()` only; no CUDA branch. Training on MPS; the
  evaluator is CPU-only (~1.6 vCPU, ~197.7 MiB — unverified envelope, design
  to it anyway). `torch.set_num_threads(1)` before heavy torch work.
- Everything under `uv run` (Stage 3: the `.venv-ppo` py3.12 side venv —
  PufferLib 3.0 pins numpy<2 and 4.0 dropped Python envs). Paths from
  `pokemon_tcg.config`. Seed 42 (+43, +44 for seed sweeps).
- Action masking is structural (Pattern-B option-scoring head) in rollout
  AND loss (Huang & Ontañón, arXiv:2006.14171: masking changes the
  gradient). Mask to finite −1e9, never −inf (Categorical.entropy NaNs).
- No opponent-private information reaches the encoder — rollouts feed the
  acting agent's `obs_dict` as served by the env; `steps[0][0]["visualize"]`
  is never read (`tests/test_privacy_no_leak.py`).
- Every comparison: ≥3 seeds, equal **steps** (never equal epochs — the
  binary/filtered arms discard rows, so equal-steps is load-bearing), RD or
  σ on every number, a chart in `reports/figures/`, control beside the claim.
- **Preflight before every training launch** (disk / RAM / competing
  processes — built into `train_ppo_puffer.py`; run the same checks by hand
  before Stage-2 arms). Chain runs; never overlap MPS jobs.
- **No approval gates.** Report projected wall-clock, launch, checkpoint
  often enough to kill/resume without loss.
- Disk budget: measured 81.9 GB free, but **no stage may write an episode
  corpus** — allowed persistent artifacts are checkpoints (~13 MB fp32 at
  3.32M params; `policy_full.pt` ~26 MB), TB/JSONL logs, figures, and a
  handful of replay-review transcripts.
- **Every "better" claim passes the ladder ritual** (§Measurement protocol).
  Local Glicko alone never declares improvement — it has now been wrong
  twice with non-overlapping intervals.

---

## Stage 1 — PRIOR (unchanged, identity pinned)

`models/il_agent`: hidden 192 / 6 layers / 6 heads, 3.32M params, step
38,562 (3 epochs over the single old train day), eval top-1 0.7534, git
f9a1b6f9, 2.85 ms/decision CPU fp32. Ladder-settled **400.0** (55149903).
This checkpoint is (a) the warm-start for every Stage-2 arm, (b) the control
arm E0's init, (c) the fallback KL anchor if Stage 2 fails its gate.

Note for the record: `models/il_agent_hfstream_combined_3ep` (83,454 steps
over the combined corpus via Hub streaming, eval 0.7527, git 024e16b) exists
as a reference point from the streaming-corpus work. It is NOT the PRIOR —
it is unmeasured on the ladder and would confound init-vs-objective in the
arm comparison. The restart holds the init fixed at the ladder-measured
PRIOR and lets E0 (plain BC, continued, on the restored corpus) absorb the
"more data, more steps" effect.

---

## Stage 2 — REWEIGHT, restarted from scratch

### 2.B0 Data recovery (blocking; nothing trains before this passes)

1. **Corpus:** stream from `Rami/ptcg-episodes` per ADR-001. The streaming
   loader (`ShardILDataset`) must gain the **meta plumbing** the weighted
   arms need — `with_meta` (outcome / seat / episode_id / turn / avg_score)
   and `winner_only` — which today exist only on the raw-JSON `ILDataset`
   path (`train_il.py` hard-exits on hub+weighted; that exit is the gap,
   remove it by closing the gap, not by deleting the guard). `episode_id`
   comes from the shard's own column (never parsed out of JSON); `avg_score`
   joins from a **freshly loaded manifest at train time** (the shard-baked
   score columns for 07-26 are permanently null — they were packed before
   the manifest was fixed).
2. **Train-day manifest rows (suspected-error source #1):**
   `ingest_episodes.py backfill-manifest` extended with `--from-hub`, so the
   episode-id universe comes from the Hub shards (4,554 ids for 07-26), not
   from the 24 surviving local files.
   **STATUS 2026-08-04: the Kaggle-CLI route is a measured dead end** — the
   `episodes` listing returns one page of most-recent games per submission,
   so a week-old day is unreachable (44 episodes listed for 07-26 across the
   current top teams, **0 of 4,554** overlapping the corpus). Used instead:
   the **team-name proxy** registered in `notes/phase0_discovery_report.md`
   §0.6(b), implemented as `scripts/backfill_manifest_names.py` — a team's
   proxy is the median `avg_score` over its episodes on the two fully-rated
   days (07-01 train, 07-27 eval; Glicko matchmaking pairs rating-adjacent
   opponents, so episode avg_score ≈ each participant's rating); a 07-26
   episode's avg_score is the mean of its two teams' proxies, written only
   when both are known. Result: 180 team proxies, **3,794/4,554 (83.3%) of
   07-26 covered**, merged (manifest now 16,678 rows).
   ⚠️ Scale caveat for E3, now with numbers: 07-01 carries TRUE at-the-time
   ratings (median ~1180); the 07-26 proxies inherit the mixed 07-01+07-27
   scale (median 1113.7). The pooled Q75 = 1189.0 therefore keeps mostly
   07-01 episodes — the E3 report MUST chart per-day threshold sensitivity
   so a scale artifact can't masquerade as a skill effect.
3. **Count resolvable files, never `splits.json`** (measured counts in the
   corrections table above). If streaming AND the Kaggle re-fetch both fail,
   **halt and report — do NOT train on the 24 surviving train-day
   episodes** (stop condition).

### 2.B1 Audit before retraining (cheap; all of it, findings written down)

1. `scripts/check_weight_plumbing.py` (grown a `--data-source hub` mode) on
   the restored corpus: outcome distribution ~50/50 by construction, and a
   REAL avg_score histogram — zero −1.0-sentinel mass after the backfill.
   Figure saved.
2. Outcome remap verified end to end: cabt's raw −1/0/1 never leaves
   `il_dataset._seat_outcome`; {0, 0.5, 1} everywhere; unknown-outcome rows
   (−1.0 sentinel) get weight exactly 0 in every arm.
3. Seat/winner pairing hand-checked on a sample: seat outcomes perfectly
   complementary per episode; winner seats ≈52.9% of rows (the v1-measured
   figure should roughly reproduce on the restored corpus).
4. No opponent-private fields reach the encoder: `test_privacy_no_leak.py`
   green on the streamed path too (same encoder, but re-run explicitly).
5. Leaderboard-check: record the settled scores (done — table above);
   **the restarted Stage 2's bar is s2_e1_s43's settled 395.0 and PRIOR's
   400.0**, i.e. a candidate must convincingly clear 400 to be worth an
   active-set slot.

**STATUS 2026-08-04 (all five run on the restored corpus):**
streamed scan saw **9,820 episodes / 1,663,092 decision rows**; row outcomes
**53.0% win / 46.9% loss / 0.1% draw** (winner-row share reproduces v1's
52.9%); episode-seat outcomes perfectly complementary (**9,815 win / 9,815
loss / 10 draw-seats**); **zero unknown-outcome (−1 sentinel) rows** — the
suspected-error source is gone; manifest join **92.3%** (9,060/9,820), pooled
avg_score quartiles Q25 = 1120.7 / median = 1150.8 / **Q75 = 1189.0** (E3's
starting threshold); figure `s2_plumbing_outcomes_avgscore.png`. Remap
hand-checked on sample episodes (raw −1/1 → {0,1} per seat, exact);
`test_privacy_no_leak.py` + the full IL suite green in this worktree (29
tests). One flag, not fixed (audit discipline): `benchmark_agents.py`'s
`AGENT_FILES` contains the s2_arms block twice — duplicate identical keys,
harmless, left in place.

### 2.B2a Critic first (blocking prerequisite for the advantage arms)

Train a state-value head V(s) on the restored corpus — architecture:
`OfflineValueModel` (PRIOR trunk copy + scalar head on cls_hidden),
train-time only, never shipped. **Target: one-step TD backups on the
remapped terminal reward** — V(s_t) ← γ·V(s_{t+1}) for non-terminal
decision rows, V(s_T) ← outcome ∈ {0, 0.5, 1} at the seat's last decision
(γ ≈ 1 for this episodic terminal-reward setting; decision-level
successor pairs within one seat's trajectory). The existing Monte-Carlo
regression mode (fit V(s_t) = outcome for every t — what `train_critic.py`
shipped with) is retained as an ablation arm; with γ=1 the two share a
fixed point in expectation, TD trades variance for bootstrap bias. If TD
and MC critics disagree materially on the audit below, that disagreement
is itself a finding to report.

**Critic audit — all three before ANY arm consumes advantages:**

1. V(s) predicts the held-out day's outcomes better than the base rate
   (report MSE vs the constant-0.5 predictor AND classification accuracy
   at the 0.5 threshold vs majority class; both must beat baseline).
2. Advantage distribution Â = outcome − V(s) on held-out data: roughly
   centered at 0, sane spread (report mean/σ/quantiles; a mean far from 0
   or a degenerate spike means a broken critic).
3. Spot-check **10 hand-read decisions with extreme |Â|** — the sign must
   be defensible to a human reader (e.g. "took the losing attack with
   lethal on board" should get Â < 0). An audit script dumps the decision
   context (turn, select type/context, options, chosen action, outcome,
   V, Â) for human reading.

**If the critic fails the audit: the advantage arms are BLOCKED; run
E-fallback (registered below) and say so loudly in every report.**

**STATUS 2026-08-04 ~09:50 — BOTH critics trained, BOTH blocked; fallback
path engaged.**
- `critic_td` (one epoch, frozen-target TD(0) after fixing a measured
  live-bootstrap divergence, MSE 0.30→284 by step 21k): audit (i) FAIL /
  (ii) FAIL / (iii) FAIL — one epoch ≈ 27 target refreshes cannot propagate
  terminal signal through ~68-decision games; V pinned near an
  out-of-range floor for whole state clusters (mean Â +0.247).
- `critic_mc` (direct outcome regression): accuracy 65.0% vs 53.8% base —
  real ranking signal — but calibration FAILS even after clamping V to the
  valid [0,1] range at consumption (clamped MSE 0.2679 vs constant-0.5's
  0.2499; raw out-of-range rate 22.2%); (ii) PASS; (iii) extremes are
  magnitude artifacts (V=0.000 on live positions with a deck-out risk).
- **Finding:** at 9,820 episodes a 3.32M critic ranks but cannot calibrate —
  consistent with the ~50× data-scale gap to the Metamon paper, whose
  critic-based objectives this register imports. Full audits:
  `reports/critic_audit_critic_{td,mc}.md` (+ figures).
- **Consequence (registered rule applied):** E1/E2/E4 do not run. Running
  instead: E-fallback ×3 seeds, and E3 adapted to
  `efb weight × 1[avg_score ≥ 1189.0]` — the skill gate is orthogonal to
  the critic and remains the most externally-validated technique in the
  evidence base. TD-with-longer-propagation (multi-epoch, Polyak targets)
  is registered as future work, not run now (fall back, don't knob-twiddle
  past the deadline).
- **Re-trigger (keeps the Metamon schema alive, registered 2026-08-04 per
  Rami):** the corpus grows daily and the critic audit is cheap. When the
  train corpus reaches ~2× today's 15k episodes (or before any future
  Stage-2 iteration), retrain the MC critic on the enlarged corpus and
  re-run `audit_critic.py`. If it passes all three parts, the true
  advantage arms (E1 Binary / E2 Exp — the paper's actual objective
  family) UNBLOCK and take priority over outcome-weighting. The blocked
  state is a data-scale verdict, not a method verdict.

### 2.B2 The arm register — UPGRADED from v1 §2.1

Rationale for the upgrade: v1's E1/E2 weighted by **episode outcome** — a
poor-man's proxy for the Metamon paper's actual objective family (their
Table 1 / Eq. 2: a unified weighted-BC actor loss with the weight computed
from a **per-action advantage** estimated by a critic trained on the same
data). Outcome-weighting keeps winners' blunders at full weight and
discards losers' good moves; advantage-weighting scores each decision
against the state's expected value. The old outcome arms are demoted to
fallback/ablation status.

All arms: warm-start from PRIOR (`models/il_agent`), same step count
(13,000 steps ≈ one winner-filtered epoch of the restored corpus at batch
64), same LR schedule (warm 1e-4, cosine), 3 seeds {42, 43, 44}, streamed
corpus (07-01 + 07-26), loss = per-row-weighted cross-entropy over the
legal option set. The only difference between arms is the weight `w`:

| ID | Arm | Weight `w` | Notes |
|---|---|---|---|
| E0 | control | 1 (plain BC, continued) | separates "more steps + more data" from "better objective" |
| E1 | **Binary advantage** (paper's "Binary", CRR-style) | `1[Â(s,a) > 0]` | the paper's strong simple performer; discards ~half the rows — equal-steps rule is load-bearing |
| E2 | **Exp advantage** (paper's "Exp", AWR-style) | `exp(β·Â(s,a))`, clipped at 20 | β ∈ {0.5, 1, 2}; clipping guards a miscalibrated critic handing one row the whole gradient |
| E3 | **skill × advantage** | best-of(E1, E2) weight × `1[avg_score ≥ Q75]` | the manifest fix unblocks the rating field. Rows with unknown rating keep the base weight (gate applies only where a rating exists — the unknown fraction is reported); chart threshold sensitivity Q50–Q90 and per-day splits. Skill-filtering is the most externally-validated technique in the evidence base (Metamon; Orbit Wars 2nd place's 1500/1600 thresholds; 49th place's winner-states scaling) |
| E-fallback | outcome-weighted (the OLD E2) | `exp(β·(outcome − 0.5))` | **registered contingency, run ONLY if the B2a critic audit fails** — the critic-free fallback, not a primary arm |
| E4 (optional) | Binary + MaxQ (paper's fourth variant) | E1's weight, plus λ·E[Q(s,a)] term | gated on E1/E2 showing signal; leans hardest on critic quality, so it goes last |

Implementation note (already true in code): `train_il.py --weight-arm
adv-binary|adv-exp --critic-dir <dir>` are E1/E2; with V frozen at 0.5 they
collapse exactly to the old winners-only/outcome arms
(`tests/test_e4_critic.py` asserts the equivalence — the correctness anchor
for the whole weight path). E3 = a skill-gate multiplier composed on top of
the winning advantage arm. Unknown-outcome rows get weight 0 in every arm.

**Selection rule:** the local tournament (round-robin vs PRIOR + public
pool, mirrored pairs, pooled seeds, Glicko with RD quoted) and the ladder
only. The offline accuracy check is a pipeline check, never a selection
metric (it already failed to separate arms once — expected).

**STATUS 2026-08-04 12:06 — tournament run (14 agents, 12 mirrored
pairs/pairing, isolated ratings, `reports/s2v2_tournament.json`, figure
`s2v2_arms_tournament.png`). All numbers LOCAL — unverified on the ladder:**

| Arm (seeds pooled) | vs PRIOR (n=72) | vs strong trio (n=216) | vs e3 |
|---|---|---|---|
| e0 control | **79.2% [68.4, 86.9]** | **19.4% [14.7, 25.2]** | 59.7% [53.1, 66.0] |
| efb outcome-weighted | 72.2% [61.0, 81.2] | 12.5% [8.7, 17.6] | 63.0% [56.3, 69.1] |
| e3 skill-gated | 69.4% [58.0, 78.9] | 13.4% [9.5, 18.6] | — |
| (PRIOR reference) | — | 9.7% [4.8, 18.7] | — |

- Every arm beats PRIOR locally with a CI floor above 50% — necessary but
  NOT sufficient (the old E1's 62.5% did too and settled at 395).
- **e0 vs efb: statistical tie** (48.1% [41.6, 54.8]). The pre-registered
  row-count tiebreak cannot separate them (neither discards rows), so
  selection falls to the documented secondary margins: e0 nearly doubles
  efb against the strong trio and carries zero extra assumptions.
  **Selected arm: e0; champion checkpoint: `s2v2_e0_s43`** (20/24 vs
  PRIOR, best-in-family 52.7% overall, best trio count 16/72).
- **Negative result, written down: the skill gate HURT.** e3 loses to e0
  (40.3%) and efb (37.0%) with non-overlapping-from-50 CIs. Mechanism
  consistent with the registered scale-mix caveat: the pooled Q75=1189
  threshold keeps mostly 07-01 (oldest, real-rated) episodes, so the gate
  trades away both volume and recency at equal steps. The most
  externally-validated technique in the evidence base did not survive
  contact with this corpus's rating field.
- efb β-sweep {0.5, 2} NOT run — registered scope decision: efb tied e0 at
  β=1, the full-corpus experiment (below) supersedes the family, and the
  calendar prices each extra arm at ~1h + tournament time.
- E4 never triggered (its gate required advantage-arm signal). **If arms do not separate
(overlapping CIs), say so plainly and pick by the pre-registered tiebreak:
fewest training rows** (E1 ties beat E2 ties, a skill-gated arm beats an
ungated one), rather than narrating a winner.

### 2.B3 Stage-2 gate (v1 §2.2 unchanged, with the v2 bar)

1. Offline accuracy check vs the 0.381 majority line (pipeline check only).
2. Local tournament with RD quoted; enough mirrored pairs that intervals
   separate.
3. Ladder ritual: bundle build (read the printed tarball MiB) → forced-CPU
   rehearsal (`PTCG_DEVICE=cpu`, ms/decision recorded) → submit with a
   DETAILED message (what changed, from which checkpoint, expected effect)
   → ledger entry → leaderboard-check after scoring, and again at settle.
   Only a checkpoint whose **settled** score clears PRIOR's 400.0 becomes
   the Stage-3 init. If no arm passes, Stage 3 initializes from PRIOR and
   the writeup says why (v1 stop-condition, still in force).

---

## PRIOR-v2 re-base rule (registered 2026-08-04, per Rami)

The pipeline is a function of its base imitation model; a better base
upgrades every downstream stage. Two candidates are on the ladder as of
08-04: **continued imitation** (`s2v2_e0_s43`, 55246108, first read 600.0)
and **all-days imitation** (`il_agent_full_0804`, 15,032 episodes,
submitting 08-04 PM). Whichever has the higher **settled** score (~08-07)
becomes **PRIOR-v2**: Stage-3 init AND KL teacher re-base onto it (two
flags + a generation restart, ~hours), the critic re-trigger trains from
its trunk, and any future weighted-arm rerun warm-starts from it. Corpus
recovery/audits and all machinery carry over unchanged. A provisional
Stage-3 generation may run from the local champion before settle — labeled
provisional, discarded without ceremony if the settle picks the other base.

---

## Stage 3 — SELFPLAY (on-policy PPO, nothing stored)

**The core deviation from the Metamon paper, restated so it survives
rewrites:** the paper's third stage aggregates self-play trajectories into
an ever-growing offline corpus and retrains offline RL on the union — every
retrain sees all past data, so the human prior can never be forgotten. That
needs corpus storage this laptop does not have (and ADR-001 deliberately
keeps raw corpora off-disk). Stage 3 is therefore **ON-POLICY PPO**:
rollouts live in RAM, are consumed by the update, and are discarded; disk
sees only checkpoints and logs. The stated cost: PPO sees only the current
policy's states, so **forgetting the prior is a live failure mode** — the
KL anchor below is the named guard, watched via per-context KL/entropy
logging, never assumed.

Base design = v1 §3.1–3.3 as built (code wins): PufferLib 3.0 `PuffeRL`
trainer in `.venv-ppo` (`scripts/train_ppo_puffer.py`) over `PTCGGym`
(cabt driven step-wise, learner decisions only) with the Multiprocessing
backend, **one env per worker always** (the cg engine is a per-process
singleton), 50/30/20 opponent mix (mirror hot-reload / frozen league /
public trio), sampling at temperature 1.0 at rollout (never argmax),
terminal-only reward {0, 0.5, 1}, structural masking in rollout and loss
(−1e9), critic (value head) train-time only, deck fixed to the shipped
60-card list — Stage 3 improves *this deck's* policy and reports must say
so. Measured integrated throughput ~61 agent-steps/s at 8×8.

### 3.1 KL anchor, concretized (Orbit Wars 1st place; Lux AI S1 winner)

Sources, cited: Orbit Wars 1st place
<https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin>
and the Lux AI Season 1 winner
<https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021>.

- **CONTINUAL anchoring:** the loss carries `β · KL(π_θ(·|s) ‖ π_ref(·|s))`
  for the entire run (`PuffeRLPriorKL`, a verbatim vendored copy of
  pufferl 3.0's `train()` with the KL addition fenced), both distributions
  softmaxed over the **IDENTICAL legal-action mask**. A mask mismatch
  silently corrupts the anchor gradient — guarded twice: the trainer
  asserts mask agreement on the first minibatch after every prior swap,
  and a unit test (`tests/test_kl_mask.py`) proves the assert fires on a
  crafted mismatch and that KL(π‖π)=0 on identical inputs.
- **π_ref initializes from the restarted Stage-2 winner** (logged in run
  metadata `init_from` / `kl_prior`; if Stage 2 fails its gate, π_ref =
  PRIOR and the writeup says so).
- **TWO promotion gates, different purposes — do not conflate:**
  - **(a) ANCHOR promotion (cheap, internal, frequent):** every N=20
    updates, live vs frozen π_ref head-to-head, mirrored seats, ≥50 pairs
    (100 games), both sides sampling at T=1.0; live wins **strictly >70%
    of decisive games** (and ≥20 decisive) → π_ref ← frozen copy of live
    (`promotion.evaluate_gate` + `retarget_prior`; every verdict appended
    to `promotion_log.jsonl`). This ratchets the anchor forward so the KL
    pull never fights genuine improvement. Evidence it works as a brake:
    the first real KL-anchored run (62k steps, 2026-08-03) held 3 gates at
    50/51/62% — no promotion, anchor visibly bounding drift.
  - **(b) CANDIDATE promotion (expensive, rare) = v1 §3.3 unchanged:**
    non-overlapping Glicko vs current best AND still beats the public pool
    AND ladder-confirmed (settled, not first-read). **Only (b) gates
    submissions.** Two consecutive (b)-failures = the loop has stalled;
    stop and write up.
- **Value-anchor** (live-vs-π_ref win-prob cross-entropy) **DEFERRED**
  unless critic instability is observed (exploding value loss,
  sign-flipping advantages across consecutive updates).

### 3.2 Entropy schedule (Orbit Wars 3rd place: "by far the most important
training knob"; 2nd place used per-head entropy coefficients)

Source: <https://www.kaggle.com/competitions/orbit-wars/writeups/3rd-place-ab-in-den-orbit>.

- The entropy coefficient gets its own config knobs: `--ent-coef-init`,
  `--ent-coef-final`, `--ent-anneal-frac` (linear anneal over that fraction
  of total timesteps, then flat at final).
- **Stated initial schedule: 0.01 → 0.001 over the first 50% of the run.**
  Rationale: the KL anchor already owns drift control (Rami's PPO notes,
  cross-checked 08-03), so entropy starts small-but-real to keep rollout
  diversity while the anchor is tight, and anneals to near-zero so
  late-run convergence isn't fought by the bonus. β_KL and entropy interact
  — sweep β_KL first (it dominates), entropy schedule second.
- **Per-context logging every update:** mean policy entropy and mean
  KL-to-prior **per SelectContext** (the 0–48 context id in the packed
  obs), appended to `train_metrics.jsonl`. Entropy collapse in one context
  (e.g. attack-selection) with a flat aggregate is exactly the failure the
  aggregate hides; the per-context lines are the stop-signal instrument.

### 3.3 PFSP-lite (3rd place's mechanism, translated to this architecture)

v1's 50/30/20 mix stays. The 3rd-place mechanism — "fix the sampled
opponent for 2 consecutive PPO updates and harvest win-rate estimates from
the rollouts (free), instead of separate eval games for opponent
weighting" — assumes one env; with 8 parallel envs each mid-episode at
update boundaries, a literal port is incoherent. The honest translation,
keeping both properties (stable per-opponent estimates; no separate eval
games):

- **Opponent persistence:** each env keeps its drawn opponent for
  K consecutive episodes (default K=4) before redrawing — the same
  estimate-stabilizing effect the 2-update freeze bought.
- **Free win-rate harvest:** every terminal step's info carries
  `(opponent_id, outcome)`; the driver aggregates per-opponent win-rate
  EMAs from rollout data alone and logs them every update.
- **Frontier weighting:** public-pool draw weights follow the harvested
  win-rates (PFSP: overweight the strongest opponent we beat often enough
  to learn from — near-zero-win matchups yield ~no gradient under
  terminal-only reward). Weights are refreshed live via a small JSON file
  the env workers re-read periodically (the same hot-reload idiom the
  mirror opponent already uses), and every refresh is logged.

### 3.4 PPO hyperparameter table (one value + one-line justification)

| Knob | Value | Why |
|---|---|---|
| γ | **0.997** | Rami's recorded gen-2 call: adds the time preference gen-1 lacked (observed game-dragging) while staying ≈1 for a terminal-reward episodic task. |
| GAE λ | **0.95** | Standard credit horizon over ~68-decision seats; pufferl default, no measured reason to fight it. |
| clip ε | **0.2** | PPO default; gen-2's clipfrac ~0.5% says the trust region is not the binding constraint. |
| PPO epochs | **1** | Orbit Wars 3rd place: extra epochs bought instability on fresh self-play data; gen-2's under-movement is addressed by LR, not by re-chewing stale rollouts. Registered sweep {1, 3}. |
| Rollout batch | **1024 rows** (8 envs × 128 bptt-horizon) | Sized to the MEASURED ~61 steps/s: one update's data collects in ~17 s; a 4096-row batch (the v1 aspiration at the 425 rows/s probe) would starve the learner for ~67 s per update. |
| Minibatch | **512** | Two minibatches per epoch; large enough for stable advantage normalization at this batch size. |
| LR | **1e-4, annealed** | Fine-tuning a 3.32M-param prior, not training 200M from scratch; directly targets gen-2's diagnosis (clipfrac ~0.5%, approx_kl ~0.002 at 3e-5 = updates far inside the trust region, little net movement per budget). |
| Entropy coef | **0.01 → 0.001, linear over first 50%** | §3.2; the anchor owns drift, entropy owns rollout diversity, annealed so it doesn't fight convergence. |
| β (KL anchor) | **0.05**, sweepable {0.01, 0.05, 0.2} | Gen-2 measured the anchor actively bounding drift at 0.05 (KL pulled 0.60→0.37) without freezing progress; sweep brackets it an order of magnitude each way. |
| Rollout temperature | **1.0** | PPO importance ratios assume samples from π_θ; ≠1.0 silently biases the objective. |
| Optimizer | **Adam** | Recorded decision (never muon for fine-tuning a BC prior). |
| vf_coef / grad-norm | **0.5 / 1.0** | pufferl defaults; value head is train-time only, no reason measured to deviate. |

### 3.5 The KL-to-prior term, in plain language

The anchor measures, at every state in the minibatch, how far the live
policy's action distribution has drifted **from** the frozen reference's:
`KL(π_θ(·|s) ‖ π_ref(·|s))`, both distributions computed by softmaxing
logits over the SAME legal-action mask for that state, then averaged over
the minibatch and added to the loss scaled by β. β sets the leash: β → 0
is unconstrained PPO (free to forget the human prior); β large means the
policy cannot leave the prior at all. Direction matters — it is measured
from the live policy TO the reference (states the live policy actually
visits, penalized where IT puts mass the prior wouldn't). "KL = 1 nat"
means substantial average drift — roughly, the live policy's choices carry
a full nat of surprise under the prior. Interpret logged KL against the
run's own baseline trace, not an absolute bar; a sudden blowup (order-of-
magnitude jump between updates) is a stop signal, and so is entropy
collapse in any single SelectContext.

---

## Measurement protocol (binding, all stages)

After EVERY phase gate / candidate promotion: bundle → printed MiB →
forced-CPU rehearsal → submit with detailed message → ledger entry
(`scripts/submission_ledger.py`; Kaggle's 500-char cap doesn't apply to the
ledger) → leaderboard-check after scoring AND at settle (~3 days). Local
Glicko alone never declares improvement. Submissions displace the active
set (§Ladder context) — plan against the ~3-day settle time and the 08-16
close; last confident slot ≈ **08-13**. ≥3 seeds wherever arms are
compared; equal steps; RD/σ on every number; charts in `reports/figures/`.

## Stop conditions (any → halt and report)

v1's list minus the retired 1-hour rule, plus the v2 additions:

- Opponent-private field reaching the encoder from any rollout path.
- Measured steps/s too low for a meaningful update budget (report the
  number and the arithmetic).
- Stage-2 gate unmet by every arm → Stage 3 initializes from PRIOR, writeup
  says why.
- KL-to-prior blowup or per-context entropy collapse during PPO.
- **Mask mismatch between live and reference distributions** (the trainer
  assert or the unit test firing).
- **Anchor promotion flapping:** reference replaced >3 consecutive times at
  near-threshold win rates → raise the gate's game count before touching
  the 70% bar.
- **B0 data recovery impossible** (streaming and Kaggle re-fetch both fail)
  → report what was tried; do NOT silently train on the 24 surviving
  train-day episodes.
- Any design that wants to write episodes to disk beyond replay-review
  transcripts; bundle MiB over the envelope.

## Work items in order

1. **[B0]** ShardILDataset meta plumbing (`with_meta`, `winner_only`,
   manifest join at train time) + `train_il.py` hub-weighted wiring +
   `check_weight_plumbing.py --data-source hub`; manifest backfill for
   2026-07-26 via `backfill-manifest --from-hub`.
2. **[B1]** Full audit register (plumbing figure, remap check, seat pairing,
   privacy test, leaderboard baseline) — findings written down.
3. **[B2a]** `train_critic.py` TD(0) mode + critic audit script; train on
   the restored corpus; run the three-part audit; decide advantage arms vs
   E-fallback.
4. **[B2]** Arms E0/E1/E2(β sweep) → offline accuracy check (pipeline
   only) + local tournament → E3 (+ optional E4 if signal) → pick by
   selection rule/tiebreak.
5. **[B3]** Stage-2 gate: local tournament with RD → bundle → CPU rehearsal → submit
   (detailed) → leaderboard-check → **Stage-3 init declared**.
6. **[C]** Stage-3 additions to `train_ppo_puffer.py`/`pufferl_kl.py`:
   entropy schedule knobs, per-context KL/entropy logging, PFSP-lite
   (persistence + harvest + weight refresh), `tests/test_kl_mask.py`;
   then launch from the Stage-3 init with the §3.4 table; anchor gate
   every 20 updates; candidate gate + ladder per §3.1(b).
7. **[C-loop]** Snapshot → local tournament → (if decisive) ladder → league; repeat
   until two consecutive candidate failures or the calendar ends it.
