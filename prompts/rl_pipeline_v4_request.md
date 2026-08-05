# Prompt — RL Pipeline v4 (paste into Claude Code)

/engineering:system-design

## What I want

Produce `prompts/rl_pipeline_v4.md` — a design doc, not code. It supersedes
`prompts/rl_pipeline_v2.md` and the never-run v3 request of 2026-08-04
(leave v1 and v2 on disk unchanged for diffing). This request folds in my
comments from 2026-08-04 and 2026-08-05.

Read first, in this order, before writing a line:

1. `prompts/rl_pipeline_v2.md` — the current plan. v4 is a **revision of it**,
   not a from-scratch rewrite. Anything v2 already settles with a measurement
   stays settled; carry it forward and cite the measurement.
2. `prompts/rl_pipeline_v1.md` — for the STATUS blocks: PufferLib-PPO gen 1
   (`ppo_u120832`, ladder first read 532.2) and gen 2 (NOT promoted) already
   ran. v4 starts from that reality, not from zero.
3. `prompts/bc_pipeline_v2.md` §8 — the RL-readiness notes.
4. The Metamon paper: *Human-Level Competitive Pokémon via Scalable Offline
   Reinforcement Learning with Transformers* (Grigsby et al., RLC 2025,
   arXiv:2504.04395). The three-phase shape is theirs:
   **IL prior → offline RL reweighting on the same corpus → self-play
   fine-tuning.** Use their phase names.

The one deviation from the paper, accepted knowingly — do not re-litigate:
**Phase 3 is on-policy PPO, not the paper's offline aggregation over
self-play generations.** The original reason was disk; the honest current
reason is that storing and re-training over generations of self-play episodes
is still not something this laptop workflow supports (the human corpus now
lives on Hugging Face and streams — the laptop is not a corpus store). I
accept the trade — online PPO instead of offline RL on aggregated self-play —
and I want v4 to state what the trade costs (forgetting the prior is a live
failure mode) and how the cost is watched (the KL anchor and its logs), not
to argue about whether to make it.

## Tooling decisions already made — carry forward, don't reopen

- **PufferLib is the Phase-3 engine**: its vectorized environments AND its
  own PPO implementation (`scripts/train_ppo_puffer.py` already runs PuffeRL).
  Do not hand-roll PPO math. Check what the **latest** PufferLib release is
  (jsuarez keeps improving it), report whether we can move off the pinned 3.0
  (4.x was C-env-only at last check), and keep us as current as the cabt
  integration allows.
- **Reward: win = 1, anything else = 0.** Terminal only, no shaping. Note:
  v2 mapped draws to 0.5 — this is a deliberate simplification; state it as
  a change and keep the standing rule that a loss is never −1 (negative
  terminal reward under γ < 1 pays the agent to delay losing).

## The gaps v4 exists to close

### 1. The anchor: Orbit Wars 1st-place frozen-teacher KL, with the ratchet

v2's "KL penalty to the frozen prior" is a *static* anchor. I want the Orbit
Wars 1st-place mechanism (IsaiahPressman — search Kaggle for "orbit wars 1st
place solution scaling reinforcement learning" if you need the writeup)
specified in full:

- The KL/value pull is toward the **previous-best checkpoint**, not a fixed
  IL prior, and it is **continual — on for the entire run**, not a warm-up
  that decays off.
- **Promotion rule:** periodically the live model plays evaluation matches
  head-to-head against the frozen best. If the live model wins **more than
  70%** of those games, a frozen snapshot of it becomes the new best and
  training continues anchored to that new reference. If it doesn't clear
  70%, the old reference stays. The bar is 70% and not 50% so promotion
  cannot fire on noise. Connect this explicitly to the double-oracle /
  promotion ideas in ByteRL — *Mastering Strategy Card Game (Legends of Code
  and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play*
  (arXiv:2303.04096) — that's where my vocabulary comes from.
- Two anchor terms, not one: a policy-KL anchor and a value/win-prob anchor
  to the same frozen reference (the value term ships only if critic
  instability is actually observed).
- Both π_θ and π_ref must apply the **identical action mask before softmax**,
  or the KL puts mass on illegal actions — correctness requirement with a
  test, not a footnote.
- One sentence distinguishing the anchor from PPO's own clipping (which is
  against π_old, refreshed every rollout, and does a different job).

### 2. Answer the KL-scale question in the doc

I need to read the training logs and know what I'm looking at:

- What are the **units** of the logged KL-to-reference — nats per decision,
  averaged over what, against *which* reference (initial IL prior, current
  promoted best, or both logged separately)?
- **What does KL = 1 actually mean?** How far is the policy from the
  reference at 0.01 / 0.1 / 1.0 / 10 — which is healthy drift, which is "the
  anchor stopped binding," which is the stop signal?
- Log KL **per SelectContext**, not just a global scalar, and state the stop
  rule in numbers.

### 3. Explicit PPO hyperparameters — planned values, not search ranges

v4 must contain a table of the **starting values you actually plan to run**,
one row per knob:

| knob | planned value | why this value (source / measurement) | swept later? |

Cover at minimum: learning rate, clip ε, γ, GAE λ, value coefficient,
entropy coefficient, KL-anchor β, update epochs, minibatch size, rollout
buffer size, rollout sampling temperature, parallel envs/workers, promotion
cadence, and the number of head-to-head games behind the 70% test. Mark each
value as (a) a PufferLib default kept, (b) a published self-play-PPO
convention, or (c) derived from a measurement in this repo — and where it's
a guess, say "guess." Fold in the gen-2 diagnosis already on record
(clipfrac ~0.5% → LR too low for the budget; critic re-init cost;
anchor drag).

Also answer directly: **what are typical hyperparameters for self-play PPO**,
and where do self-play defaults diverge from single-agent PPO defaults?
Terminal-only reward argues γ ≈ 1; games are ~68 decisions/seat so λ
interacts with credit assignment at that length; the entropy coefficient is
secondary once the KL anchor owns drift control.

### 4. League composition — the weak-opponent reward bias

The current Phase-3 mix is ~50% mirror self-play + ~50% league (frozen past
checkpoints + public pool agents — fictitious-self-play shaped, OSFP-style).
My worry, stated as a hypothesis for v4 to address with a design:

> The public pool deliberately contains both weak and strong agents. A
> learner rewarded per win can inflate its win-rate by farming the weak
> agents while learning nothing — or getting worse — against the strong
> ones. I suspect a version of this already happened in the earlier
> Metamon-style attempt.

v4 must specify an opponent-sampling scheme that neutralizes this —
PFSP-style weighting (overweight the strongest opponents we sometimes beat,
e.g. kiyotah; underweight opponents we already beat >90%), or explicit
per-opponent reward normalization — and name the per-opponent win-rate logs
that would show farming if it happens.

Related, and I want a verdict, not a dodge: **should a local-pool benchmark
gate Kaggle submissions at all in Phase 3?** Tension to resolve: the
anchored 8-agent pool reached Spearman ρ +0.929 vs the ladder, but local
rankings have inverted against the real ladder three separate times, and
gating costs submission slots that expire daily. Separate the two roles
cleanly: the *training league* (wants diversity, including weak agents) vs
the *evaluation pool* (wants ladder-predictiveness only).

### 5. Deck robustness — a new axis, and where it goes

A big suspected error source: in self-play the agent only ever fights the
decks in its league, then meets unseen decks on the ladder. v4 must design a
deck-diversity mechanism:

- **Build a meta-deck list.** Sources: every distinct deck carried by the
  local pool agents; plus decks mined from the episode corpus — when an
  opponent's deck.csv isn't published, reconstruct it from episodes where
  they played out their deck. (Standing trap: public decks can be
  cg-illegal — e.g. >1 ACE SPEC → INVALID every game — so probe
  `battle_start` legality before a deck enters the list.)
- **Randomize the opponent's deck during self-play rollouts**, drawn from
  that list, possibly with small mutations (swap one card) to widen coverage.
- **Decide the placement**: is deck randomization a phase of its own between
  Phase 2 and Phase 3, or a knob inside Phase 3's opponent sampler? Give a
  recommendation with the reasoning.
- The goal is an agent that is good against deck combinations it never
  trained on — say how that generalization gets measured (held-out decks).

Separately, the **pilot-deck question**: I wanted to switch our own deck from
Lucario to Marnie's Grimmsnarl ex (78.4% vs 29.7% local win-rate). But the
2026-08-05 ladder read **falsified the first attempt**: `il_alldays_0804` on
Grimmsnarl read 314.1 vs 418.0 for the Lucario-deck comparator (submission
55270787). Do not assume either deck; treat pilot-deck choice as an open
experiment under the `deck-selection` skill, gated on settled ladder reads.

### 6. Data refresh — the corpus grew, and it streams now

- The training corpus is **past the old 9,820 episodes** and lives on
  Hugging Face (`Rami/ptcg-episodes`) — the ONLY copy. Training streams it
  (`--data-source auto --num-workers 4`); the local raw split dirs are
  stubs — never point a trainer at them.
- The Kaggle index dataset
  (https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index)
  carries per-episode avg/median score — the rating source for any
  skill-weighted arm.
- **The all-days IL retrain already happened**: `il_alldays_0804` (no
  skill-gate arm, per my 08-04 note) was trained and submitted 2026-08-04.
  Its first read is the 314.1 above — confounded with the deck switch, so
  fold in its *settled* read on the ledger before drawing conclusions.
- v4 must specify a **retrain cadence**: as new episode days land on the Hub,
  when does the IL prior (and therefore the Phase-2/3 warm starts) get
  refreshed, and what does a refresh invalidate downstream?

### 7. Ladder — a standing daily task, not a gate clause

Carry the whole Ladder section forward from the v3 request; it is still the
gap I care most about. Today is 2026-08-05, the ladder closes **2026-08-16**,
ratings need ~3 days to settle, so **~2026-08-13 is the last readable
submission**. Compute the remaining slot budget, maintain a ranked submission
queue (untested seeds, KL-β variants, intermediate PPO checkpoints), one
control per submission, detailed submit messages, ledger entry after every
submit, read-back after settling — and **after every phase of this pipeline,
the trained model goes to the ladder and the doc records its Glicko before
the next phase starts.** A slot that expires unused is discarded information.

## Deep research pass

Before writing v4, run deep research and fold the findings in — do not just
cite the notes I already have:

- Self-play PPO hyperparameter conventions in 2-player zero-sum
  imperfect-information games.
- The latest PufferLib release: PPO API surface, vectorization backends,
  Protein changes, ARM64/macOS issues, and whether the 3.0-pin reasons
  still hold.
- KL-regularized / behavior-regularized RL — the canonical umbrella term for
  the anchor, so the doc's vocabulary matches the literature.
- **DeepNash / R-NaD** (arXiv:2206.15378) — the theoretically grounded
  version of this anchor+promotion loop for PTCG's game class. Verdict:
  should Phase 3 be "PufferLib rollouts + R-NaD's regularization schedule"
  instead of PPO + a hand-set β?
- **ByteRL / OSFP** (arXiv:2303.04096) — closest published domain match;
  this is also the source of the double-oracle promotion framing in gap 1.

Then update the vault note *Comments on the second edition of running it
produce instriction* — say what in it is superseded and what still stands.

## Constraints that are not negotiable

Restate these in v4 so the doc is self-contained:

- `resolve_device()` only, no CUDA branch. Training on MPS; the evaluator is
  CPU-only (~1.6 vCPU, ~197.7 MiB). `torch.set_num_threads(1)` first.
- Everything under `uv run`. Paths from `pokemon_tcg.config`. Seed 42.
- Action masking is **structural** (Pattern-B option-scoring head) and part
  of the objective — in the rollout, the loss, and both sides of the KL.
- **No opponent-private information reaches the encoder** — rollouts feed
  the acting agent's `obs_dict` as the ladder serves it.
- Disk: ~98 GB free as of 2026-08-03, but the HF dataset is the only copy of
  the corpus — never delete or overwrite raw episode data without asking.
  No phase writes an episode corpus; checkpoints, TB logs, figures, and a
  few transcripts only.
- Critic is train-time only; the shipped bundle contains the actor alone.
- Every comparison: ≥3 seeds, equal **steps**, an RD or σ on every number, a
  chart in `reports/figures/`, the control beside the claim.
- Long runs don't stop for approval — the 1-hour gate is retired. Report the
  projection, chain runs so MPS is never contended, nice everything, cap
  workers, keep the laptop responsive.
- Plain checkpoint names (method-data-seed words), roles via
  `model_registry.py` — no opaque codes.

## Deliverable

`prompts/rl_pipeline_v4.md`, containing: §What changed from v2 (numbered);
the phase map with the paper's names and each phase's gate; §Anchor (gap 1 +
the KL-units subsection, gap 2); §Hyperparameters (the planned-values table,
gap 3, with the Protein sweep space as a second table marked "later, not
now"); §League (gap 4, with the local-pool-gate verdict); §Decks (gap 5);
§Data (gap 6); §Ladder (gap 7, with the slot arithmetic and running results
table); §Research findings (incl. the R-NaD verdict); §Work items in order
with the standing daily Ladder item and explicit stop conditions.

**Before anything else runs, tell me what I can submit to Kaggle today** —
refresh the submission ledger first (concurrent sessions submit too), then
give me the queue for the next three days.

Before writing, tell me the **three design decisions you are least sure
about** and what evidence would settle each. If the research contradicts
anything I asked for above, say so plainly with the numbers — I would rather
change my mind than have the doc quietly agree with me.
