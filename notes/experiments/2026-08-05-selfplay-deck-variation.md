# Self-play with deck variation on both sides

**Status:** designed, code landed, NOT YET RUN. Date: 2026-08-05.

## Hypothesis

Every RL result this project has was measured with our agent piloting one deck
against opponents piloting the *same* deck. Verified, not assumed:
`run_selfplay_g1/g2/g3.sh` pass no deck flags, so `PTCGGym.deck` fell back to
`selfplay.load_deck()` — `agents/il_agent/deck.csv`, Mega Lucario ex — and a
checkpoint opponent independently answered the engine's deck request with
`SamplingPolicy.deck`, the same list. Three generations of self-play learned one
mirror matchup.

**H:** that is why local self-play wins keep failing to transfer to the ladder
(three recorded local-vs-ladder inversions). A policy trained in one matchup
overfits to it.

**Treatment:** the learner's deck AND the opponent's deck are drawn
independently and uniformly per episode from a pool of legality-verified lists.
The diversity axis is the DECK. Opponents stay lineage-only.

## Design decision: self-play stays pure

Self-play means the agent plays its own **lineage**: the live mirror plus our
own frozen checkpoints, nothing else. Externals (rule baselines, other
competitors' submissions) are never training opponents, so they stay unseen and
the local evaluation pool remains a real out-of-sample filter.

The env-level bucket is deleted, not zeroed — a zeroed bucket `_draw_opponent`
could still fall through into is how externals would quietly re-enter. On top of
that, three CLI guards make a reintroduction a **loud failure** rather than a
silent one, because that is how it would actually arrive: as flags from another
script or another line of work.

| guard | rule | rationale |
|---|---|---|
| `--mix` | last share is always `0`; nonzero is a hard error | three-number mixes are written all over the repo; silently dropping a nonzero third is worse than refusing it |
| `--field-pool` | must be empty | accepted only so a field pool of external agents fails loudly instead of training against them |
| `--league` | every entry must be a dir with a `config.json` | an agent NAME here becomes `("ckpt", "wmh_grimmsnarl")`, which fails inside a spawned worker — that seat crashes every episode, i.e. free wins for the learner, and nobody sees an error |

`--opponent-module` remains the one deliberate way to train against a single
frozen external, and it is labelled exploiter mode. Pinned by
`tests/test_lineage_only.py`.

## Deck distribution — uniform, deliberately

Sources: `configs/deck_lists/*.csv` (8 curated) ∪ `agents/*/deck.csv` (49 files).
After content dedup and the legality gate: **K = 29**.

Uniform over pool members, one draw per seat per episode. We have no ladder
meta-share data, so uniform is the honest choice and it targets worst-case
robustness. **A policy trained this way has NOT been trained on the meta
distribution.** Meta-weighting is a separate experiment.

Uniform over *lists* is still not uniform over *strategies*:
`reports/deck_pool_census.json` puts 29 lists into 17 archetypes, heaviest
(Fezandipiti ex, Mega Lucario ex) at 10.3% of draws each. Report it that way.

### Legality gate

`scripts/probe_deck_legality.py --deck-pool all:decks` → 29/29 pass (submit +
full play-out). An illegal deck returns INVALID at deck submission and hands
away every episode it is drawn for, which would manufacture a result.

Two fixes the gate needed on the way:
- The 400-step play-out cap reported `kiyotah_iono` as a timeout in a batch run
  and passed it at 165 steps alone — game length under safe-choice agents is
  high-variance. Cap raised to 1500; re-probe any timeout alone before believing
  it.
- Manifests stored absolute paths, baking in whichever worktree generated them.
  Now repo-relative.

### Dedup bug found by the census

`_dedup_by_content` hashed *file bytes*, so lists identical up to card order or
whitespace survived as separate pool members. Four such groups existed among the
35 legality-passed lists (e.g. `tb_starmie` and `wmh_froslass`, jaccard 1.000).
Now keyed on the sorted 60-card multiset: **35 → 29**. Left in, they would have
handed their archetypes extra weight under a flag whose entire point is a
controlled uniform draw.

## Run design

Two arms, **equal wall-clock and equal seeds** (`--seed`, new flag; previously
the seed was hardcoded to `config.RANDOM_SEED` and arms could not be matched).

Budgeted by wall-clock, not by a pinned step count: PR #55 (merged 2026-08-06)
rescinded repo standing rule 4, and the argument applies directly here. The
treatment arm's games are ~1.3x longer (pool median 156 engine steps vs 119 for
the control deck), so an equal-STEP budget would hand it ~1.3x the compute —
exactly the distortion that PR's own evidence describes. `--total-timesteps` is
set far above reach so `--max-seconds` binds; `global_step` is recorded per run
as the throughput diagnostic.

| arm | flags | both seats' decks |
|---|---|---|
| control | *(no deck flags)* | the single hardcoded deck, forever |
| treatment | `--deck-pool '@configs/deck_pools/legal_decks.txt'` | two independent uniform draws per episode |

Deck randomization makes the task strictly harder, so a lower treatment win rate
is **unreadable without the control**. `--mirror-deck` remains available as the
same-deck-both-seats control for measurements that must exclude deck matchup
(the exploitability sweep).

Driver: `scripts/run_selfplay_g4_deckvar.sh` (arms run sequentially — never two
MPS jobs at once).

## Reporting

**Not the pooled win rate.** It is exactly the statistic that hides a policy
dominating one deck and losing the rest. The env emits three cuts at every
terminal step — `wr_<opponent>`, `wrd_<learner deck>`,
`wrm_<learner>~<opponent>` — and `scripts/report_deck_arms.py` prints them
worst-first with n and binomial SE.

**Read n before reading a win rate.** The matchup grid is K² = 841 cells while
episodes grow only linearly with compute, so most matchup cells will be in the
single digits and a 0.000 there is noise. Cells below `--min-n` are marked
UNREADABLE and excluded from the worst-case verdict. The per-deck cut (29 cells)
is the one that will actually be readable.

Training-rollout rates average over a moving policy and a moving mirror; they
monitor the run, they do not compare the arms. The arm comparison is a post-hoc
evaluation of the final checkpoints on a fixed deck grid against fixed
opponents.

## Also fixed: the promotion gate was single-deck

`promotion.evaluate_gate` played live-vs-reference on `selfplay.load_deck` only.
With a deck pool that would have ratcheted the KL reference on the Mega Lucario
ex mirror while the policy trained on 29 decks — selecting for exactly the
overfit this experiment exists to remove. The gate now samples decks the same
way the env does, seeded per game index so the two games of a mirrored pair are
the same deck matchup with the seats swapped.

## Evaluation — no held-out split, by decision

1. Fallback diagnostic on the resulting checkpoint before trusting any number
   it produces.
2. Local Glicko against the anchored pool. Its externals were never training
   opponents, so this is genuinely out-of-sample — but it is a **go/no-go on
   whether the checkpoint earns a submission slot**, not evidence of
   improvement. Local rankings have inverted against the real ladder three
   times.
3. Refresh the submission ledger (concurrent sessions submit the same
   artifacts), submit with a detailed message, read the ladder. That is the
   measurement.

## Removed in this change (TASK 1)

The public-pool bucket, outright: the third branch of `_draw_opponent`, the
hardcoded `pool_names` trio, `_maybe_reload_weights` + the `pool_weights.json`
hot-reload, `--pool-weights`, `--pfsp-refresh-every`, `--pfsp-ema`, and
`POOL_NAMES`. The env's `mix` is two shares; the CLI still accepts a third,
pinned to zero and hard-erroring otherwise (see the guard table above).

`selfplay.sample_league` was **not** dead code as believed — `scripts/train_ppo.py`
and `scripts/probe_selfplay_throughput.py` both called it. Replaced by
`sample_lineage_opponent` (two buckets) and both callers updated, rather than
deleted and left broken.
