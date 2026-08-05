# Skill-filtered demonstrations: a second negative result

**Date:** 2026-08-04 · **Branch:** `claude/il-agent-feature-ablation-543b5c`
**Verdict: filtering the BC corpus by player rating does not help at this
corpus's rating range. Nothing ships; il_agent unchanged.**

## Question

Follow-up to the feature-ablation negative result
(`reports/feature_ablation/report.md`), whose diagnosis was "BC accuracy
rewards imitating the logged population." The obvious lever: make the
logged population better. Does cloning only episodes whose LOWER-rated
player (`min_score`) clears a per-day skill bar beat cloning everything,
at equal training steps?

## Design (and the confound the first attempt hit)

A pooled rating threshold turned out to be a **day filter in disguise**:
day 2026-07-01's ratings run ~60 points hotter than 2026-07-26's, so the
pooled top-50% came out 91% from one day and scored 57.6% — an ~11 pp
day-shift artifact caught after one run (archived in
`reports/skill_filter/results_pooled_confounded/`). Final design cuts at
each day's OWN median / 75th percentile, so all arms share an identical
day mix and the only treatment is within-day skill. Days with no scores
anywhere (2026-07-28..08-01, 08-03; the 07-26 shard scores were joined
from manifest.csv) are excluded from every arm including the baseline.

Arms × seeds 42/43/44, equal 4,000 steps, Hub streaming:
`unfiltered_scored` (9,441 episodes), `top50_wd` (4,778), `top25_wd`
(2,423). Two gate metrics per run on the held-out day (150 episodes):
all-eval top-1, and top-1 on the eval day's own top-25% `min_score`
subset ("does it imitate strong players better" — the outcome filtering
targets). Accept = high-skill mean paired gain > seed spread, without an
all-eval collapse.

## Result

| arm | high-skill top1 by seed (%) | mean | paired d (pp) | mean d | spread | all-eval mean d | verdict |
|---|---|---|---|---|---|---|---|
| unfiltered_scored | 66.5, 68.4, 68.9 | 67.93 | — | — | — | — | reference |
| top50_wd | 66.3, 66.9, 68.6 | 67.27 | −0.20, −1.50, −0.30 | −0.67 | 1.30 | −1.40 | drop |
| top25_wd | 62.5, 65.8, 61.6 | 63.30 | −4.00, −2.60, −7.30 | −4.63 | 4.70 | −5.60 | drop |

Monotone: the harder the filter, the worse — on BOTH metrics, including
the high-skill eval subset the filter was supposed to win. top25_wd is a
replicated, large negative (every seed −2.6 pp or worse).

## Why

1. **The corpus has no strong players to select.** Within a day the
   `min_score` band is ~1100 ± 30 — the difference between "median" and
   "top quartile" demonstrators is noise-level skill, so the filter buys
   almost no quality...
2. **...while halving/quartering the data.** At equal steps the top25 arm
   re-passes ~2.4k episodes ~7 times; the diversity loss dominates. Data
   quantity beats marginal demonstrator quality at this rating spread.
3. Same day-shift lesson as the confound: *composition* of the corpus
   (which day, hence which meta/engine version) moves accuracy by many pp
   — far more than skill filtering can at this rating range. Corpus
   curation should reason about days, not ratings.

## Combined takeaway (both experiments)

Neither better inputs (deterministic-future features, 0/7) nor better
demonstrators (skill filtering, 0/2) improves this BC policy. The binding
constraint is the demonstration corpus itself: mostly weak, tightly
clustered players. Cloning it harder, with better features or fewer
teachers, does not escape it. The levers that can: win-rate-driven
training (PPO/self-play, MCTS lines already in progress elsewhere) and
corpus growth/day curation. Recommend no further BC-input or BC-filter
experiments without new evidence (e.g. a genuinely high-rated episode
source appearing in the feed).

Infrastructure kept from this line (all merged-safe): per-checkpoint
feature configs, episode-id allowlist streaming, `--min-score` hub eval,
pruned-stub guards, and the two ablation drivers.
