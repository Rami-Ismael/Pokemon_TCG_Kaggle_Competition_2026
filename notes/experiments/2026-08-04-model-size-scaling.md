# Model size — downward scaling study

Design: **scaling study** (ablation shape 1), swept *downward*. Not a controlled
comparison, because the question is a trend, not a winner: does accuracy fall off as
capacity shrinks, or is it flat all the way down?

## The motivation I pitched, and why it is wrong

I proposed this as "a smaller model frees CPU, freed CPU becomes MCTS rollouts."
**That mechanism does not exist here.** Measured before designing:

- `agent_core_improved.py:748` — `SEARCH_TIME_BUDGET = 1.5` **seconds** per decision.
- `agent_core_improved.py:912-919` — the BC prior is computed **once per decision at
  the root**: *"Computed once (the distribution doesn't change during this search)."*
  It is not called per node.
- Single-thread CPU forward latency (`torch.set_num_threads(1)`, 300 real decisions):
  3.32M → **3.70 ms median**; 10.99M → **7.94 ms median**.

So the model costs **3.70 ms of a 1500 ms budget — 0.25%**. Dropping 10.99M → 3.32M
frees 4.2 ms, i.e. **0.28% more rollout time**. That cannot move a win rate. Any
version of this study justified by inference speed is dead on arrival.

Second time this session that a mechanism check killed a proposal after the headline
number looked promising (see `2026-08-04-pass-filter-bc.md`). **Check the mechanism
before the magnitude.**

## The question that IS worth answering

Reframed: **is this task capacity-limited or data/representation-limited?**

The repo already holds two points, and the top of the curve is flat — 3.3x the
parameters bought **+0.11 pt**:

| checkpoint | h/L/heads | params | reported eval_accuracy |
|---|---|---:|---:|
| `il_agent` | 192/6/6 | 3.32M | 0.7534 |
| `il_agent_medium` | 320/8/8 | 10.99M | 0.7545 |

Those two numbers are not trustworthy as-is: both are **forced-row inflated** (+1.5 pt,
see the other card), both come from `eval_batches=100` × batch 64 = **6,400 rows** not
the full eval day, and `il_agent_medium` has `epoch: None` so training parity is
unknown. They motivate the study; they are not its result.

If the curve is *also* flat downward, capacity is not the bottleneck anywhere in the
range we can afford, and further effort belongs in the encoder or the corpus — not in
the model. That is a real redirect, and it is what the ALA 2024 result predicts
(their 256/128 MLP beat their own LSTMs; larger nets overfit).

- **Hypothesis:** held-out contested-row top-1 is flat from 10.99M down to ~1M and only
  degrades below it, because the policy is limited by what the encoder exposes and by
  demonstration diversity, not by parameter count.
- **Independent variable:** model size. One axis, swept jointly as (hidden, layers,
  heads) — the repo's existing configs already move these together, so an
  each-separately grid would be a different (larger) experiment.
- **Baseline:** `il_agent` @ 192/6/6, 3.32M — the current shipped size, mid-grid.
- **Arms** (measured param counts and single-thread CPU latency, real rows):

| arm | h/L/heads | params | fp32 MB | median ms |
|---|---|---:|---:|---:|
| tiny | 64/2/2 | 0.31M | 1.2 | 0.78 |
| small | 96/3/3 | 0.65M | 2.6 | 1.05 |
| mid-small | 128/4/4 | 1.22M | 4.9 | 1.56 |
| **current** | 192/6/6 | **3.32M** | 13.3 | 4.44 |
| medium | 320/8/8 | 10.99M | 43.9 | 9.42 |

Spans **35x** in parameters — comfortably more than the order of magnitude the design
guide asks for. Every arm's latency is noise against a 1500 ms budget, so latency is
reported for the record only, never as a selection criterion.

- **Metric & protocol:**
  - Primary: **contested-row top-1** on the eval day (`opt_mask.sum() >= 2`), the
    unbiased metric added to `eval_rung1.py` today. Reported against the mask-aware
    `rand-legal` floor, not the mask-blind majority baseline.
  - Secondary: contested top-3; per-`SelectContext` table (capacity may matter in the
    few high-branching contexts and nowhere else — a global flat line can hide that).
  - **Equal steps, not equal epochs**, and identical data for every arm.
  - Same seed set across arms; ≥2 seeds, or the run is labelled provisional.
- **Pre-registered decision:**
  - If **tiny (0.31M) is within 2.0 pt** of current on contested top-1 → the task is
    not capacity-limited. **Stop investing in model size**; redirect to encoder
    features and corpus. Ship the smallest arm that is within 1.0 pt.
  - If accuracy degrades **monotonically** with size across the sweep → capacity is a
    real lever; the follow-up is an upward arm, budgeted against the ~197.7 MiB
    envelope (10.99M is already 43.9 MB fp32 before the rest of the bundle).
  - Mixed/non-monotonic → inconclusive, write it up, do not cherry-pick a winner.
  - No "better" claim leaves this card without `scripts/check_leaderboard.py`.
- **Cost estimate:** 5 arms × ≥2 seeds. Per-run wall clock **not yet measured** — time
  the tiny arm first and report the projection before committing to the rest.
  Run **cheapest-first (tiny → medium)**: if tiny already matches current, the
  expensive arms are answering a question that is already settled.
- **Prior work checked:**
  - `il_agent` vs `il_agent_medium` above — the two existing points.
  - Vault: *Find a relationship between a size of the model and the performance…* —
    records the Metamon 15M/50M/200M grid rescaled to 3.32M/10.99M for our data budget.
    Consistent with the arms here; this card extends the grid **downward**, which that
    note never did.
  - Haluška & Schmid, ALA 2024 (arXiv:2404.16689) — 256/128 MLP, no recurrence, beat
    their own LSTM sweep; larger nets overfit. The source of the downward hypothesis.

## Phase 0 — free data points before training anything

`il_agent` (3.32M) and `il_agent_medium` (10.99M) already exist. Re-evaluate **both**
with the new contested-row metric on the full eval day. Costs two eval passes, no
training, and it either (a) confirms the top of the curve is flat on a trustworthy
metric, or (b) shows the +0.11 pt was an artifact of the 6,400-row eval and the gap is
real. Training parity is still unknown for `medium`, so Phase 0 informs the design —
it does not substitute for the retrained arms.

## Phase 0 result (2026-08-05) — the top of the curve is flat, and the old gain was noise

Paired evaluation, both checkpoints scored on **identical** eval-day rows, 400 episodes
streamed from the Hub, **70,778 contested rows** (forced rows excluded — correct by
construction for every model).

| arm | params | contested top-1 | contested top-3 |
|---|---:|---:|---:|
| current | 3.32M | **74.00%** | 95.14% |
| medium | 10.99M | **73.89%** | 95.20% |

```
delta (medium - current): -0.11 pt
discordant pairs: current-only-right b=2301  medium-only-right c=2223
                  (agree on 66,254 rows, 93.6%)
McNemar chi2=1.31  p=0.2523   paired 95% CI on delta: -0.30 .. +0.08 pt
```

**The +0.11 pt that motivated this study flips sign to -0.11 pt and is not
significant** (p=0.25, CI spans zero). 3.3x the parameters buys nothing measurable.
The old number was an artifact of a 6,400-row eval with forced-row inflation; on
70,778 clean rows the effect is gone in both magnitude and direction.

**A detail worth more than the headline:** the two models disagree on **6.4% of rows**
(4,524 discordant) while netting out to zero. They are not converging on the same
function — they make substantially different errors of equal quantity. That is an
ensemble/distillation signal, not a scaling signal, and it is a different experiment.

- **Decision:** grid sweeps **downward** as designed. The upward arm is dropped —
  there is no evidence capacity above 3.32M buys anything, and 10.99M costs 43.9 MB
  fp32 against a ~197.7 MiB envelope for nothing.
- **What we learned:** **a paired design turned an unresolvable question into a settled
  one at feasible cost.** The two arms agree on 93.6% of rows; all the information about
  the difference lives in the 6.4% where they disagree. Two independent runs would have
  compared 74.00% ± sampling noise against 73.89% ± sampling noise and concluded
  nothing. Same data, same wall clock, decisive answer — because the comparison was
  made row-by-row instead of mean-to-mean.

### Protocol caveat found while running this

Contested top-1 measured **76.1%** on 100 episodes (yesterday) and **74.00%** on 400
episodes. At n=20,888 the SE is ~0.3 pt, so a 2.1 pt shift is ~6 SE — **not sampling
noise.** `--max-episodes N` takes a deterministic **prefix** of the sorted Hub shards,
not a random sample, so a small N is a *biased* slice of the eval day, not a small
unbiased one.

Consequences: (1) never compare accuracy numbers taken at different `--max-episodes`;
(2) all arms in the main sweep must use the **same** episode count; (3) the paired
design is **immune** to this — both models saw the identical prefix, so the bias
cancels exactly in the delta. Which is a second argument for pairing beyond power.

## Result (main sweep) — 2026-08-06

Four arms trained from scratch, **identical on every axis**: 38,562 steps (verified
from each `train_metadata.json`, not assumed), seed 42, `--data-source hub`,
`--num-workers 4`, same post-merge code. Evaluated paired on **70,778 contested rows**,
400 eval episodes, every arm scoring the same row in one pass.

| arm | params | contested top-1 | top-3 | vs rand-legal floor |
|---|---:|---:|---:|---:|
| tiny | 0.31M | 68.62% | 92.44% | +45.19 |
| small | 0.65M | 70.61% | 93.61% | +47.19 |
| mid-small | 1.22M | 72.37% | 94.45% | +48.94 |
| **current** | **3.32M** | **74.46%** | 95.28% | +51.03 |

rand-legal floor 23.42% (Phase 0 measured 23.6% on the same day — consistent).

```
pairwise (McNemar, continuity-corrected)
         small -> tiny      -2.00 pt   b=4169 c=2755  p<0.0001  CI -2.23 .. -1.77
     mid-small -> tiny      -3.75 pt   b=5581 c=2927  p<0.0001  CI -4.01 .. -3.49
     mid-small -> small     -1.75 pt   b=3695 c=2455  p<0.0001  CI -1.97 .. -1.53
       current -> tiny      -5.84 pt   b=7014 c=2879  p<0.0001  CI -6.12 .. -5.57
       current -> small     -3.84 pt   b=5069 c=2348  p<0.0001  CI -4.08 .. -3.61
       current -> mid-small -2.09 pt   b=3498 c=2017  p<0.0001  CI -2.30 .. -1.89
```

**Every pairwise gap is significant and every CI excludes zero.** Monotonic, no
plateau anywhere in the swept range.

Per doubling of parameters:

| step | ratio | delta | per doubling |
|---|---:|---:|---:|
| 0.31 → 0.65M | 2.11x | +2.00 | +1.85 |
| 0.65 → 1.22M | 1.87x | +1.75 | +1.94 |
| 1.22 → 3.32M | 2.72x | +2.09 | +1.45 |
| 0.31 → 3.32M | 10.8x | +5.84 | +1.70 |

**Roughly log-linear at ~+1.7 pt per doubling, with no sign of flattening.** My earlier
"diminishing returns" read came from the inflated in-script metric and does not survive
the clean one.

- **Decision:** the pre-registered rule fires against the hypothesis. Tiny is **-5.84 pt**
  from current, far outside the ±2.0 pt band, so the task **is capacity-limited** in this
  range. The ALA 2024 "small MLP suffices" result **does not transfer** — that finding
  came from distilling a single unimodal RL agent's self-play, a far easier target than
  our heterogeneous corpus, exactly the direction of bias flagged when it was first cited.
  Do not shrink the model. 3.32M is not oversized.
- **What we learned:** the headline is the shape, not any single number. A clean
  log-linear trend with no knee inside 10x of swept range says the bottleneck in this
  band is capacity, not the encoder or the corpus — which is the opposite of the
  redirect this card was written to test for.

### ⚠️ This invalidates the Phase 0 conclusion — reinstate the upward arm

Phase 0 found 3.32M → 10.99M = **-0.11 pt, p=0.25**, and I dropped the upward arm on it.
That now looks wrong. At the sweep's measured +1.70 pt/doubling, 3.32M → 10.99M
(1.73 doublings) predicts roughly **+2.9 pt**. Phase 0 observed **-0.11**. A miss that
large against an otherwise clean log-linear trend is more likely a defect in the
comparison than a real cliff.

The defect is identifiable: **Phase 0 compared two *shipped* checkpoints with unknown
training parity** — `il_agent_medium` has `epoch: None` and no recorded step count, so
there is no evidence it ever trained the 38,562 steps the sweep arms did. An
undertrained 10.99M arm would land exactly where Phase 0 put it.

Phase 0 was still worth running — it was free and it correctly redirected the sweep —
but its conclusion cannot bear weight. **Follow-up: train a 10.99M arm (320/8/8) at
38,562 steps, seed 42, same data, and re-run the paired evaluation with all five arms.**
If it lands near +2.9 pt the curve is log-linear throughout and the shipped model is
undersized; if it lands flat, the knee at 3.32M is real. Budget the answer against the
~197.7 MiB envelope: 10.99M is 43.9 MB fp32 before the rest of the bundle.

Anything trained-vs-shipped is not a controlled comparison. Same trap as
`il_agent_medium` in the first place, and I walked into it a second time.
