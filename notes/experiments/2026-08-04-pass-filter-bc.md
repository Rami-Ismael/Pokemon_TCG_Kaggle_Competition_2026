# Forced-move / DECLINE filtering in the BC corpus — controlled comparison

Design: **controlled comparison** (2 arms, one variable), not an ablation — the
question is "is X better than what we have?", not attribution across n components.
A measurement-validity sub-check rides along (see §Result B).

- **Hypothesis:** Removing decisions with no real choice from the BC training corpus
  raises held-out top-1 accuracy on *contested* decisions, because forced rows
  contribute gradient that the inference-time action mask already guarantees — they
  teach the network to spend capacity re-deriving legality instead of preference.
- **Independent variable:** the corpus filter. Exactly one thing changes: whether
  rows with effective-choice ≤ 1 are dropped from the training set.
- **Baseline:** current `il_agent` BC recipe, unfiltered corpus, `train-2026-07-26`.
  Same architecture, same steps, same seeds, same encoder.
- **Metric & protocol:**
  - Primary: held-out top-1 on `eval-2026-07-27`, **computed on contested rows only**
    (effective-choice ≥ 2) so the two arms are scored on the same denominator.
    Reported next to the per-context majority-class baseline, per `eval_rung1.py`.
  - Secondary: top-3, and the per-`SelectContext` breakdown (the filter's mass is not
    uniform across contexts — a global null can hide a real per-context effect).
  - 3 seeds per arm. **Equal steps, not equal epochs** — the filtered arm has ~6%
    fewer rows, so equal-epoch would secretly under-train it.
- **Pre-registered decision:** adopt if contested-row top-1 improves by **≥ 1.0 pt**
  averaged over 3 seeds *and* the per-seed ranges do not overlap. Drop if |Δ| < 1.0 pt.
  Anything in between is inconclusive and gets written up as such.
  No "better" claim leaves this card without `scripts/check_leaderboard.py` (local
  ordering has inverted on the real ladder twice).
- **Cost estimate:** 6 BC runs (2 arms × 3 seeds). Per-run wall clock to be timed on
  the first run before committing to the rest; the filter itself is a dataset flag.
  Run cheapest-first: seed 42 both arms, read the Δ, then decide whether to pay for
  seeds 43/44.
- **Prior work checked:**
  - Haluška & Schmid, *Learning to Beat ByteRL*, ALA 2024 (arXiv:2404.16689) — the
    source of the idea; their pass-filter cut 3M → 2.3M pairs (**23% of the corpus**)
    and was worth ~+7 win-rate points.
  - `notes/il_v2_todo_from_exploiter_diagnostic.md` — passes-with-attack-available
    measured at ~1.3%, "not a weakness". That is an *agent-behavior* measurement at
    inference; this card measures *corpus composition* at training time. Different
    quantity, but it lowers the prior that our agent over-passes.
  - Feature-ablation and skill-filter negative results — two prior corpus-level
    interventions that did not help. This is a third of the same family; the prior
    should be pessimistic.

## Feasibility measurement (done before designing — this is the load-bearing number)

**947 episodes of `train-2026-08-03`, 155,077 decisions** (163.8 per episode), via
`iter_decisions`:

| quantity | share |
|---|---|
| `DECLINE` label (the literal pass/no-op analogue) | **0.76%** |
| effective choice ≤ 1 after exclude-masking (**forced**, no real choice) | **6.71%** |
| multi-select (`maxCount > 1`) | 12.72% |

Effective-choice histogram: 1 → 6.71%, 2 → 14.06%, 3 → 12.22%, 4 → 11.51%,
5 → 12.12%, 6+ → 43.38%.

> **Stub-split trap, walked into and corrected.** The first pass of this measurement
> ran on `train-2026-07-26` asking for 300 episodes and silently got **24** — that
> folder is a pruned stub (24 of 4,554 registered), and `iter_decisions` globs
> whatever is there without complaining. It reported DECLINE 1.26% / forced 6.06%.
> The real numbers above differ most on DECLINE (0.76%, ~40% lower).
> `train-combined-0701-0726` is worse than a stub: **9,820 dangling symlinks** into
> the pruned `train-2026-07-01`, which yields exactly 1 decision and no error.
> Only `train-2026-08-03` (947 files) is real on disk; everything else lives on the
> Hub (ADR-001). **Always print the episode count actually consumed, not the count
> requested.** This is the same trap the feature-ablation experiment hit.

**The direct port of the LOCM filter is near-vacuous here: 0.76% vs their 23%.**
Dropping 1.26% of rows cannot plausibly move a metric by 7 points, or by 1. So the
literal replication is not worth running, and this card tests the *mechanism* instead
of the *implementation*: filter forced moves (6.06%), which is the closest thing in our
corpus to "rows where the demonstrator exercised no preference."

Even 6.06% is roughly a quarter of the LOCM filter's mass. Stated up front so the
result is read correctly: **a null here does not refute the LOCM finding**, it says the
intervention does not have enough mass in our corpus to reproduce it.

## Result A — does filtering help training?
- **Observed:** not run. Result B (below) removed the mechanism, not just the mass.
  The model already scores **100%** on forced rows *by construction* — one scoreable
  slot, and `il_model.py:227` does `logits.masked_fill(~opt_mask, -inf)`. Rows the
  model gets right by construction carry near-zero loss, therefore near-zero gradient.
  So filtering them removes ~6.7% of rows that were contributing ~0% of the learning
  signal. The hypothesis on this card — "forced rows make the network spend capacity
  re-deriving legality" — is false as stated: the architecture never lets those rows
  express a wrong answer, so there is nothing to re-derive.
- **Decision:** **dropped before running.** 6 training arms not paid for.
- **What we learned:** two independent reasons to kill it, and the second is the
  interesting one. (1) *Mass:* our analogue is 0.76% vs the ALA paper's 23%. (2)
  *Mechanism:* architectural action masking already neutralizes the rows the filter
  would remove. LOCM's PASS is a real action a policy can wrongly select; our forced
  rows are not selectable-wrongly at all. **A data-cleaning trick only transfers if the
  target architecture leaves the same failure mode open.** Checking the mask before
  the mass would have killed this in one grep.

## Result B — measurement-validity sub-check (independent of A)

`scripts/eval_rung1.py:138` scores *every* decision, including the 6.06% forced rows.
Those are free correct answers for the model **and** for the majority-class baseline,
so they inflate both absolute numbers. The reported gap-vs-majority is roughly
preserved, but the headline top-1 is not a measure of decision quality on the rows
that matter.

This is worth knowing whether or not Result A is a null, and it costs one eval pass
with a row filter — no training.

**Observed.** `models/il_agent` (3.3M params, 3ep), eval day `eval-2026-07-27` streamed
from the Hub, 100 episodes / **20,888 rows**. Forced share 6.45%, consistent with the
6.71% measured on the train day.

```
    bucket        n   share  rand-legal%  majority%    top1%    top3%  gap-vs-rand
    forced     1348   6.45%       100.0%      97.6%   100.0%   100.0%  +0.0%
 contested    19540  93.55%        23.6%      34.8%    76.1%    95.7%  +52.5%

GLOBAL        20888              (n/a)         38.6%    77.6%    96.0%
headline top-1 77.6% (all rows) vs 76.1% (contested only) -- +1.5 pt inflation
```

Three things, in increasing order of importance:

1. **Headline top-1 is inflated by +1.5 pt.** Real, modest, now quantified. Every
   historical top-1 in `scores.md` and the training metadata carries this bias
   (`train_metadata.json` records `eval_accuracy` 0.7534 — same inflation applies).
2. **Forced rows are worth exactly nothing.** Model 100%, mask-aware floor 100%,
   gap **+0.0%**. Not "nearly free" — provably free.
3. **The majority-class baseline is broken on forced rows.** It scored **97.6%**,
   *below* the 100% random-legal floor, because it picks a per-context label INDEX and
   never sees `opt_mask`. So the old "+2.4% gap vs majority" on those rows was a pure
   artifact of comparing a mask-aware model against a mask-blind baseline. This is why
   the new table reports **gap-vs-rand-legal**, not gap-vs-majority, as the headline.

On the rows that matter the model is genuinely strong: **76.1% vs a 23.6% mask-aware
floor, +52.5 pt.** Dropping the free rows makes the model look *better* relative to its
floor, not worse — the old global gap-vs-majority (+39.0) understated it.

- **Decision:** **adopted.** `scripts/eval_rung1.py` now partitions on
  `opt_mask.sum()` and reports a mask-aware `rand-legal%` floor = mean(1/n_choices),
  the offline twin of the `random_legal` benchmark agent. Existing per-context output
  is unchanged, so old numbers stay comparable.
- **What we learned:** **a baseline must have access to the same information as the
  model, or the gap it produces is meaningless.** The majority baseline was mask-blind
  while the model's head is `-inf`-masked; on 6.45% of rows that handed the model a
  free +2.4 pt. The general move — when a metric looks suspiciously high, partition the
  eval set by how much choice the model actually had — costs one accumulator and is
  worth running on any masked-action policy.
- **Not investigated:** `ctx 30` (n=224) scores **-0.9% below** its majority baseline.
  Unrelated to this card, but it is the one context where the model is worse than
  guessing the mode. Worth its own look.
