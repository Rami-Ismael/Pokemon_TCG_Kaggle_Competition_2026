# Energy option identity: does resolving `Option.energyIndex` help the BC policy?

**Design:** headroom bound (measurement sanity check) → add-one-in feature
ablation, gated on the first.

Chose *bound-first* over going straight to the ablation because the question is
not "which of n features matters" but "is there enough signal here for our
metric to see anything at all". The prior 7-group study
(`reports/feature_ablation/report.md`) measured a per-seed spread of 0.8–4.3 pp
around a 70 % top-1; a feature that can touch only a fraction of a percent of
rows is unfalsifiable at that noise floor, and running the arm anyway would
produce a number that looks like a result and isn't.

- **Hypothesis:** `_encode_options` never reads `Option.energyIndex` / `count`
  (verified by grep), so options that move *different energy types off the same
  Pokémon* encode into bitwise-identical rows and are separable only by
  `opt_pos_emb`. Adding an energy-type feature resolved against the target's
  public `energies` list should raise top-1 on the energy-selection contexts —
  ATTACH_FROM (21), ATTACH_TO (22), DETACH_FROM (23), DISCARD_ENERGY_CARD (26),
  SWITCH_ENERGY_CARD (28), DISCARD_ENERGY (30), SWITCH_ENERGY (33).
- **Independent variable:** presence of one new `OPT_FEATURE_SPECS` group,
  `energy_identity` (energy-type one-hot of the option's target energy, +
  `count`). Everything else — data, steps, seed policy, architecture — fixed.
- **Baseline:** `models/il_agent`'s recipe: `PTCGILConfig(opt_features=[])`,
  trained by `scripts/run_feature_ablation.py`'s baseline arm.
- **Metric & protocol:** `eval_rung1` top-1 on the held-out day (2026-07-27),
  reported both aggregate and **per-context** for the seven contexts above,
  each with an episode-cluster bootstrap 95 % CI. Seeds 42/43/44, equal
  `--total-steps 4000` (standing rule #4), Hub shards for both splits (local
  split folders are pruned stubs — see [[local-raw-splits-are-stubs]]).
- **Cost estimate:** Gate A ≈ 15 min CPU, no training. Gate B ≈ 2 arms × 3
  seeds × ~25 min ≈ 2.5 h MPS, chained so nothing overlaps.
- **Prior work checked:** `reports/feature_ablation/report.md` (0/7 groups
  accepted — strong prior that added features do not move BC top-1);
  `reports/ctx30_discard_energy_analysis.md` (ctx 30 is ~95 % forced moves +
  genuine ties); `notes/feature_ablation_candidates.md`.

## Pre-registered decision

**Gate A — headroom.** Measure, on held-out-day episodes with no model in the
loop, the fraction of decision rows where the recorded label sits in a
bitwise-identical option group whose members differ **semantically** (resolved
energy type, `count`, or `cardId`). Rows separable only by a raw `energyIndex`
or `serial` are counted separately and explicitly **excluded** — when two
options move the same energy type off the same Pokémon, which index the logger
recorded is arbitrary, and no feature can predict it.

- If the semantic ceiling is **< 0.3 pp** (the prior study's aggregate SE):
  the aggregate top-1 metric provably cannot resolve this feature. **Do not run
  Gate B as an aggregate-accuracy test**; report the bound as the result.
- If it is **≥ 0.3 pp**: run Gate B.

Either way, report the per-context ceilings, since a feature can be worth
having for a context that aggregate accuracy will never show.

**Gate B — ablation.** Accept the feature iff *both*:
1. mean paired top-1 improvement over the three seeds exceeds the across-seed
   spread (max − min) — the same bar the 7-group study used; and
2. per-context top-1 on the affected contexts improves with a bootstrap 95 % CI
   excluding zero.

Anything else is a drop. A positive on (2) with a null on (1) is written up as
"real but too rare to ship", not as an improvement.

## Checkpoint compatibility (resolved, not a risk)

The task brief flagged `il_model.py:212` as a width assertion that new option
features would invalidate. It is not one, and the architecture already handles
this:

- `_encode_options` always emits `extra_opt` at the full `N_EXTRA_OPT` width;
  `PTCGILConfig.opt_features` **column-selects** by group name via
  `feature_columns` (`il_model.py:109`). The `:212` raise fires only when a
  checkpoint asks for option features and no `extra_opt` tensor is passed at
  all — an encoder/model mismatch, not a width change.
- Verified: with the group **appended** to `OPT_FEATURE_SPECS`,
  `feature_columns(['attach_enable'])` still returns `[3, 4]`. Inserting it in
  the middle instead shifts that to `[17, 18]` and silently misaligns every
  existing checkpoint — so **append only**, as the comment above the dict
  already says.
- `models/il_agent/config.json` lists no `opt_features` at all, so its
  `extra_opt_proj is None` and it ignores the tensor entirely. No shipped
  checkpoint is affected.
- No encoded-tensor cache exists on disk (only the raw-episode HF download
  cache), so nothing stale needs invalidating.

## Result

**Gate A: FAILED. The feature is unfalsifiable with our metric and was not
implemented.** Full numbers, commands and per-context tables:
`reports/energy_option_identity/report.md`.

- **Observed** (500 held-out-day episodes, 93,621 decision rows):
  - Semantic ceiling **0.170 %** of all decisions, 95 % CI
    [0.127 %, 0.218 %] (episode-cluster bootstrap) — already below the prior
    ablation's SE of 0.3 pp, and far below its 0.8–4.3 pp seed spread.
  - Actual ceiling **0.045 %** (42 rows): the current `models/il_agent` already
    gets 117 of the 159 separable rows right.
  - A further **1.418 %** of rows are separable only by a raw `energyIndex` or
    `serial` — arbitrary labels no feature can predict. Counting those as
    headroom is what makes this idea look bigger than it is.
  - ATTACH_FROM (21) and ATTACH_TO (22) carry **zero** energy-identity signal
    (`energyIndex` is `None` on every option); DETACH_FROM (23) never occurs.
    The brief's nominated contexts were the wrong ones.
  - `TO_DECK_ENERGY` (32) is 63 % semantically-separable collisions and the
    model scores **55/55** on exactly those rows.
- **Decision:** dropped at Gate A; Gate B not run.
- **What we learned:** measure the ceiling before you measure the effect, and
  do it with no model in the loop so it is a property of the data. A feature
  can be *correctly identified as missing* — the grep was right, the bitwise
  collision is real — and still be worth nothing. The transferable move is to
  split "information the encoder drops" from "information the label depends
  on"; only the intersection is learnable, and here the intersection is 0.045 %.
  The `TO_DECK_ENERGY` 55/55 is the second lesson: a degenerate *encoding* is
  not a degenerate *prediction*, because position embeddings can absorb a
  logger's index conventions.
