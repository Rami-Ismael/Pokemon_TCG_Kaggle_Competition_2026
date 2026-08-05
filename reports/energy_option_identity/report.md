# Energy option identity: the encoder really does drop it, and it is worth ~0.17 pp

**Date:** 2026-08-04 · **Branch:** `claude/relaxed-ardinghelli-ad93f1`
**Verdict: the feature is not implemented. The experiment was stopped at its
pre-registered headroom gate, before any training run.**
Experiment card: `notes/experiments/2026-08-04-energy-option-identity.md`.

## The defect is real

Confirmed exactly as reported. `_encode_options` (`il_dataset.py:704`) reads
only `type`, `attackId`, `specialConditionType`, `number`, and the
`_resolve_option_refs` card reference. The `cg.api.Option` fields `energyIndex`,
`count`, `toolIndex`, `serial`, and `cardId` are never read. So two options that
move *different energy types off the same Pokémon* produce bitwise-identical
feature rows, separable only by `opt_pos_emb` (`il_model.py:122`).

What was not known is how much of the corpus that costs. This report measures
it, with no model in the loop for the main number, so it is a property of the
encoder and the corpus rather than of any checkpoint.

## Method

`scripts/diagnostics/option_field_census.py` streams held-out-day decisions
(2026-07-27, Hub shards — the local split folders are pruned stubs), encodes
each, groups the real option slots by **bitwise-identical feature signature**
across all seven per-option tensors, and then asks, for each colliding group,
which dropped `Option` field would separate its members.

The critical distinction the earlier ctx-30 pass did not draw:

- **Semantic separation** — the group's members differ in *resolved energy type*
  (`energyIndex` looked up against the target Pokémon's public `energies` list),
  `count`, or `cardId`. The choice is a real game decision; a feature could
  learn it.
- **Arbitrary separation** — the members differ only in a raw `energyIndex` or
  `serial` while pointing at the *same* energy type or the *same* card. The
  recorded label is one of several interchangeable indices, so no feature can
  predict which one the logger happened to write down. Counting these as
  headroom is the mistake that makes a dropped field look valuable.

Only the semantic set is headroom. The CI resamples whole **episodes**, not
rows — rows within an episode share a board and two players and are correlated.

```bash
uv run python scripts/diagnostics/option_field_census.py --max-episodes 500 --out reports/energy_option_identity/census_eval_500.json
```

## Result: 500 held-out-day episodes, 93,621 decision rows

**Semantic ceiling: 159 / 93,621 = 0.170 % of all decisions,
95 % CI [0.127 %, 0.218 %]** (episode-cluster bootstrap, 500 episodes).

That is the *optimistic* bound — it assumes the new feature separates every one
of those rows correctly and that the current model gets all of them wrong.

For scale: the 7-group feature ablation (`reports/feature_ablation/report.md`)
measured a per-seed spread of 0.8–4.3 pp around a 70 % top-1, with SE ≈ 0.3 pp.
**The entire achievable effect is below that study's standard error**, so an
aggregate top-1 comparison could not distinguish this feature from noise no
matter how many seeds were run. Gate A of the pre-registration fails.

A separate 1.418 % of rows (1,328) are **arbitrary** — the label sits in a group
separated only by a raw index or serial. These are unlearnable by construction,
and 862 of them are a single context (`SKILL_ORDER`, two copies of the same
card differing only by `serial`).

### The actual ceiling is 4× smaller again: 0.045 %

Re-running the census with `--model-dir models/il_agent` asks how many of those
159 rows the current policy **already gets right** despite the degenerate
encoding. It gets 117 of 159 (73.6 %). Only 42 rows are both semantically
separable *and* currently wrong:

**42 / 93,621 = 0.045 % of all decisions.** The optimistic bound's upper CI
limit (0.218 %) is already below the ablation's standard error, and the actual
figure is a fifth of that again.

| ctx | name | semantic rows | model already right | could improve | sem. % of ctx |
|---|---|---:|---:|---:|---:|
| 32 | TO_DECK_ENERGY | 55 | **55** | **0** | 63.2 % |
| 30 | DISCARD_ENERGY | 44 | 22 | 22 | 4.0 % |
| 26 | DISCARD_ENERGY_CARD | 34 | 26 | 8 | 20.7 % |
| 28 | SWITCH_ENERGY_CARD | 18 | 8 | 10 | 23.7 % |
| 33 | SWITCH_ENERGY | 8 | 6 | 2 | 11.8 % |

The `TO_DECK_ENERGY` row is the sharpest result in this report. It is the one
context where the dropped field dominates — 63 % of its rows are
semantically-separable collisions — and the model scores **55/55** on exactly
those rows. Position embedding plus the logger's index conventions already
recover the whole signal there. A degenerate encoding is not automatically a
degenerate prediction.

### Per context

| ctx | name | rows | % of all | multi-opt | ≥1 collision | fully degenerate | **semantic** | sem. % of ctx | arbitrary |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | ATTACH_FROM | 3,531 | 3.77 % | 3,150 | 832 | 25 | **0** | 0.0 % | 0 |
| 22 | ATTACH_TO | 4,256 | 4.55 % | 3,805 | 3,759 | 3,611 | **0** | 0.0 % | 0 |
| 23 | DETACH_FROM | 0 | — | — | — | — | — | — | — |
| 26 | DISCARD_ENERGY_CARD | 164 | 0.18 % | 138 | 123 | 27 | **34** | 20.7 % | 42 |
| 28 | SWITCH_ENERGY_CARD | 76 | 0.08 % | 73 | 68 | 0 | **18** | 23.7 % | 18 |
| 30 | DISCARD_ENERGY | 1,105 | 1.18 % | 424 | 417 | 389 | **44** | 4.0 % | 354 |
| 32 | TO_DECK_ENERGY | 87 | 0.09 % | 58 | 58 | 58 | **55** | 63.2 % | 3 |
| 33 | SWITCH_ENERGY | 68 | 0.07 % | 57 | 57 | 57 | **8** | 11.8 % | 49 |
| 34 | SKILL_ORDER | 862 | 0.92 % | 862 | 862 | 862 | **0** | 0.0 % | 862 |

Three findings that contradict the brief's framing:

1. **ATTACH_FROM (21), ATTACH_TO (22) and DETACH_FROM (23) — the contexts the
   brief nominated as where energy identity "plausibly matters more" — carry no
   energy-identity signal at all.** Counted over all 500 episodes, the options
   carrying a non-`None` `energyIndex` fall in exactly five contexts, all of
   them at 100 % coverage: 26 (551/551), 28 (340/340), 30 (1,850/1,850),
   32 (174/174), 33 (146/146). Contexts 21 and 22 have **zero** — they are
   `OptionType.CARD` options naming a card or a board slot, not an attached
   energy — and 23 never occurs. ATTACH_TO's very high collision rate (3,611
   fully degenerate rows) is real but is *duplicate identical cards* — see
   below.
2. **The energy-identity contexts are the rare ones**: 26, 28, 30, 32, 33 sum
   to 1,500 rows, 1.60 % of decisions.
3. **`TO_DECK_ENERGY` (32) is the one context where the defect genuinely
   dominates** — 63 % of its rows are semantically-separable collisions. It is
   also 0.09 % of decisions, so it contributes ≈ 0.06 pp.

### The large collisions are not this bug

`scripts/diagnostics/ref_resolution_probe.py` was written to test whether the
huge collision counts in MAIN, TO_HAND and ATTACH_TO were caused by card
references failing to resolve (empty `select.deck`, out-of-range indices). They
are not:

```
colliding option slots by AreaType of ref1:
  DECK      resolved 8466   unresolved    0
  HAND      resolved 14308  unresolved    0
  BENCH     resolved 2126   unresolved    0
  PRIZE     resolved 0      unresolved 2101   (face-down; correctly opaque)
rows whose collision would be separated by correct card resolution: 0 / 13640 = 0.00 %
```

Every colliding group outside PRIZE resolves to the *same card id* — they are
duplicate copies of one card, i.e. genuine ties. PRIZE is 100 % unresolved
because prizes are face-down, which is the privacy rule working correctly, not
a bug. **The encoder's card identity is sound**; the only real information loss
is the energy one, and it is small.

## Decision

**Dropped at Gate A. The `energy_identity` feature group was not implemented and
no training arm was run.**

Running the 2-arm × 3-seed ablation would have cost ~2.5 h of contended MPS to
chase an effect of at most 0.045 pp against a measurement whose standard error
is 0.3 pp and whose seed-to-seed spread reached 4.3 pp — a
guaranteed-inconclusive result that would nonetheless read like evidence.

If it is wanted anyway, the implementation is small and safe (see below), and
the only defensible protocol is a **per-context** test on contexts 26/28/30/33
over the full eval day (~4,430 episodes → ~12,000 such rows), scored on the
separable rows only. Note that 32 should be dropped from that list on the
evidence above — the model is already perfect there — which leaves ~370
improvable rows day-wide as the entire target.

## Checkpoint compatibility: not a risk

The brief flagged `il_model.py:212` as a feature-count assertion that a new
option feature would invalidate. It is not one.

- `_encode_options` always emits `extra_opt` at the full `N_EXTRA_OPT` width.
  `PTCGILConfig.opt_features` **column-selects** by group name via
  `feature_columns` (`il_model.py:109`). The `:212` raise fires only when a
  checkpoint asks for option features and no `extra_opt` tensor is passed at all.
- Verified: appending a group to `OPT_FEATURE_SPECS` leaves
  `feature_columns(['attach_enable'])` at `[3, 4]`. Inserting it in the middle
  moves it to `[17, 18]` and would silently misalign every existing checkpoint.
  **Append only** — as the comment above the dict already instructs.
- `models/il_agent/config.json` lists no `opt_features`, so its
  `extra_opt_proj is None` and it ignores `extra_opt` entirely.
- No encoded-tensor cache exists on disk (only the raw-episode HF download
  cache), so nothing stale needs invalidating.

## Reproduce

```bash
uv run python scripts/diagnostics/option_field_census.py --max-episodes 500 --out reports/energy_option_identity/census_eval_500.json
```

```bash
uv run python scripts/diagnostics/option_field_census.py --max-episodes 500 --model-dir models/il_agent --out reports/energy_option_identity/census_eval_500_model.json
```

```bash
uv run python scripts/diagnostics/ref_resolution_probe.py --max-episodes 60 --out reports/energy_option_identity/ref_probe_60.json
```

Artifacts: `census_eval_500.{json,txt}`, `census_eval_500_model.{json,txt}`,
`ref_probe_60.json`. All runs are CPU-only and deterministic
(`config.RANDOM_SEED` seeds the bootstrap); `--max-episodes` is the only knob
that changes the numbers.
