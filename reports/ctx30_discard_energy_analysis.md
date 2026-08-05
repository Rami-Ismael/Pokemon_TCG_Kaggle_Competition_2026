# SelectContext 30 (`DISCARD_ENERGY`): why `eval_rung1` shows a negative gap

**Verdict: the gap is not real, and the context is ~95% unscoreable.**
Hypothesis (a) is correct, in a stronger form than stated — ctx 30 is not merely
"near-deterministic", it is dominated by *forced* moves and *genuine ties*.
Hypothesis (d) is ruled out. Hypothesis (b) is confirmed as a real encoder defect
but is far too small to move this metric. Hypothesis (c) is true but downstream of (a).

Model: `models/il_agent`. Data: `--data-source hub`, `--eval-split eval`,
`PTCG_DEVICE=cpu`. Reproduced the reported row exactly at 100 episodes
(20,888 rows scored; ctx 30 n=224, top1 86.6%, majority 87.5%, gap −0.9%).

## 1. What ctx 30 is

`SelectContext(30) == DISCARD_ENERGY` (`cg.api`, enum spans 0..48).
Every ctx-30 option carries `OptionType.ENERGY` (=6) and the field set
`{type, area, index, playerIndex, energyIndex, count}`. The decision is
"which attached Energy card gets discarded".

At 500 episodes: 1,105 ctx-30 rows across 366 distinct episodes (mean 3.0 rows
per episode — clustering is mild, not a single-episode artifact).
Target is the agent's own Pokémon in 363/424 multi-option rows, the opponent's in 61.

## 2. The gap is sampling noise

At n=224 the model loses by exactly **2 rows**. Under a paired test that is nothing:

| statistic | n=224 (100 eps) | n=1105 (500 eps) |
|---|---|---|
| model top-1 | 86.61% | 88.05% |
| majority baseline | 87.50% | 87.96% |
| gap | **−0.89%** | **+0.09%** |
| Clopper-Pearson 95% CI on top-1 | [81.4%, 90.8%] | [86.0%, 89.9%] |
| baseline inside that CI? | yes | yes |
| McNemar discordant pairs | 2 vs 0 | 3 vs 4 |
| exact McNemar p | 0.50 | 1.00 |
| episode-cluster bootstrap 95% CI on gap | [−2.28%, +0.00%] | [−0.35%, +0.58%] |

The sign flips when the sample grows 5×. **There is no evidence the model is
worse than the mode in ctx 30.** The one context flagged `<-- BELOW majority
baseline` is a false alarm produced by reading a point estimate off n=224.

## 3. Why the model *cannot* beat the mode here — the real finding

Decomposing the 1,105 rows by how much signal they carry:

| bucket | n | model correct | majority correct | delta |
|---|---:|---:|---:|---:|
| forced — exactly 1 legal option | 681 (61.6%) | 681 | 681 | +0 |
| multi-option, **all options encode identically** | 389 (35.2%) | 276 | 276 | +0 |
| partial collision | 28 | 12 | 11 | +1 |
| no collision | 7 | 4 | 4 | +0 |

Two independent things flatten this context:

**(i) 62% of rows are forced.** `minCount == maxCount == 1` on every single
ctx-30 row, and 681/1105 have exactly one legal option. Model and baseline are
both trivially 100% there. These rows are not a measurement of anything.

**(ii) The remaining choices are mostly ties.** Resolving each option's
`energyIndex` against the target Pokémon's attached `energies` list:

- **375 of 424 multi-option rows (88.4%) — every selectable option points at the
  same energy type on the same Pokémon.** Discarding attached energy #0 vs #1 is
  the same game state. There is no correct answer; the recorded label is an
  arbitrary index the engine happened to log.
- Only **49 of 424 (11.6%)** are decisions where the choice differs in the game.

Net: **~95% of ctx-30 rows carry no learnable signal at all**
(681 forced + 375 ties = 1,056 of 1,105).

On the 389 fully-degenerate rows the model always picks slot 0 — and slot 0 is
the modal label, so it hits **276/389 = 71.0%**, which is *exactly* the ceiling
for any policy that sees only `(context, n_options, slot index)`. The model is
already saturating the information available to it.

## 4. Hypothesis (b) — confirmed defect, wrong scale

The encoder genuinely drops the field that distinguishes these options.
`_resolve_option_refs` (il_dataset.py:533) maps `OptionType.ENERGY` to
`ref1 = (option.area, option.index, playerIndex)`, and `_encode_options`
(il_dataset.py:731) reads only `type / attackId / specialConditionType / number`
plus that ref. **`energyIndex`, `count`, `toolIndex`, `serial` and `cardId` are
never read anywhere in `il_dataset.py`** (verified by grep). For DISCARD_ENERGY
every option targets the same Pokémon, so all options collapse to one identical
feature row — confirmed bitwise across all seven option tensors.

The model still separates them only because `opt_pos_emb` (il_model.py:122,
"position IS the action index to output") injects the slot index; logit spread
within an identical-feature group reaches 5.25.

But this defect is worth at most the 49 rows (4.4%) where energy type actually
differs; the model gets 23/49 there. Fixing it cannot produce the −0.9%, which
lives entirely in 7 discordant rows. It is a correctness issue in the encoder,
not the explanation for this number.

## 5. Hypotheses ruled out

- **(d) multi-select unroll label noise — ruled out.** All 1,105 ctx-30 rows have
  `minCount == maxCount == 1` and `exclude == frozenset()`. The `maxCount > 1`
  unroll branch (il_dataset.py:1005) never fires in this context, and no ctx-30
  row carries a DECLINE label.
- **(c) undertrained — true but not causal.** Ctx 30 is 0.83% of training rows
  (134 of 16,234 over 100 train episodes), rank 17 of 30 contexts present; 65.7%
  of those are forced, matching the eval day. So the *effective* training signal
  is ~0.03% of rows. That scarcity is a consequence of (a) — the context is
  mostly forced moves and ties — not an independent cause.

## 6. Recommendations

1. **Don't chase this number.** Ctx 30 is not a model weakness. If `eval_rung1`
   is to flag contexts, it should suppress rows where `n_real_options == 1`
   (61.6% of ctx 30, 6.71% of the corpus per the existing forced-row measurement)
   and attach a CI, or the flag will keep firing on noise.
2. **Separately, fix the encoder.** Add `energyIndex`-resolved energy type (and
   `count`) to the option features. Justify it on the ATTACH/DETACH/`SWITCH_ENERGY`
   contexts where it plausibly matters more — not on ctx 30. Needs its own
   before/after measurement; do not assume it helps (cf. the feature-ablation
   negative result).
3. The in-context "majority" in `eval_rung1` is computed **on the eval rows
   themselves**, so it is an oracle baseline and slightly optimistic. Worth noting
   wherever the gap column is quoted.

## Reproduce

```bash
PTCG_DEVICE=cpu uv run python scripts/diagnostics/ctx30_collide.py --max-episodes 500 --out reports/ctx30_collide.json
PTCG_DEVICE=cpu uv run python scripts/diagnostics/ctx30_energytype.py --max-episodes 500 --out reports/ctx30_etype.json
PTCG_DEVICE=cpu uv run python scripts/diagnostics/ctx_freq.py train 100
```
