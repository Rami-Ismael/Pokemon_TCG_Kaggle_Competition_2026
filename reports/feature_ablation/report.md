# Deterministic-future features for il_agent: a negative result

**Date:** 2026-08-04 · **Branch:** `claude/il-agent-feature-ablation-543b5c`
**Verdict: no feature group ships. il_agent is unchanged.**

## What was tried

Port of the Orbit Wars 2nd-place idea ("feed the simulator-rolled
do-nothing future so the net doesn't learn game arithmetic") to our
behavior-cloning agent, as hand-computed deterministic futures from the
visible observation: KO-race turn counts, prize-race arithmetic, energy
timelines, per-option "this ATTACK KOs now / wins the game" flags, plus
board status conditions. 17 candidates documented with per-feature
inference-cost and no-private-information arguments in
`notes/feature_ablation_candidates.md`; 7 groups implemented in
`src/pokemon_tcg/il_dataset.py` (encoder always emits them; a checkpoint's
`PTCGILConfig.global_features/opt_features` selects columns, so legacy
checkpoints are untouched). Weakness ×2 and base-damage 1:1 verified
empirically from visible logs; resistance −30 applied but unconfirmed
(sparse sample).

## Protocol

One group per arm; each arm = baseline + that group only. Seeds 42/43/44;
identical data per seed; equal `--total-steps 4000` (standing rule #4);
metric = eval_rung1 top-1/top-3 on the held-out day (150 episodes,
~27k rows, SE ≈ 0.3 pp) vs the majority-class baseline on the same rows;
accept iff mean paired top-1 improvement > across-seed spread (max−min).
Corpus: Hub shards, train day 2026-07-26 (4,554 episodes), eval day
2026-07-27.

**First attempt invalidated (recorded deliberately):** `--data-source auto`
silently trained on a 24-episode pruned stub of the local train folder and
evaluated on 12 episodes (ADR-001 prunes raw days after Hub-verify; the
folder existed but was 99.5 % empty). Symptoms: 98.8 % train accuracy
(memorization), ±1.1 pp eval noise, sign flips between eval samples.
Artifacts archived in `reports/feature_ablation_INVALID_local_stub/`.
Fix: train_il.py and eval_rung1.py now refuse local splits holding <90 %
of their registered episode count, and eval_rung1 gained a deterministic
`--data-source hub` path.

## Result (valid rerun)

| arm | top1 by seed (%) | mean top1 | paired d vs baseline (pp) | mean d | spread(d) | verdict |
|---|---|---|---|---|---|---|
| baseline | 69.5, 69.6, 70.9 | 70.00 | — | — | — | reference |
| ko_race | 68.0, 68.7, 69.1 | 68.60 | −1.50, −0.90, −1.80 | −1.40 | 0.90 | drop |
| prize_race | 70.9, 69.9, 70.8 | 70.53 | +1.40, +0.30, −0.10 | +0.53 | 1.50 | drop |
| energy_deficit | 67.9, 69.4, 66.4 | 67.90 | −1.60, −0.20, −4.50 | −2.10 | 4.30 | drop |
| status_conditions | 68.0, 69.1, 69.3 | 68.80 | −1.50, −0.50, −1.60 | −1.20 | 1.10 | drop |
| attack_tactical | 68.6, 69.5, 70.2 | 69.43 | −0.90, −0.10, −0.70 | −0.57 | 0.80 | drop |
| attach_enable | 68.7, 68.6, 69.3 | 68.87 | −0.80, −1.00, −1.60 | −1.13 | 0.80 | drop |
| retreat_switch | 68.1, 68.8, 68.0 | 68.30 | −1.40, −0.80, −2.90 | −1.70 | 2.10 | drop |

Majority-class baseline on the same rows: 39.0 %. Top-3 was flat across
all arms (92.1–92.8 %). `ko_race`, `status_conditions`, `attach_enable`,
and `retreat_switch` are *replicated negatives* (all seeds below paired
baseline). `prize_race` is the only positive lean (+0.53) and still fails
the spread bar. No combined arm exists (nothing accepted), so the Step-4
benchmark is vacuous: **there is no candidate model to benchmark, and no
"better" claim is being made about anything** (nothing here touches the
ladder; leaderboard-check not applicable).

## Why hand-computed futures don't help this model (diagnosis)

1. **BC accuracy can't reward knowledge the demonstrators don't act on.**
   The label is "what the logged ladder agent did," and much of the corpus
   is simple rule bots and mid players. "This attack KOs now" only lifts
   action-match where demonstrators reliably take the KO — and there the
   model has already learned it from ~800k rows. Features encoding *good*
   play cannot improve a metric that scores *typical* play.
2. **The information is nearly redundant with the embeddings.** Orbit Wars
   fed futures of continuous physics — a hard function a small net can't
   do internally. Here the arithmetic is lookups over a discrete card
   vocabulary the model already embeds; "Dragapult usually KOs this" is
   learnable as an association without arithmetic.
3. **Why slightly negative rather than flat:** at equal steps an extra
   input head is a transient optimization tax (random projections inject
   noise until downweighted, and the cosine schedule ends first), and the
   features are approximations (variable-damage attacks read 0; ability /
   effect modifiers are pervasive), so on some decks the feature is
   confidently wrong — a misleading input is worse than none.

Scope caveats, stated honestly: measured at a 4,000-step budget (deployed
model trains 38.5k steps) — a full-budget interaction was not tested
because no arm earned it; and accuracy-gating means a feature that helps
*win rate* but not action-match would be invisible here. Those are the
protocol's terms, applied as specified.

## What this buys the project

- The encoder/model plumbing (self-describing feature configs, extra
  tensors ignored by legacy checkpoints) is merged-safe and reusable —
  any future feature idea is now a `--features` flag, not a fork.
- Two data-integrity guards that already caught one silent failure mode
  (pruned-stub local splits), plus a hub-native eval_rung1.
- The negative result itself: do not spend more time on input feature
  engineering for the BC policy. The lever it isn't: features. The levers
  it plausibly is: demonstration quality (skill filtering — see
  `reports/skill_filter/`, the follow-up experiment), corpus size, and
  win-rate-driven training (PPO/MCTS lines).
