# Self-play vs its BC init, measured against the FIELD (controlled comparison)

Rami's "option 4". Runnable today because the pool *can* measure Mega Lucario ex
arms — 18 pilots, several strong (`pixiux_lucario_v63` 1735.5,
`makthanithin_improved_prob` 1705.1, `romanrozen_strong_start` 1687.4,
`kojimar_lucario` 1664.6). That is exactly what it cannot do for Grimmsnarl
(see `2026-08-05-mcts-rl-deck-confound.md`, Stage 0 FAIL).

- **Hypothesis:** the self-play-over-BC gain is an in-distribution artifact and
  will not survive being measured against the field. Mechanism (verified in
  code, not assumed): every league was **100% our own checkpoints**
  — g1 `models/il_agent,models/s2v2/e0_s43`; g2/g3 those plus
  `il_agent_full_0804` and `selfplay_g1/refs/u430080` — all piloting the **one
  hardcoded deck** (`selfplay.py:39`). A policy optimized against copies of
  itself on one deck has no pressure toward field robustness, so a head-to-head
  win over its own init measures the training distribution.
- **Design:** controlled comparison. Not an ablation — the question is whether a
  known effect survives a change of measurement context, not which component
  causes it.
- **Independent variable:** training method (BC vs BC+self-play), holding deck,
  checkpoint family, and opponent set fixed.
- **Baseline:** `il_alldays_0804` — the actual initialization of g1/g2/g3, so the
  comparison is against what self-play started from, not an unrelated model.

## Arms & protocol

Arms (6): `il_agent`, `il_alldays_0804` (BC init), `selfplay_g1_ref430k`,
`selfplay_g1_final`, `selfplay_g2_final`, `selfplay_g3_final`.

Field (9, identical for every arm): `pixiux_lucario_v63`, `kojimar_lucario`,
`makthanithin_improved_prob`, `romanrozen_strong_start`, `tb_archaludon`,
`wmh_alakazam`, `mechi22_alakazam`, `kiyotah_dragapult`, `random_legal` (floor).

Mirrored pairs, `--games 15` → 30 games/ordered pair, **270 games/arm vs the
field** (σ ≈ 3.0pp, 95% CI ≈ ±5.9pp). `--no-glicko-persist`, out to
`reports/option4_selfplay_vs_bc_field.json`.

**Checkpoint integrity verified before launch** (this run was nearly invalid):
`models/selfplay_g{1,2,3}` and `models/il_agent_full_0804` were **absent from
main** — present only inside one sibling worktree's local `models/`. Every
self-play arm was failing to load and would have been measured as a silent
`_safe_choice` fallback. Restored into main (~52 MB; `il_agent_full_0804` is
byte-identical to `il_alldays_0804`, md5 `ebb501c6…`, so it is a symlink).
All 6 arms re-verified `_load_model() is not None`.

## Pre-registered decision (committed before any result)

- **Self-play gain is real and general** if the best self-play arm's field win%
  exceeds `il_alldays_0804`'s by **≥5 pp with non-overlapping 95% CIs**.
- **Self-play gain is in-distribution only** if the self-play arms land within
  ±5 pp of, or below, `il_alldays_0804` on the field — despite the recorded
  head-to-head g2-over-BC of **59.0% [54.2, 63.8]** (n=400).
- **Positive control:** the g2-vs-`il_alldays_0804` head-to-head cell in this
  same run must reproduce ≈59%. If it does not, the harness changed and the
  field numbers are uninterpretable — diagnose before reading anything else.
- **Floor check:** any arm not decisively beating `random_legal` is a broken
  build, not a result.

## Result (2026-08-05, 1878 s wall clock, `reports/option4_selfplay_vs_bc_field.json`)

**Fallback check first:** 0.00–0.01% across every arm (23k–31k decisions each).
Every model genuinely ran; nothing is a `_safe_choice` artifact.

### vs the field (9 opponents, identical for every arm, 270 games/arm)

| Arm | Field win% | Δ vs BC init |
|---|---:|---:|
| `selfplay_g3_final` | **63.0% ± 5.8** | **+17.0** |
| `selfplay_g2_final` | 59.3% ± 5.9 | +13.3 |
| `selfplay_g1_final` | 58.9% ± 5.9 | +13.0 |
| `selfplay_g1_ref430k` | 56.3% ± 5.9 | +10.4 |
| `il_alldays_0804` (BC init) | 45.9% ± 5.9 | — |
| `il_agent` | 12.6% ± 4.0 | −33.3 |

**The pre-registered criterion PASSES:** g3 at [57.2, 68.8] vs BC init at
[40.0, 51.8] — +17.0 pp, non-overlapping. All four self-play arms clear the
+5 pp bar. Floor check clean (96.7–100% vs `random_legal`; `il_agent` only 76.7%).

### …and the ladder says the opposite

Three arms here have ladder scores we read ourselves:

| Agent | Ladder | Field win% | ladder rank | local rank |
|---|---:|---:|:-:|:-:|
| `il_alldays_0804` | 418.0 | 45.9% | 1 | 2 |
| `il_agent` | 397.3 | 12.6% | 2 | 3 |
| `selfplay_g1_ref430k` | **267.4** | **56.3%** | **3** | **1** |

**Spearman rho = −0.500** on the verified anchors. The pool ranks
`selfplay_g1_ref430k` *first* and Kaggle ranked it *last*, 150 points below the
BC init. This is a direct inversion on **Mega Lucario ex** — the deck I had
argued the pool *could* measure, because it has 18 pilots with several strong
ones. Pilot count did not buy predictiveness.

Second discrepancy: `il_agent` (397.3) and `il_alldays_0804` (418.0) are ~21
ladder points apart but **33 pp** apart locally. The pool wildly over-separates
two builds Kaggle rated nearly the same.

### Positive control: FAILED

Pre-registered: the g2-vs-BC head-to-head must reproduce ≈59% (the recorded
`reports/g2_vs_alldays.json` figure, n=400). Observed **80.0% [65.7, 94.3]**
at n=30 — the 59.0% [54.2, 63.8] target sits below this interval, so the two do
not overlap. The direction agrees (self-play ahead) and n=30 is thin, but by the
letter of the pre-registration this control did not reproduce.

## LADDER VERDICT (ref 55284059, submitted 2026-08-06 00:29)

`selfplay_g3_final` — the winning arm above — was submitted to settle this,
because no local tournament can adjudicate a question about its own validity.
Bundle: same Mega Lucario ex deck as the 418.0 imitation submission (no deck
confound), checkpoint md5-verified, CPU rehearsal clean.

**Score readings, in order: 564.9 → 564.9 → 405.4.** Pre-registered prediction
was 250–420, with ">418 = the local field test transfers."

| Reading | Score | Would have concluded |
|---|---:|---|
| 1–2 (first ~30 min) | **564.9** | transfer confirmed, self-play wins by +147 |
| 3 (~2 h) | **405.4** | **no effect** — 12.6 pts *below* its own init |

**Verdict: no demonstrated improvement.** 405.4 vs the init's 418.0 is Δ −12.6,
deep inside the ±100 same-build band. Self-play neither beat nor lost to
imitation training on the ladder; it tied. The +17.0 pp local field margin did
**not** translate into ladder points.

This is the fourth time an early ladder reading has misled here (the MCTS
submission read 600.0 and settled at 291.4). The 564.9 reading held across two
consecutive polls over half an hour and was still wrong by 160 points. **Two
agreeing early readings are not convergence.**

### Pool predictiveness, recomputed at each stage

| Anchor set | rho |
|---|---:|
| n=3, before this submission | **−0.500** |
| n=4, using the 564.9 reading | +0.400 |
| n=4, using the settled 405.4 reading | **+0.000** |

Zero correlation. The pool's ordering of these four arms carries no information
about their ladder ordering — it is neither inverted nor predictive, just
unrelated. Note how far the coefficient swung on one unsettled number: any rho
computed against a <2 h-old submission is itself provisional.

## Decision: **NOT adopted** — the local margin did not transfer

The measurement passed its own bar (+17.0 pp, non-overlapping CIs) and the
ladder then scored the winning arm level with the baseline it supposedly beat.
The local field margin is real as a local fact and worth zero ladder points.

## What we learned

1. **The pool's Lucario measurements are not trustworthy either.** The
   working assumption — Grimmsnarl is unmeasurable, Lucario is fine — is now
   falsified on its own terms: rho −0.500 on Lucario-piloting arms with verified
   anchors. The pool-repair job is bigger than adding Grimmsnarl opponents.
2. **Opponent-pilot count is not the mechanism.** 18 Lucario pilots did not make
   Lucario measurable. Whatever makes local results transfer, headcount of
   same-archetype opponents is not sufficient for it.
3. **Self-play, as currently configured, does not beat imitation training on
   the ladder.** Answered — not open. `selfplay_g3_final` 405.4 vs its own init
   418.0. The most likely cause is the training setup, not the method:
   the league was 100% our own checkpoints on one hardcoded deck
   (`selfplay.py:39`), so there was no pressure toward field robustness. That
   is the variable to change before the next self-play arm, not the algorithm.
4. **Two agreeing early ladder readings are not convergence.** 564.9 held across
   two polls over 30 minutes and was 160 points wrong. Only elapsed hours count.
5. Transferable lesson: when an experiment's instrument is under suspicion, a
   passing result is not evidence — pre-register the *external* check alongside
   the internal one. Here the pre-registration is what stopped a +17 pp local
   win from being written up as a method that works.
