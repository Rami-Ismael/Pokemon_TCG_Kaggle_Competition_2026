# Which IL checkpoint, piloting which deck?

Follows the `deck-selection` skill's ①→②→③→④ ordering, extended with a second
axis the earlier study held fixed. [`reports/deck_selection.md`](deck_selection.md)
answered *"which deck, with `il_agent` fixed"*. This answers *"which **checkpoint**,
and does the deck answer survive changing it"*.

Full cell-level numbers: [`reports/il_model_deck_selection.xlsx`](il_model_deck_selection.xlsx).

> **Ladder status at the time of this run (2026-08-04):** rank **4784 / 6272**, team
> score **486.0**. Best-ever is reported as **804.0** (sub 55162376), but that is a
> *single early read* — four same-build resubmits of that agent settled at 699.0 /
> 692.7 / 666.1 / 683.2, so **~685** is what the build actually polls at. Treat 804.0
> as the top of the noise band, not the agent's strength.
>
> Everything below is a **local** measurement. Per the local-vs-ladder inversions on
> record, no result here is a ladder claim until something is submitted and read back.
> Note also that most third-party ladder figures used here are **self-reported by their
> authors and unverified by us** — see the anchor-provenance table in ⑤.

## ⓪ Two silent failures found before any games were played

Both would have produced clean-looking, wrong numbers. Neither raises.

**1. The deck override never reached wrapped checkpoints.** `load_agent()` in
`scripts/benchmark_agents.py` implemented `<agent>@<deck-tag>` by assigning
`mod.my_deck` on the module it had just exec'd. But `agent()` returns the
`my_deck` it sees in *its own* globals, and for every wrapper arm
(`agents/grid_cells/`, `agents/s2_arms/`, `agents/ppo_arms/`, and the new
`agents/il_arms/`) that owner is the **inner `il_agent` core module the wrapper
imported**, not the wrapper. So the override reported success and changed
nothing: every wrapped arm kept piloting whatever deck its wrapper had already
injected — Mega Lucario ex, the 1-episode control deck. Fixed by writing through
to `fn.__globals__`, keeping the `mod` write for plain modules where the two are
the same dict.

**2. `models/il_agent_medium_combined` is an empty directory.** It contains no
`config.json` and no `model.safetensors`. `il_agent._load_model()` catches its own
exceptions and returns `None`, so the pre-existing `grid_medium_comb` arm plays
non-ML `_safe_choice` moves while looking like a model. Any past number attributed
to that arm is a measurement of the fallback heuristic, not of a checkpoint. It is
deliberately **not** re-wired here.

Both are guarded now by [`scripts/verify_il_arms.py`](../scripts/verify_il_arms.py),
which the sweep is gated on. For every (arm, deck) cell it asserts the core
resolved the intended `MODEL_DIR`, `_load_model()` returned a real module, the
weights hash to the expected safetensors, and — end to end — that `agent({})`,
the real deck-submission call, returns exactly the 60 requested cards.
**35/35 cells verified**; the empty checkpoint correctly FAILs.

## ① Enumerate — the model axis

Deduped by sha256, because the `models/` directory double-counts:

- `models/il_agent_3ep` is **byte-identical** to `models/il_agent`
- `models/il_agent_winning_827.8` is **byte-identical** to `models/il_agent_2ep_backup`

So nine directories are **seven distinct checkpoints**.

| arm | checkpoint | params | train data | epochs | offline acc | ECE |
|---|---|---:|---|---:|---:|---:|
| `il_bc_2ep` | `il_agent_2ep_backup` | 3.32M | train 2026-07-26 | 2 | 0.7478 | — |
| `il_bc_3ep` | `il_agent` | 3.32M | train 2026-07-26 | 3 | 0.7534 | — |
| `il_bc_4ep` | `il_agent_4ep` | 3.32M | train 2026-07-26 | 4 | 0.7592 | — |
| `il_medium_3ep` | `il_agent_medium` | 10.99M | train 2026-07-26 | 3 | 0.7545 | 0.0231 |
| `il_small_comb_2ep` | `il_agent_small_combined` | 3.32M | combined 07-01+07-26 | 2 | 0.6366 | 0.1630 |
| `il_hfstream_comb_3ep` | `il_agent_hfstream_combined_3ep` | 3.32M | combined (HF stream) | 3 | 0.7527 | 0.0231 |
| `il_alldays_3ep` | `il_alldays_0804` | 3.32M | all days (127,748 steps) | 3 | 0.7583 | 0.0193 |

Majority-class baseline on the same eval rows: **0.381**.

**The identity worth noticing:** the checkpoint behind our best-ever ladder score
(the build saved as `il_agent_winning_827.8`) is the **2-epoch** model — the one
offline accuracy ranks *last* of the 07-26 family. The checkpoint `il_agent` ships
today is the 3-epoch one.

## ② Familiarity audit — and where it is missing

Corpus episode counts per deck are **carried forward** from
[`reports/deck_selection.md`](deck_selection.md) (train split 2026-07-26, 4554
episodes, deck identity = ace/carry Pokémon). They were **not** re-derived: the
local episode splits are now stubs — `train-2026-07-26` holds **24 real files of
4554**, and `train-combined-0701-0726` is **9,796 dangling symlinks of 9,820**.
Re-deriving would mean re-streaming the corpus from the HF dataset.

| deck | episodes (07-26) | clears 150 floor |
|---|---:|---|
| Marnie's Grimmsnarl ex | 3488 | yes |
| Alakazam | 1542 | yes |
| Team Rocket's Spidops | 784 | yes |
| Cynthia's Garchomp ex | 625 | yes |
| **Mega Lucario ex** (control) | **1** | **no — unmeasured** |

⚠️ **The gap this leaves.** Three arms (`il_alldays_3ep`, `il_hfstream_comb_3ep`,
`il_small_comb_2ep`) trained on *more days than 07-26*, so they saw strictly more
episodes of these decks than the column above states. Their true per-deck
familiarity is **UNMEASURED**, not equal to the numbers shown. This matters for
exactly one interpretation — "did the extra-data arm win because it is better, or
because it had more episodes of this particular deck?" — and this run cannot
separate those. Stated rather than papered over.

## ③ Measure — setup

- **Policy/deck cross:** `<arm>@<deck-tag>`, one identical wrapper per arm
  (`agents/il_arms/`), so nothing but the weights varies across the model axis.
- **Opponent pool (8):** `tb_archaludon` (ladder 1196.1), `makthanithin_1084_baseline`
  (1084.5), `romanrozen_strong_start` (~950), `tb_dragapult` (880.9), `wmh_alakazam`
  (860.3), `wmh_garchomp` (713.8), `tb_heuristic` (633.0), `random_legal` (floor).
  Chosen over the earlier study's pool because 7 of 8 carry real ladder anchors
  spanning 532→1196, which makes pool predictiveness measurable rather than assumed.
- Mirrored pairs (harness default), `--games 10` = **20 games per arm-vs-opponent
  cell, 160 field games per arm**.
- `PTCG_DEVICE=cpu` — what the Kaggle evaluator uses, so this is the honest device.
- **Isolated:** `--no-glicko-persist`; the standing `reports/glicko_ratings.json`
  was never touched. These are synthetic identities that have never played a ladder
  game, so Glicko is recomputed by
  [`scripts/merge_il_sweep.py`](../scripts/merge_il_sweep.py) over the union of all
  runs from a flat 1500 prior. Arms never play each other; they are placed on one
  scale through the shared pool.

⚠️ **Pool caveat, recorded before reading the results.** Every arm beats
`tb_archaludon` (ladder 1196) more comfortably than `romanrozen_strong_start`
(ladder ~950). Romanrozen runs live UCB1/MCTS rollouts through the cg engine and
receives **unbudgeted think time locally**, which the ladder's time budget would
constrain. This pool therefore probably **overrates search agents** relative to the
real ladder, and a win rate against it should not be read as a ladder forecast.

## ③ Results

### Integrity check first — is this the model playing?

`scripts/fallback_diagnostic.py` on the six decisive cells, `PTCG_FALLBACK_TRACK=1`:

| cell | decisions | fallbacks | rate |
|---|---:|---:|---:|
| `il_alldays_3ep@marnies_grimmsnarl_ex` | 919 | 0 | 0.00% |
| `il_bc_3ep@marnies_grimmsnarl_ex` | 845 | 0 | 0.00% |
| `il_small_comb_2ep@marnies_grimmsnarl_ex` | 926 | 0 | 0.00% |
| `il_alldays_3ep@mega_lucario_ex` | 732 | 0 | 0.00% |
| `il_bc_3ep@mega_lucario_ex` | 584 | 0 | 0.00% |
| `il_small_comb_2ep@mega_lucario_ex` | 688 | 0 | 0.00% |

**0 fallbacks in 4,694 decisions.** Every number below was produced by the checkpoint
under test, not by `_safe_choice`. This is the check that the earlier deck study had
to retrofit after the fact; here it gates the result.

### Stage A — the model axis, deck held fixed at Marnie's Grimmsnarl ex

| arm | field win% | 95% CI | Glicko | RD | games |
|---|---:|---|---:|---:|---:|
| `il_alldays_3ep` | 86.6 ± 1.9% | [82.4, 89.9] | 1877.1 | 30 | 320 |
| `il_hfstream_comb_3ep` | 83.1 ± 2.1% | [78.6, 86.8] | 1841.7 | 30 | 320 |
| `il_bc_3ep` *(shipped)* | 81.9 ± 3.0% | [75.2, 87.1] | 1826.5 | 41 | 160 |
| `il_bc_4ep` | 80.6 ± 3.1% | [73.8, 86.0] | 1813.7 | 41 | 160 |
| `il_medium_3ep` | 79.1 ± 2.3% | [74.3, 83.2] | 1799.8 | 30 | 320 |
| `il_bc_2ep` | 78.1 ± 3.3% | [71.1, 83.8] | 1788.1 | 41 | 160 |
| `il_small_comb_2ep` | 75.0 ± 2.4% | [70.0, 79.4] | 1757.9 | 30 | 320 |

Unequal n is deliberate: the four arms where separation looked plausible at n=160 were
re-run to n=320; the three in the middle were not, because their gaps need ~6,900
games/arm to resolve and that is not worth the compute.

**Separation: 1 of 21 pairs.** Only `il_alldays_3ep` > `il_small_comb_2ep` (+11.6pp)
has non-overlapping 95% CIs. In particular **`il_alldays_3ep` vs the shipped
`il_bc_3ep` is NOT separated** — the +4.7pp is inside noise.

Two things that did *not* work, worth recording as negative results:

- **Capacity.** `il_medium_3ep` has 3.3× the parameters (10.99M vs 3.32M) and places
  5th of 7. Consistent with `notes/adr_metamon_grid_rescaled_not_literal.md`.
- **Offline accuracy as a proxy.** `il_bc_4ep` has the best offline accuracy of the
  07-26 family (0.7592) and places 4th. `il_small_comb_2ep` is 12 accuracy points and
  7× the calibration error worse than everything else, and still finishes 11.6pp back
  — a fraction of what the offline gap would imply. Offline accuracy is again a weak
  guide to play strength.

### Stage B — the deck axis, and the interaction

Field win% ± σ, 160 games per cell, same pool throughout:

| checkpoint | Grimmsnarl ex (3488) | Garchomp ex (625) | Spidops (784) | Alakazam (1542) | Lucario ex (1) |
|---|---|---|---|---|---|
| `il_alldays_3ep` | **86.6 ± 1.9** | 78.1 ± 3.3 | 71.2 ± 3.6 | 69.4 ± 3.6 | **55.0 ± 3.9** |
| `il_bc_3ep` | 81.9 ± 3.0 | 73.8 ± 3.5 | 70.0 ± 3.6 | 70.6 ± 3.6 | **28.8 ± 3.6** |
| `il_small_comb_2ep` | 75.0 ± 2.4 | 65.0 ± 3.8 | 53.1 ± 3.9 | 63.1 ± 3.8 | **48.8 ± 4.0** |

**The earlier deck conclusion replicates under a different pool.** The earlier study
measured `il_agent@mega_lucario_ex` at 29.7% against its 8-agent pool; the same weights
(`il_bc_3ep`) score **28.8%** here against a completely disjoint pool, and Grimmsnarl ex
reproduces too (78.4% → 81.9%). "Stop piloting Mega Lucario ex" is not a pool artifact.

**The interaction is the new finding.** How much the checkpoint matters depends on how
well the corpus supports the deck:

| deck | corpus episodes | spread across checkpoints | separated pairs |
|---|---:|---:|---|
| Marnie's Grimmsnarl ex | 3488 | 11.6pp | 1 of 21 (only the extremes, at n=320) |
| **Mega Lucario ex** | **1** | **26.2pp** | **both gaps clean** |

On Mega Lucario ex:
- `il_alldays_3ep` (55.0%) > `il_bc_3ep` (28.8%) — **+26.2pp**, CIs [47.3, 62.5] vs
  [22.3, 36.2], no overlap.
- `il_small_comb_2ep` (48.8%) > `il_bc_3ep` (28.8%) — **+20.0pp**, no overlap.

Both checkpoints that beat the shipped one on the data-poor deck are exactly the two
trained on **more than the 07-26 day**; `il_bc_3ep` is 07-26-only. Read together:
**extra training data buys robustness on decks the corpus barely covers, and buys
nothing measurable on decks it covers well.**

⚠️ **The confound this run cannot remove.** The extra-data arms also saw more *Mega
Lucario ex* episodes than the 1 in the 07-26 census — and because the local splits are
stubs, that count is unknown. "Better model" and "more episodes of this specific deck"
are not separated here. Separating them needs the per-day, per-deck census from the HF
corpus (§② above).

### Is this pool predictive of the ladder?

`scripts/pool_predictiveness.py` over the 7 pool agents carrying ladder anchors:

**Spearman ρ = +0.929 (n=7, permutation p = 0.0071)** — up from the +0.63 recorded for
the previous pool. The pool orders its own members close to their real ladder order.

Necessary, not sufficient: it says the *pool* is internally well-ordered, not that our
arms' win rates against it convert to ladder points. The known distortion is still
there — `romanrozen_strong_start` sits at local Glicko 1643.8, statistically level with
`tb_archaludon` (1647.3), while their ladder anchors are 950 vs 1196. Romanrozen runs
live MCTS with unbudgeted local think time. Expect this pool to flatter search agents.

## ④ Decide

**Fork 1 — one deck or a portfolio.** Still one deck. The deck ranking is stable across
all three checkpoints tested (Grimmsnarl ex first in 3 of 3), and the corpus is still
too imbalanced to support fielding several. Unchanged from the earlier study.

**Fork 2 — best-vs-field or most-learnable.** The Stage B interaction sharpens this:
"most learnable" and "wins against the field" are the *same* choice only while you keep
piloting a deck the corpus covers. The moment the deck is data-poor, the checkpoint
starts mattering a lot, and it is the broader-data checkpoint that wins.

**The new fork this run creates — which checkpoint to ship.** The two candidates are
not separated on the deck we would actually pilot:

- `il_bc_3ep` (shipped today): 81.9% [75.2, 87.1] on Grimmsnarl ex.
- `il_alldays_3ep`: 86.6% [82.4, 89.9] on Grimmsnarl ex — **not separated** — but
  **+26.2pp on the data-poor deck**, which is real.

The tie-breaker is not local win rate; it is that `il_alldays_3ep` is strictly more
robust where the corpus is thin, at identical inference cost (3.32M params, same
architecture, same bundle size). That is an argument for `il_alldays_3ep` **on
robustness grounds**, not on a demonstrated strength advantage — and it should be
written that way in any submit message.

## The one sentence

> **Pilot Marnie's Grimmsnarl ex with `il_alldays_3ep`: against an 8-agent pool spanning
> ladder 532–1196 it wins 86.6% ± 1.9% (277/320 games), on a deck with 3488 training
> episodes — versus 28.8% ± 3.6% for the currently-shipped `il_bc_3ep` on the Mega
> Lucario ex deck it pilots today, which has 1.**

Carried forward honestly: `il_alldays_3ep` is **not** statistically separated from
`il_bc_3ep`, `il_bc_4ep`, `il_hfstream_comb_3ep`, `il_bc_2ep`, or `il_medium_3ep` on
Grimmsnarl ex. What is decisive is the **deck**, and — only on a data-poor deck — the
**breadth of the training data**. All of it is local; the ladder has not seen any of it.

## What would change my mind

- **A ladder read-back that inverts this.** Three local-vs-ladder inversions are already
  on record. ρ = +0.929 makes this pool the most trustworthy we have measured, and that
  is still not a guarantee.
- **A per-day, per-deck census from the HF corpus** showing `il_alldays_3ep` simply saw
  many Mega Lucario ex episodes — that would recast the robustness finding as ordinary
  deck-specific familiarity, and it is the single most valuable follow-up.
- **A time-budgeted re-run of the pool.** If `romanrozen_strong_start` is throttled to
  the ladder's real budget, every arm's field win rate shifts, plausibly unevenly.
- **More games on the Grimmsnarl ex column.** The top-5 arms there are one tie; ~6,900
  games/arm would be needed to order adjacent pairs, and that ordering could differ
  from the one printed above.

## ⑤ Local leaderboard — how do our agents compare to everyone else?

Added 12 more challengers against the *same* 8-agent pool (our shipped agents plus
ladder-anchored public ones), then recomputed Glicko once over the union of all 39
agents. Challengers never play each other; the shared pool is what places them on a
single scale. Full table: the **Local leaderboard** sheet in the xlsx.

Top of the board, and the entries that matter:

| # | agent | Glicko | RD | 95% CI | real ladder |
|---:|---|---:|---:|---|---:|
| 1 | `il_alldays_3ep@marnies_grimmsnarl_ex` | 1877.1 | 30 | [1818, 1936] | — |
| 3 | `il_bc_3ep@marnies_grimmsnarl_ex` | 1826.5 | 41 | [1746, 1907] | — |
| 14 | `tb_archaludon` | 1685.9 | 30 | [1627, 1745] | **1196.1** |
| 15 | `romanrozen_strong_start` | 1665.5 | 30 | [1607, 1724] | 950.0 |
| 16 | `makthanithin_1084_baseline` | 1655.3 | 30 | [1596, 1714] | 1084.5 |
| 18 | `improved_prob_main` | 1640.9 | 41 | [1560, 1721] | 701.6 (verified) |
| 22 | `agent_core_improved` | 1602.4 | 41 | [1522, 1682] | **685.3** (verified) |
| 24 | `il_alldays_3ep@mega_lucario_ex` | 1551.2 | 41 | [1471, 1631] | **422.8** (verified) |
| 28 | `rule_baseline` | 1493.6 | 41 | [1413, 1574] | — |
| 33 | `il_bc_3ep@mega_lucario_ex` | 1282.3 | 41 | [1202, 1363] | ~398.7 (verified) |
| 34 | `mcts_il_agent` | 1256.7 | 41 | [1177, 1337] | **291.4** (verified) |
| 39 | `random_legal` | 1045.6 | 30 | [987, 1104] | — |

### ⚠️ Anchor provenance — most of these ladder numbers are alleged, not read

Of the 18 anchors the correlation uses, **only 4 are numbers we read off the ladder
ourselves**. The other 14 are self-reported by their authors: `tb_*` are
TomBombadyl's decoded-submission catalog mu, `wmh_*` are README/doc figures, and
`romanrozen`/`makthanithin` are notebook-title claims. None are verified.

**Corrected 2026-08-05: `agent_core_improved` is 685.3, not 804.0.** The 804.0 was a
single early read of sub 55162376. Four same-build resubmits settled at 699.0
(55191752), 692.7 (55219194), 666.1 (55224682) and 683.2 (55228113) — mean **685.3**.
Anchoring on best-ever inflated it by ~119 points.

| slice | ρ | n | p | what it rests on |
|---|---:|---:|---:|---|
| all anchors, corrected | **+0.765** | 18 | 0.0004 | 14 of 18 alleged |
| all anchors, with the stale 804.0 | +0.803 | 16 | 0.0004 | superseded |
| **verified-only** | **+1.000** | 4 | 0.0866 | our own ladder reads |
| the 8-agent pool alone | +0.929 | 7 | 0.0071 | **all 7 alleged** |

The +0.929 figure quoted earlier in this report rests *entirely* on unverified
third-party claims. The honest summary is narrower: on the four agents where we have
read both sides ourselves, the local board orders them **perfectly** (ρ = +1.000) — but
n=4 is not significant (p=0.087), so that is encouraging, not established.

### What this does and does not license

**The deck effect, restated as rating points.** `il_bc_3ep@marnies_grimmsnarl_ex`
(1826.5) and `il_bc_3ep@mega_lucario_ex` (1282.3) are the **same weights, 544 Glicko
points apart** — 23rd place separating rank 3 from rank 33. Nothing else in this study
moves a number that far.

**Our IL arms occupy ranks 1–13, above every public agent — and this is exactly the
shape that has misled us three times.** `tb_archaludon` sits at local 1685.9 with a
**real ladder score of 1196.1**, while our top arm has never played a ladder game on
this deck. Local rank 1 is a hypothesis about the ladder, not a claim on it.

**RETRACTED — the "internal inversion" was an artifact of the stale anchor.** An earlier
version of this section claimed `improved_prob_main` (local 1640.9, ladder 701.6) sat
*above* `agent_core_improved` while the ladder ranked them the other way, and used that
to argue the pool cannot resolve ~100-point differences between our own agents. With
`agent_core_improved` corrected to **685.3**, the ladder order is
`improved_prob_main` (701.6) > `agent_core_improved` (685.3) — **the same order as
local**. There is no inversion here. The pool got this pair right.

The real caution is smaller and different: the two are 16.3 ladder points apart, well
inside the ~±100 same-build drift, so this pair is a coin-flip that happened to land
right — not evidence the pool resolves fine distinctions. It still should not be trusted
to resolve the 4.7pp gap between `il_alldays_3ep` and `il_bc_3ep`.

**`mcts_il_agent` ranks 34/39** (1256.7), statistically tied with
`il_bc_3ep@mega_lucario_ex` (1282.3; CIs overlap) and clearly below `rule_baseline`
(1493.6). Consistent with its ladder history (600 → 294.7) — the local board does not
contradict the ladder here.

**Biggest anchor outlier:** `wmh_bellibolt`, ladder **836.0** but local 1231.1 (rank 35
of 39). Whatever it does well on the ladder, this pool does not reward.

## ⑥ The recommended checkpoint already has a ladder read — on the wrong deck

`models/il_alldays_0804` (this study's `il_alldays_3ep`, sha `1d67d1acdbb0`) was
submitted the same day as this sweep, as ref **55248985**, piloting **Mega Lucario ex**
(`agents/il_agent/deck.csv`). Its score timeline:

| read | score | note |
|---|---:|---|
| 2026-08-04 20:02 | 600.0 | the μ₀ prior, not a score |
| 2026-08-04 22:20 | 450.6 | settling |
| 2026-08-05 00:13 | **422.8** | latest |

**This is not a test of the recommendation.** The recommendation is `il_alldays_3ep`
piloting **Marnie's Grimmsnarl ex**; 55248985 pilots the 1-episode Mega Lucario ex deck
— the deck this whole study says to abandon. What it does provide is a *same-deck*
ladder check, and on that comparison the local board is consistent:

| agent (Mega Lucario ex) | local Glicko | ladder |
|---|---:|---:|
| `il_alldays_3ep@mega_lucario_ex` | 1551.2 | 422.8 |
| `il_bc_3ep@mega_lucario_ex` (= `il_agent`) | 1282.3 | ~398.7 |

Local ranks `il_alldays_3ep` above `il_bc_3ep` on this deck; the ladder does too. The
margin (24 points) is inside same-build drift, so this corroborates rather than proves.

**What it costs the recommendation:** nothing directly — but it removes the "maybe the
checkpoint alone is enough" option. Submitting this checkpoint on Lucario has now been
tried and lands at 422.8, in the same 395–450 band as every prior BC submission. If the
deck-switch hypothesis is right, the Grimmsnarl-ex build is the thing that has to move
that number, and it has never been submitted.

## ⑦ Equal-steps control — the gain is data breadth, not training longer

A concurrent session trained `models/il_alldays_equalsteps_0804` (sha `ee493052d8f4`):
the same all-days corpus at **38,562 steps instead of 127,748** — the equal-steps
control standing rule #4 asks for. Offline accuracy 0.7414 (vs 0.7583) but **ECE 0.0124,
the best-calibrated checkpoint in the family**. It was not in the original sweep; wired,
verified, and run against the identical pool.

| arm | Grimmsnarl ex | Mega Lucario ex |
|---|---|---|
| `il_alldays_3ep` (127,748 steps) | 86.6 ± 1.9% | 55.0 ± 3.9% |
| `il_alldays_equalsteps` (38,562 steps) | 79.4 ± 3.2% | 56.9 ± 3.9% |
| `il_bc_3ep` (07-26 only) | 81.9 ± 3.0% | 28.8 ± 3.6% |

- On **Grimmsnarl ex**, equal-steps is **not separated** from either `il_alldays_3ep`
  (−7.2pp) or `il_bc_3ep` (−2.5pp) — consistent with the rest of the model axis.
- On **Mega Lucario ex**, equal-steps scores **56.9%**, statistically identical to the
  full 127,748-step run (+1.9pp, not separated) and **separated from `il_bc_3ep` by
  +28.1pp** — the same clean robustness gap.

**This isolates the cause.** The 3.3× extra optimisation steps buy nothing measurable;
the robustness on the data-poor deck reproduces at **less than a third of the training
compute**, purely from training on more days. It is the *breadth of the corpus*, not the
length of the run, that makes the model hold up on a deck it barely saw. It also removes
one horn of the ②-confound: the effect survives at matched steps, so it is not an
artifact of the longer run.

Calibration again fails to predict play: equal-steps is the best-calibrated checkpoint
(ECE 0.0124) and mid-pack on Grimmsnarl ex.
