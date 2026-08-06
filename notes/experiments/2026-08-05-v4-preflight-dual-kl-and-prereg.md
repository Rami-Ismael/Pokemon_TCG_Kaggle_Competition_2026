# 2026-08-05 — RL Pipeline v4 pre-gate work: dual-reference KL, and pre-registering the pool

Everything in v4 Phases 2–3 is conditional on the §0 kill-gate, which resolves
~08-08. This session did only what is correct under **either** branch of that
gate: the standing daily ladder item, the instrumentation half of work item 2,
and work item 7. No training run was started, no submission was made.

---

## 1. Standing daily item — the ladder moved, and not the way §9 assumed

`submission_ledger.py refresh` → 3 new readings + 1 new ref at 19:50; 2 more at 20:57.
`check_leaderboard.py` → rank **4235 / 6361**, team score **547.7** (19:50 read: 4761/6359, 493.9).

| Ref | What | Reading trail | Latest |
|---|---|---|---:|
| `55253900` | selfplay_g1_ref430k (**the kill-gate**) | 248.6 → 267.4 → 267.4 → 254.9 | **254.9** *(unchanged at 20:57)* |
| `55270787` | il_alldays_0804, Grimmsnarl redeck | 361.8 → 314.1 → 353.9 → 366.5 | **366.5** |
| `55279487` | il_agent_v2, BC on the 9-day corpus | 493.9 → 547.7 *(2 reads)* | **547.7** |

Four things follow, and one of them is a correction to this note.

**The kill-gate ref is drifting down, not up.** 267.4 → 254.9 on its fifth
reading, three days from the verdict. It sits ~295 below the 550 PASS line and
~195 below the 450 FAIL line. Nothing is settled until 08-08 and this note does
not call the gate — but the trajectory is not ambiguous-band behaviour.

**§7.4's deck finding weakened rather than reversed.** The Lucario/Grimmsnarl
gap on identical weights went 107 points → 52 (418.0 vs 366.5), i.e. further
*inside* the ~±100 same-build band. The confident local claim that the deck axis
was worth 544 Glicko points stays falsified; the reverse is still not
established, and n is still 1 per arm.

**`55279487` is the highest number a learned agent has posted, and it is not yet
a number.** Two readings: 493.9, then 547.7 an hour later.

*(Corrected at 20:57. The first version of this note read 493.9 as "still
falling through the μ₀ = 600 prior," reasoning from `55248985`'s 600.0 → 418.0
descent. The next reading went **up**. The directional guess was wrong; what
survives is only that 2 readings do not settle a score.)*

**This is the one thing here that could undercut §0.** §0's argument is that the
learned and heuristic families do not overlap, with a ~377-point median gap
against a ~±100 same-build spread. The lowest heuristic score is 602.6 and
`55279487` is at 547.7 and rising — a 55-point gap. Still literally
non-overlapping among settled scores, but no longer the clean separation the
argument leans on. If it settles above 602.6, the §0 FAIL branch is reasoning
from a premise the ladder has withdrawn, and that has to be checked before
08-08 rather than discovered after.

**Both active slots hold learned experiments.** The §9.4 slot-1 heuristic
restore was not executed; a concurrent session spent the slot on `il_agent_v2`.
That is the two-experiments-back-to-back configuration §9.2 identifies as how
the team fell 804.0 → 395.0 and ~4,500 ranks on 08-03. There is no heuristic
build under the current floor. Flagged for the user; no submission made.

---

## 2. Work item 2, logging half — KL against two references

**The defect.** `kl_prior_policy` is retargeted on every promotion, so after
the first retarget `kl_to_prior` answers "how far from the last promoted best"
and *nothing* answers "how far from the human prior". That second question is
the entire reason the KL anchor exists: v4 §2 accepts on-policy PPO instead of
the paper's offline retrain-over-the-union specifically because the anchor is
supposed to make forgetting watchable. It was not watchable.

**The change** (`src/pokemon_tcg/pufferl_kl.py`, `scripts/train_ppo_puffer.py`):
a second frozen reference, `--il-prior`, never retargeted, logged as
`kl_to_il_prior` globally and as `kl_il` per SelectContext. It is **diagnostic
only** — detached, never multiplied by `kl_coef`, never added to `loss`. Adding
a second pull would change the objective; this changes only what is written to
`train_metrics.jsonl`.

Two details that are not incidental:

- **The second forward pass is skipped while the two references hold the same
  policy.** Generation 1 passes one checkpoint as both, so until the first
  promotion `kl_to_il_prior == kl_to_prior` exactly. In the measured regime —
  1 promotion in 170 gates (§3.2) — that is the whole run, and the run pays
  nothing for the instrument.
- **Whether they are the same is decided from the weights, not the caller's
  intent** (`kl_math.same_policy_weights`). `run_selfplay_g3.sh` anchors to a
  promoted teacher (`selfplay_g1/refs/u430080`) while initialising from a
  gen-2 checkpoint — exactly the shape where the shortcut would be wrong, and
  where taking it would silently log the anchor's KL under both names. The
  comparison is over the `actor` submodule only, because `PTCGPufferPolicy`
  randomly initialises a value head, so two wrappers built from the same
  checkpoint path differ in the critic while being identical policies.

**Verification.** Three CPU smoke runs (`--smoke`, 3 updates, seed 42):

| Run | Shape | Result |
|---|---|---|
| A | `--il-prior` ≠ `--kl-prior` (gen-2 shape) | series distinct from update 1: `kl_to_prior` 0.019 vs `kl_to_il_prior` 0.877 |
| B | identical refs, gate forced at update 2 | caught the labelling bug below |
| C | identical refs, gate forced at update 2 | control clean, claim reproduced |

Run C, which is the chart:

| update | kl_to_prior | kl_to_il_prior | refs distinct |
|---:|---:|---:|---|
| 1 | 0.0322 | 0.0322 | no *(control)* |
| 2 | 0.0374 | 0.0374 | no *(control)* |
| 3 | **0.00009** | **0.0225** | yes |

The pre-retarget rows are the control: the two references are the same policy,
so the series must coincide exactly, and they do. Post-retarget the anchor
reads 9e-05 — "no drift at all" — while drift from the human prior is **250×
larger**. Per-context is the same story (ctx 0: `kl` 0.00016 vs `kl_il`
0.03737). That gap is precisely what could not be logged before.

Chart: `reports/figures/kl_dual_reference.png`. It is a 3-update CPU smoke run
— an instrument check, not a training result, and labelled as such.

**A bug this found in itself.** Run B reported `kl_refs_distinct: True` while
both references were the same checkpoint, because the initial comparison ran
over the whole `state_dict` and `PTCGPufferPolicy`'s value head is randomly
initialised per construction. The logged *numbers* were unaffected (the run took
the slow path and reproduced the shortcut's values exactly, which is its own
cross-check), but the label was wrong and the run paid for a redundant forward
pass. Fixed by comparing the actor only; run C is the re-verification.

**A calibration point worth keeping.** Run A puts two BC checkpoints
(`il_agent` vs `il_alldays_0804`) ~0.88 nats apart. Against §4.2's ladder —
where 0.1–0.3 is the entire operating range of three full runs and 1.0 was never
observed — two independently-trained BC checkpoints are nearly a full "anchor
has stopped binding" apart from each other. The §4.2 stop rule (`> 0.6`
sustained) is about `kl_to_prior`, the term in the loss; `kl_to_il_prior` has no
stop rule attached and should not inherit that one.

Tests: `tests/test_kl_mask.py`, 8 passed (3 new, covering the load_state_dict
copy, the gen-2 shape, the post-retarget divergence, and key/shape mismatch).

---

## 3. Work item 7 — pre-registering the pool's ordering

**Why it is worth anything.** The anchored pool's ρ = +0.929 is in-sample: the
pool has never been tested on a submission it did not help select. §6.3 refuses
to let it gate submissions for that reason. A pre-registration is the only thing
that converts it, and it costs nothing.

**What is registered, and what deliberately is not.** The registered claim is
the **ordinal** prediction. A score-level map exists and is frozen, but it is
not the claim:

- Fit on the compounded local Glicko scale, n = 19 overlapping agents:
  `ladder = -554.4 + 0.8076 × local_glicko`, **residual SD 198.7**.
- In-sample Spearman **ρ = +0.718** (permutation p = 0.0007).
- Residual SD is ~2× the ~±100 same-build ladder noise band, so the 95% band is
  ±397 — wide enough that a point prediction claims almost nothing.
- Most anchors are self-reported rather than ledger-verified
  (`ladder-anchors-mostly-alleged`); the verified-only slice cannot support a
  fit at all. Spearman is ordinal, so registering the ordering is both what
  §6.3 asks for and the part of the claim the alleged anchors cannot move.

The coefficients are frozen in source with a `FROZEN` fence, and the ratings
they were fit on are snapshotted immutably to
`reports/pool_prereg_source_glicko.json` — `reports/glicko_ratings.json` keeps
compounding across runs and would not be reproducible from this commit. Refitting
after an outcome lands is the specific failure this arrangement makes impossible.

**Slate locked today** (`reports/pool_prereg.jsonl`, append-only):

| # | candidate | local Glicko | predicted ladder | status |
|---:|---|---:|---:|---|
| 1 | `proto` | 1808.0 ±58 | 905.7 | never submitted — purest out-of-sample test available |
| 2 | `mega_lucario_restore` | 1748.7 ±30 | 857.9 | §9.4 slot 1, awaiting authorization |
| 3 | `rule_baseline` | 1601.7 ±30 | 739.1 | never submitted — spreads the slate across the range |

Two slots of the five are deliberately unlocked: the post-gate candidates do not
exist yet. Locking the *coefficients* today is what makes their predictions
non-tunable when they do.

**`il_agent_v2` is excluded, on purpose.** It was submitted before this session
and already has a reading, so any prediction for it would be post-hoc and would
contaminate the out-of-sample number. Worth recording as an unregistered
observation, though: the frozen map puts it at **831.0**, its submitter
predicted **380–460**, and it is reading **547.7** on 2 reads and rising. It
sits *between* the two — above the human prediction, below the pool's. Both
cannot be right. If it settles low, the pool's map is miscalibrated upward for
learned agents, the same direction as all four recorded local-vs-ladder
inversions; if it settles high, the human prediction was the pessimistic one and
that is itself worth knowing. Either way it is excluded from the out-of-sample ρ.

**Tooling.** `scripts/prereg_pool_prediction.py` with `lock` / `show` /
`expects` / `bind` / `score`. `expects` prints the string to hand to
`submission_ledger.py log --expects`, so the prediction lands in the ledger at
submit time rather than being reconstructed later. `score` refuses to report an
out-of-sample ρ until ≥4 locked candidates have settled reads.

Chart: `reports/figures/pool_prereg_calibration.png` — anchors, fit, ±2 SD band,
and the three pre-registered candidates plotted with no actual yet.

---

## What is explicitly NOT done

- No generation-4 training run. The gate is unresolved; g2 and g3 already burned
  2.5M steps for zero promotions.
- No hyperparameter changes applied (LR 1.5e-4, entropy 0.01, β 0.02). Those are
  the *other* half of work item 2 and are conditional on §0 PASS.
- No sweep. `ppo-hyperparameter-sweep-negative-result` and §5 both say not before
  the gate.
- No submission. The heuristic restore is recommended and awaiting the user's
  call; see §9.4 and the ladder section above.
