# Critic leaf + margin-gated override on the depth-1 bandit (controlled comparison)

Follow-up pre-registered in `2026-08-11-il-ucb1-depth1-rerank.md`, reshaped by
two 08-12 events: (1) the perspective-inversion bug voided that card's 0/10
and the sibling's 35% — `evaluate_state`'s quality is an OPEN question again;
(2) the dataviz session's retrained critic
(`critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42`) passed the
08-09 acceptance (AUC 0.741, beats base-rate constant, clip 2.5%) with Platt +
centering fitted (`calibration.json`; ECE 0.044 after Platt). Its
shuffled-label control is still training in that worktree — final acceptance
of the critic itself is pending; battles here are provisional until it clears.

- **Hypothesis:** with a calibrated, centered, perspective-correct leaf, the
  depth-1 bandit's overrides of the IL top-1 are net-positive — and a small
  override margin δ (only displace the IL pick when the value gap is decisive)
  dominates unconditional override, because low-margin disagreements are
  coin flips where the teacher is right more often than the leaf.
- **Independent variables:** leaf (evaluate_state, sign-fixed vs critic) and
  override margin δ — swept one at a time against the shared control.
- **Baseline/control:** bare `bc_alldays52_jun16_aug07_seed42` (no search).
  δ=∞ recovers it exactly, so the sweep has a built-in anchor.
- **Arms** (all same checkpoint, deck, device; `agents/search_arms/`):
  1. `bc_alldays52_ucb1_rerank` — evaluate_state leaf, sign-fixed, δ=0
     (rerun of the voided Stage 2; running as of this card's creation)
  2. `bc_alldays52_ucb1_criticleaf` @ δ=0 — critic leaf, unconditional
  3. `bc_alldays52_ucb1_criticleaf` @ δ∈{0.1, 0.25} — margin-gated
     (OVERRIDE_MARGIN env var; leaf units: centered value in [-1,1])
- **Mechanism numbers (live-game smoke, 1 game):** critic leaf at δ=0
  overrides 34/51 = 67% of MAIN decisions; ~120 sims/decision (critic
  inference is ~40× slower than evaluate_state per leaf, still >>K=8);
  ~80 s/game local unbudgeted — cabt's 600 s bank with 2–3× CPU factor holds.
- **Metric & protocol:** per arm, 5 mirrored pairs (10 games) vs control,
  head-to-head, `--no-glicko-persist`, early-stop when the Wilson 95% CI
  excludes the decision boundary (the 08-11 card's early-stop precedent).
  Machine-load caveat: two unrelated training jobs are running (the 08-12
  weighted-BC run and the dataviz shuffled-control); head-to-head pairs are
  load-symmetric within a run, but cross-run comparisons of *game counts and
  timings* are not — decisions are made on win rates only.
- **Primary metric:** win rate vs control with Wilson 95%.
- **Guardrail:** override rate at each δ (mechanism stays live: must be >0),
  fallback rate 0, LEAF_DIAG.encode_none ≈ 0.
- **Pre-registered decision:** an arm is a candidate for anchored-pool testing
  if ≥60% over 10+ games with CI excluding 50%; dropped if CI excludes any
  value >50% (i.e. upper bound ≤50%); otherwise extend that arm to 25 pairs
  before judging. δ arms are compared only to control, not each other, until
  one clears the pool bar. No ladder claims from any of this
  (`leaderboard-check` applies before any submission talk).
- **Cost estimate:** ~80 s/game (critic arms), 10 games/arm ≈ 15 min each;
  evaluate_state rerun ~5–30 min; all CPU, nice'd, no MPS.
- **Prior work checked:** the 08-11 card and its void notice; the dataviz
  session's `2026-08-11-search-on-alldays-bc.md` (tree wrapper — different
  mechanism, shares the critic; coordinated, not duplicated);
  `search-leaf-value-must-be-centered`; `il-mcts-tb-tree-negative` (reopen
  conditions: calibrated critic through the seam ✓, ≥10% mechanism gate ✓).
- **Scale confound:** inference-time layer over the full-corpus checkpoint;
  the critic saw only 7 recent days — a null may still be "critic undertrained
  on too narrow a window", say so if it lands that way.
- **Deck scope:** Mega Lucario ex mirror throughout, as before.

## Results
- **evaluate_state sign-fixed δ=0 rerun:** (pending — appended on completion)
- **critic leaf δ=0:**
- **critic leaf δ=0.1:**
- **critic leaf δ=0.25:**
- **Decision:**
- **What we learned:**
- **Belief update:** <Rami's one-liner>
