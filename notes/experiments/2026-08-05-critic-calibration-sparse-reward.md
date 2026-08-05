# Does the critic learn a useful value signal under sparse terminal reward? (calibration audit with negative control)

- **Hypothesis:** A critic trained by regressing `cls_hidden → outcome ∈ {0, 0.5, 1}`
  on sparse terminal-only rewards learns a *discriminative and calibrated* V(s) —
  **mechanism:** the trunk's board features (prizes remaining, bench state, hand
  size, energy) correlate with the eventual outcome well before the terminal
  step, so MSE regression can separate winning from losing states even though
  every intermediate step pays 0. If the sparse signal is too thin, the critic
  collapses toward the base rate and V(s) carries no per-state information —
  which would invalidate both advantage-weighted IL (S2-E4) and using V(s) as a
  search leaf evaluator.

- **Independent variable:** none swept — this is a **measurement** with
  controls, not a comparison. Three arms, identical eval protocol:
  1. **Real critic** — `scripts/train_critic.py`, trunk warm-started from
     `models/il_alldays_0804` (the current all-days BC actor), outcomes as-is.
  2. **Constant baseline** — predict the eval-day base rate for every row (plus
     the script's built-in constant-0.5 gate).
  3. **Shuffled-outcome critic (negative control)** — same training run with
     episode outcomes permuted across episodes. Must land within noise of the
     constant baseline; if it "beats" it, the eval leaks and every other number
     in this card is void.

- **Baseline:** constant base-rate predictor on the eval day (design: a critic
  that cannot beat "predict the average" has learned nothing per-state).

- **Metric & protocol:**
  - Eval split = held-out day (2026-07-27), never trained on.
  - **Brier/MSE** vs both constant predictors (the script already prints the
    0.5 gate; add base-rate).
  - **AUC** on win-vs-loss rows (ties are measured at ~0 frequency — see
    2026-08-05 finding: 0 ties in 656 benchmark games + 60 raw episodes — so
    win/loss AUC covers essentially all rows).
  - **10-bin reliability diagram + ECE** on eval rows.
  - **Calibration by game phase:** bucket rows by within-episode decision-index
    quartile. A useful critic should sharpen (predictions migrate toward 0/1,
    AUC rises) from Q1 → Q4; a flat profile means it only reads the base rate.
  - Free secondary evidence: the online PPO critic's `explained_variance`
    curve already plotted in `reports/kl_selfplay_report.html`
    (`pufferl_kl.py:222`); read it alongside, but the offline critic is the
    primary instrument because no PPO critic checkpoint survives on disk
    (snapshots are actor-only; no `policy_full.pt` found).

- **Pre-registered decision:**
  - **Useful:** eval MSE ≥5% (relative) better than the base-rate constant AND
    win/loss AUC ≥ 0.60 → V(s) carries per-state signal; green-light spending
    runs on adv-weighted IL arms and critic-as-leaf-evaluator search experiments.
  - **Calibrated:** ECE ≤ 0.05 → usable as a probability. If useful but
    miscalibrated (AUC good, ECE > 0.05), try post-hoc temperature/isotonic
    fit on a held-out slice *before* any policy run — recalibration is minutes,
    a policy run is hours.
  - **Drop:** AUC < 0.55 → the sparse signal did not produce a per-state value
    function at our data scale; do not feed V(s) into search, and S2-E4
    advantage weighting is dead until the critic improves.
  - **Abort:** shuffled-outcome control beats the constant baseline by more
    than noise → eval leakage; stop and fix the eval before reading any arm.

- **Cost estimate:** BLOCKER first — `train_critic.py` resolves splits via
  local `resolve_split_dir()`, and local raw splits are stubs (~24 train / 12
  eval files); it has no `--data-source` streaming flag. Wiring the HF
  streaming corpus in (pattern already exists in `train_il.py --data-source
  auto`) is ~30–60 min. Training: docstring sizing `--total-steps 12900` @
  batch 64 ≈ 1 epoch over the train day; estimate 2–4 h MPS per arm × 2
  trained arms (real + shuffled). Single seed — treat as provisional; the
  shuffled control doubles as a variance read. Disk: one ~26 MB checkpoint per
  arm. Nice everything; no overlap with other MPS jobs.

- **Prior work checked:**
  - `src/pokemon_tcg/ppo.py` docstring — γ=1, terminal-only, advantage =
    `outcome − V(s)`: the critic is the *entire* credit-assignment mechanism,
    which is why this measurement gates everything downstream.
  - `scripts/train_critic.py` — already has the constant-0.5 MSE gate and the
    rule "a critic that can't beat a coin flip must not cost a policy run";
    this card extends that gate to calibration, discrimination, and a leakage
    control.
  - `notes/rl_pipeline_v1.md` §2.1 row E4 — the consumer of this critic.
  - IL-prior MCTS result (ladder 600→294.7) — the memory that flagged "critic
    still needs calibration"; no calibration measurement exists anywhere in
    `notes/` or `reports/` (checked 2026-08-05; `wmh_pool_calibration.json` is
    opponent-pool calibration, unrelated).
  - `scripts/train_ppo_puffer.py:272` — documents explained_var ≈ −0.9 during
    critic cold start, i.e. the online critic starts *worse than the mean
    predictor*; whether it ends better is exactly what this experiment answers
    offline.

## Result (filled 2026-08-05)
- **Observed:** (128,000 held-out eval-day rows each; single seed 42; CPU-forced,
  MPS was occupied — commands: `scripts/train_critic.py --init-from
  models/il_alldays_0804 --out models/critic_trainday --data-source hub
  --num-workers 2 --seed 42` [+ `--shuffle-outcomes` for the control], then
  `scripts/eval_critic_calibration.py` on both; full JSON in
  `reports/critic_calibration_trainday{,_shuffled}.json`)

  | arm | MSE (vs base-rate 0.2485) | AUC | ECE | phase AUC Q1→Q4 |
  |---|---|---|---|---|
  | real critic | 0.2047 (**+17.6%**) | **0.759** | 0.062 | 0.61 → 0.73 → 0.84 → 0.84 |
  | shuffled control | 0.3496 (−40.7%) | 0.494 | 0.233 | 0.49 → 0.50 → 0.49 → 0.50 |

  - Real critic passes both "useful" bars (≥5% rel. MSE gain, AUC ≥ 0.60) with
    a genuine value-function signature: discrimination and sharpness rise
    monotonically toward late game.
  - Calibration misses its bar by a hair (ECE 0.062 vs 0.05), driven by
    early-game rows (turn ≤ 4 ECE 0.11) — the pre-registered branch says
    post-hoc temperature/isotonic fit, not a retrain.
  - Negative control is textbook-clean: chance AUC flat across every phase,
    worse-than-constant MSE. The abort condition (control beating the
    constant = eval leakage) did not fire.
  - 0 ties in 128k rows — the morning's tie-frequency finding at 200× n.
  - Train MSE (~0.12) vs eval-day MSE (0.205): a real train→held-out-day gap;
    the 17.6% gain is the honest, generalized number.
- **Reconciliation with the concurrent v2 audits** (discovered at merge time —
  origin/main's `reports/critic_audit_critic_{td,mc}.md`, same eval day):
  a parallel session's MC critic **failed** its MSE gate (0.2679 vs 0.2499,
  22.2% of raw outputs outside [0,1]) and its TD critic failed harder
  (0.3312), leading to "advantage arms blocked". This card's critic **passes
  the same criterion** (0.2047, 0.0% out-of-range). The two MC runs differ in
  trunk init (`il_alldays_0804` here vs `il_agent` there) AND train days
  (07-26 here vs 07-01+07-26 there) — attribution is confounded; the trunk
  init is the prime suspect. Consequence: the v2 "arms blocked" decision was
  made against a weaker critic artifact and is worth revisiting with
  `models/critic_trainday`, after its (iii) hand-read. A controlled
  same-days init swap would settle attribution.
  Also learned from the small-scale control: at 40-episode dry-run size the
  relabel hash can land 60/40, letting the control "beat" the const-0.5 gate
  by learning a ~0.6 constant on a 0.59-base-rate eval day — judge controls
  against the BASE-RATE constant and AUC (this card's audit does), never
  const-0.5 alone.
- **Decision:** **adopted** (as "useful, pending recalibration") — V(s) under
  sparse terminal reward carries real per-state signal; green light for
  adv-weighted IL arms (S2-E4) and critic-as-leaf-evaluator search
  experiments, with a temperature fit on a held-out slice first if V(s) is to
  be read as a probability.
- **What we learned:** sparse terminal-only reward is sufficient supervision
  for a discriminative value function at our data scale (one day, 4,554
  episodes) — credit assignment via `outcome − V(s)` is not starved by the
  reward's sparsity. Transferable: the shuffled-label control turned a
  "trust me" number into an audited one for the price of one duplicate run;
  and per-episode (not per-row, not per-seat) relabeling is the correct
  permutation unit when labels are episode-constant and seat is readable
  from the state.
