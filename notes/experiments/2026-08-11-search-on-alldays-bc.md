# Search on top of the all-days BC policy — mechanism check, then controlled comparison

Design: **controlled comparison** (search wrapper on/off, one component), with a
**mechanism check as step 1** that can kill the experiment before any battle time
is spent. Not an ablation — the question is "does the wrapper add strength",
attribution *inside* the wrapper (prior vs critic leaf) only becomes a question
if the wrapper wins. Not a search-budget sweep — N=30 vs N=0 was already measured
on the original checkpoint (−2.5 pp at 11× cost); the budget axis is a follow-up
only if this comparison shows signal.

- **Hypothesis:** Wrapping `bc_alldays52_jun16_aug07_seed42` (the strongest BC
  checkpoint: full 53-day corpus) in `mcts_il_agent`'s search — with a
  freshly trained outcome critic (trunk = `bc_alldays52`, recent-window data),
  temperature-fitted and *centered*, as the leaf value — raises
  pooled win rate on the 8-agent anchored pool by ≥5 pp over the bare checkpoint,
  because determinized multi-world rollouts with a calibrated value can override
  the policy's myopic picks (the mechanism the PKA report credits for beating
  their heuristic).
- **Independent variable:** the search wrapper, on/off. Checkpoint fixed at
  `bc_alldays52_jun16_aug07_seed42` (Rami's call, 08-11: wrap the most powerful
  IL model, not the older `il_alldays_0804`). This trades comparability with the
  earlier search measurements for relevance to the current best model — accepted
  deliberately; no existing search number covers this checkpoint, so both cells
  (bare and wrapped) are measured fresh inside this experiment.
- **Baseline:** bare `bc_alldays52_jun16_aug07_seed42`
  (`agents/il_arms/il_bc_alldays52_final`), same deck, same device (CPU — MPS is
  15–20× slower per move).
- **Metric & protocol:**
  - **Step 1 — mechanism check (minutes, kills early):** play ~10 self-play games
    with the wrapper; count the fraction of non-forced decisions where the search
    output differs from the bare policy argmax. This is the standing reopen
    condition from the tb-tree negative result: a calibrated critic through the
    evaluate_node seam AND ≥10% decision change, in that order.
    **If <10%, stop — verdict "decoration with this prior too"; step 2 is not run.**
  - **Step 2 — strength (only if step 1 passes):** both arms vs the 8-agent
    anchored pool (`scripts/benchmark_agents.py`), 50 mirrored games per opponent
    per arm (400 games/arm), pooled win rate with Wilson 95% + Glicko.
    `run-fallback-diagnostic` on the wrapper BEFORE the battery — a silent
    `_safe_choice` fallback invalidates the comparison. Also verify (md5) that
    both arms' bundled deck files are identical before the battery — deck, not
    checkpoint, has been the dominant axis before, and a deck mismatch would
    un-control the comparison.
- **Primary metric:** pooled win rate on the anchored pool (Δ arm − baseline).
- **Guardrail metrics:** (a) sec/game against the ladder's 600 s/match bank with a
  2–3× CPU factor — locally ~14 s/game unbudgeted, so report the budgeted number;
  (b) fallback rate; (c) search-vs-policy top-1 agreement (interpretation aid,
  not a target).
- **Pre-registered decision:** adopt as a submission candidate only if Δ ≥ +5 pp
  and the 95% CI excludes 0 (and only then does `leaderboard-check` +
  ledger-refresh apply before any submission). Otherwise drop and log the
  negative result. Scale-confound statement required either way: a null here says
  "this wrapper adds nothing to this checkpoint at this data scale", not "search
  can't help".
- **Cost estimate:** critic training (step 0) is the new cost: `train_critic.py`,
  trunk `bc_alldays52`, MC target, recent-window days (the v5 corpus window per
  the 08-11 retrain commit's own instruction), plus the mandatory shuffled-label
  control run and temp fit — CPU, projection reported at launch (the one-day
  retrain was a small CPU job; more days scale roughly linearly). Acceptance
  before use: beats the base-rate constant on eval day 2026-08-09 (NOT 07-27 —
  that day is inside bc_alldays52's training data) and the shuffled control
  stays clean. Step 1 ≈ 5–10 min CPU. Step 2 ≈ 400 games × ~14 s (wrapper) +
  400 × ~0.5 s (bare) ≈ 1.7 h wall clock, CPU-only, nice'd, no MPS contention.
  Worktrees lack `models/` — symlink from main first (`run-ptcg-battle` doctor
  prints the exact fixes).
- **Prior work checked:** prior-only MCTS is decoration (tree moves 2.7% of
  decisions; N=30 −2.5 pp); shipped critic worse than a constant — use the
  outcome-critic recipe (08-11 retrain: old trunk + day-07-26 data gets AUC
  0.731 on 07-27 but FAILS on eval day 08-09 — worse than a constant; hence
  step 0 retrains it on trunk `bc_alldays52` + recent window, per that commit's
  own instruction); search leaf must be centered (+0.084 constant leaf moved 10.5%
  of decisions via turn-parity alone); IL-prior MCTS 67% local → 294.7 on ladder
  (never call transfer on a first local reading); il_vs_mcts full-field on the
  ORIGINAL checkpoint: h2h 15–15, field 25.5% vs 31.0% (290 games/side); the
  search+prior dashboards of 08-04/06 measured a heuristic host, not this
  question, and every one of them ran the older `il_alldays_0804` or original
  checkpoint — `bc_alldays52` has never been inside a search wrapper. All search
  measurements to date are the Lucario mirror — this
  experiment inherits that deck scope and says so; the meta-deck cell stays
  empty unless a separate deck-selection-skill pass opens it.

## Result (filled 2026-08-12)
- **Observed:**
  - Step 0 (critic): `train_critic.py --init-from models/bc_alldays52_jun16_aug07_seed42
    --target mc --data-source hub --hub-days 2026-08-01..07 --seed 42` →
    `models/critic_outcome_bcalldays52trunk_2026-08-01_to_2026-08-07_seed42`.
    Audit on eval day 2026-08-09 (128k rows, `eval_critic_calibration.py`):
    **AUC 0.741, MSE +12.0% vs constant** (old day-0726 critic on the same day:
    AUC 0.660, WORSE than constant). Platt+center fit on day 2026-08-10 (kept
    separate from the audit day): a=0.4901 b=0.1468 center=0.5163, leaf mean
    +0.051→+0.019. Shuffled-label control (finished 2026-08-13): CLEAN — MSE
    0.2499 vs constant 0.2500 (no gain), accuracy exactly the 0.5230 base rate;
    the real critic's +12.0% is signal, not leakage.
  - Step 1 (mechanism probe): `probe_mcts_mechanism.py --opponent
    il_bc_alldays52_final --pairs 5 --critic-dir <new critic>` →
    `reports/probe_mechanism_bcalldays52_calibcritic.json`. 615 decisions, 591
    searched, **changed_vs_prior 11/591 = 1.9%** (bar: ≥10%). The search truly
    ran: 17,368 leaf evaluations, frac_nonzero 1.0, leaf mean −0.116 std 0.555,
    maxdepth 30, 0 fallbacks, probe games 5–5.
- **Decision:** **STOP at step 1 per the pre-registered rule.** Step 2 (400-game
  battery) not run. Verdict: with the strongest prior AND a fresh, calibrated,
  centered critic, this search is still decoration — the N=30 visit budget
  (~3 visits per root child at 9.3 children) cannot overturn a confident BC
  prior, and at 428 ms/decision the budget cannot be raised ~10× inside the
  600 s/match bank.
- **What we learned:** the earlier "search is decoration" results were NOT an
  artifact of the weak prior or the broken/uncentered critic — fixing both to
  best-available changes the decision-override rate from 2.7% to 1.9% (i.e. not
  at all). The binding constraint is the search's visit budget vs prior
  confidence, not the quality of either model. Scale confound: this null is
  about THIS wrapper at N=30 under the ladder time bank; it does not say
  lookahead can't help in principle (a different search design with a larger
  effective budget per decision — e.g. fewer, deeper decisions — is untested).
- **Byproduct kept:** the new critic is a real asset independent of search —
  first critic to pass the current-meta bar (the v5 adv-exp arm wanted exactly
  this retrain). Backed up 2026-08-13 to HF `Rami/ptcg-s2v2-arms` (critic dir
  incl. calibration.json, shuffled control, both audit reports — 9/9 files
  verified against the remote listing).
- **Belief update:** <Rami's one line, not a paraphrase>
