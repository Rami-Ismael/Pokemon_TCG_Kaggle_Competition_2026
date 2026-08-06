# League pool composition — do weak training opponents hurt play vs strong ones? (controlled comparison + observational diagnostic)

- **Hypothesis:** Under the current league draw (`sample_league`: 50% mirror /
  30% past ckpts / 20% public pool, uniform *within* each bucket), the learner
  collects most of its positive reward from the weakest pool members, so
  pooled winrate rises while winrate vs held-out strong opponents stays flat
  or falls. Mechanism: policy gradient weighs every win equally; the cheapest
  reward is exploiting weak-agent mistakes, and those exploit behaviors are
  exactly the ones a strong opponent punishes (documented in this repo: the
  exploiter checkpoint hit 94% vs its BC target seeds but lost to
  rule_baseline and dropped to 38.5% vs the all-days model).
- **Independent variable:** composition of the public-pool bucket only —
  uniform-over-all-agents (current) vs strong-only. Mirror/past-ckpt buckets,
  mix ratios, steps, seeds, decks all held fixed. (PFSP-style loss-weighted
  sampling is a *different* variable — separate follow-up experiment if this
  one shows an effect.)
- **Baseline:** Phase-3 run with the current uniform pool, same step budget.
- **Metric & protocol:**
  - **Step 0 (instrumentation, ~zero cost):** episode rows already carry
    `"opponent": opp_spec` (`selfplay.py:239`) — add per-opponent-bucket
    winrate to the trainer's periodic eval log. No conclusions without this;
    it is the measurement the hypothesis is *about*.
  - **Step 1 (observational diagnostic, zero extra training):** on the first
    baseline run, plot winrate-vs-weak, winrate-vs-strong-pool, and
    winrate-vs-held-out-strong across checkpoints. If they rise together, the
    concern is not materializing → stop here, card closes negative.
  - **Step 2 (intervention, only if Step 1 shows divergence):** Arm B rerun
    with strong-only pool, same seed/steps. Primary metric: winrate vs
    held-out strong opponents **never present in either training pool**
    (improved_prob_main + 1–2 strong externals, ≥50 games each per the
    benchmark-before-submitting rule), plus anchored-pool Glicko. Any
    "better" claim goes through the leaderboard-check skill.
  - Single seed per arm initially — treat as provisional; note it.
  - Run the fallback diagnostic on both arms' checkpoints before comparing.
- **Pre-registered decision:** adopt strong-only pool if held-out-strong
  winrate improves by ≥5 pts at equal steps; drop if within the binomial CI of
  the baseline; if pooled winrate falls but held-out-strong is flat, that is a
  *measurement* lesson (pooled winrate was the illusion), not an adoption.
- **Cost estimate:** Step 0 ≈ small trainer patch; Step 1 free (reads the
  baseline run we need anyway); Step 2 = one extra Phase-3 training run —
  only paid if Step 1 justifies it.
- **Prior work checked:** exploiter diagnostic + exploit-transfer results
  (this repo, notes/exploiter_experiment_state.md and memory); ByteRL/OSFP —
  the smoothing weights over the policy pool are part of the algorithm, and
  double-oracle weights the pool by the meta-game solution, not uniformly, so
  "pool distribution matters" is the paper's own position; anchored eval pool
  rho +0.929 (eval pool is NOT the thing under test — see below).

**Explicitly out of scope:** removing weak agents from the *local evaluation
pool*. The eval pool's job is prediction of the ladder (rho +0.929, three
local-vs-ladder inversions behind that rule) and regression detection (the
exploiter ckpt was caught because rule_baseline was in the pool). Weak eval
opponents are harmless as long as selection reads Glicko / per-opponent
breakdowns, never a single pooled winrate.

## Result (fill after)
- **Observed:** not measured — Rami removed the public pool by decision on
  2026-08-04 before Step 1 ran (`--mix` added to train_ppo_puffer.py, new
  default `0.625,0.375,0`; old behavior reachable via `--mix 0.5,0.3,0.2`).
- **Decision:** adopted by fiat, not by experiment. Rationale: the public
  pool deviates from the (fictitious) self-play theory the phase is built on.
- **What we learned / still open:** whether the 20% public-pool share helped
  or hurt remains unanswered, and pure self-play now sees ZERO non-mirror
  decks in training — watch held-out external-deck winrate on the first
  no-pool run; if it degrades, the comparison arm is one flag away.
