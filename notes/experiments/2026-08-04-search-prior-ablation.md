# Does the BC prior earn its place inside search? (component ablation)

- **Where this came from:** Rami expected both `agent_core_improved` and
  `improved_prob_main` to use a neural net for action selection. Measured
  reality: `improved_prob_main` has **no ML at all** (imports are `math`, `os`,
  `random`, `time`, `collections`, `cg.api`), and `agent_core_improved` loads a
  BC model that it then **never consults** in the shipped config —
  `USE_BC_PRIOR` defaults to `1`, but the prior is only read *inside* the search
  routine, which returns at line 895 because `USE_SEARCH` defaults to `0`. The
  flag reads as "on" while the model sits idle.

- **Hypothesis:** the BC policy is a poor standalone player but a useful
  *search prior*. **Mechanism:** as a policy it must pick the single best
  action and gets punished for every mistake (it loses 50–0 to this same
  heuristic standalone); as a PUCT prior it only has to rank plausible moves
  highly enough to steer a fixed rollout budget toward them, and the engine
  rollouts correct its errors. **Therefore** search+prior should beat pure
  search, and a better-trained prior should beat a worse one.

- **Design & why this one:** component ablation (add-one-in), not a sweep — the
  question is attribution across two switches that interact, so the arms are
  chosen to isolate each contribution and their interaction.

- **Independent variables:** `USE_SEARCH` (off/on) and the prior
  (none / original checkpoint / all-days checkpoint).

- **Arms** — `agent_core_improved` is the agent under test throughout:

  | Arm | `USE_SEARCH` | prior | Role |
  |---|---|---|---|
  | `search_off_prior_original` | 0 | original | reference = the shipped default |
  | `search_off_prior_alldays` | 0 | all-days | **negative control** |
  | `search_on_no_prior` | 1 | none | pure UCB1, no neural net |
  | `search_on_prior_original` | 1 | original | the 804.0 configuration |
  | `search_on_prior_alldays` | 1 | all-days | the new question |

  **The negative control is the load-bearing arm.** With search off, the prior
  is unreachable code — so this arm *must* match the reference within noise. If
  it doesn't, the env-var plumbing is contaminating results and nothing else in
  the table can be trusted.

- **Metric & protocol:** win rate over **25 mirrored pairs = 50 games** per
  arm × opponent, seats alternated, Wilson 95% CIs. Opponents:
  `improved_prob_main` (heuristic + flat root Monte-Carlo, no ML — the strongest
  local agent) and `rule_baseline` (plain heuristic floor).
  `PTCG_DEVICE=cpu`, `PTCG_FALLBACK_TRACK=1`.

- **Implementation note that matters:** each arm runs in a **fresh subprocess**.
  `agent_core_improved` reads `USE_SEARCH`/`USE_BC_PRIOR` at import time and
  `bc_prior` caches the loaded model at module level, so toggling env vars
  in-process would silently reuse the first arm's configuration and produce a
  table of identical numbers that looks like a real (null) result.

- **Pre-registered decision:**
  - `search_on_no_prior` > `search_off` → **search itself pays**, independent of
    any neural net.
  - `search_on_prior_original` > `search_on_no_prior` → **the prior pays** on
    top of search; the BC model earns its place despite losing standalone.
  - `search_on_prior_alldays` > `search_on_prior_original` → **prior quality
    matters**, so improving the BC model improves the search agent. This is the
    result that would make the whole IL line worth continuing.
  - All search-on arms within noise of each other → the prior is decoration;
    ship pure UCB1 and stop spending compute on BC for this purpose.
  - Negative control diverges from reference → **invalid, stop and fix.**

- **Cost estimate:** ~3 s/game measured (the 1.5 s per-decision budget rarely
  binds — most decisions are not MAIN context). 5 arms × 2 opponents × 50 games
  ≈ 25–35 min, CPU only.

- **Prior work checked:** ledger 55162376 (804.0, "agent_core_improved + BC
  prior… USE_SEARCH=1/USE_BC_PRIOR=1 present but dead per
  notes/phase0_gate0_report.md"), `reports/pool_check_new_models.json`,
  `notes/experiments/2026-08-04-equal-steps-data-vs-training-length.md`.

- **Known limitation:** deck is fixed (Mega Lucario ex, as bundled), and the
  opponent panel is 2 agents — this measures the prior's value in that slice,
  not across the ladder's heterogeneity. Standing rule applies: nothing here
  becomes a "better" claim without a ladder read.

## Result (2026-08-04) — INCONCLUSIVE, UNDERPOWERED

| Arm | vs `improved_prob_main` | vs `rule_baseline` |
|---|---|---|
| `search_off_prior_original` (reference) | 0.460 [0.330, 0.596] | 0.580 [0.442, 0.706] |
| `search_off_prior_alldays` (**neg. control**) | 0.580 [0.442, 0.706] | 0.500 [0.366, 0.634] |
| `search_on_no_prior` | 0.520 [0.385, 0.652] | 0.540 [0.404, 0.670] |
| `search_on_prior_original` | 0.380 [0.259, 0.518] | 0.660 [0.522, 0.776] |
| `search_on_prior_alldays` | 0.460 [0.330, 0.596] | 0.680 [0.542, 0.792] |

**The negative control is the finding.** Rows 1 and 2 are provably the *same
configuration* — with `USE_SEARCH=0` the prior is unreachable code, so swapping
`IL_MODEL_DIR` cannot change behaviour. They differ by **12 points** vs
`improved_prob_main` and 8 vs `rule_baseline`. That is the measurement's noise
floor, measured directly rather than assumed.

- **Observed:** every arm's CI overlaps every other arm's. Worse, the prior's
  apparent effect flips sign by opponent — adding it *hurts* vs
  `improved_prob_main` (0.520 → 0.380) and *helps* vs `rule_baseline`
  (0.540 → 0.680), both well inside the 12-point noise floor. At 50 games only
  differences of roughly **28+ points** are resolvable here.

- **Decision: INCONCLUSIVE — do not adopt any reading.** Specifically, do NOT
  conclude "search+all-days prior is best" from its 0.680, which was the top
  number in the table and is pure noise. The pre-registered "all search-on arms
  within noise" branch technically fires, but with this noise floor the design
  cannot distinguish "the prior is decoration" from "the prior helps by 10
  points" — and 10 points would be well worth having. The honest state is
  **unmeasured**, not null.

- **Required redesign:** 50 games/arm is ~8× too small. Standard two-proportion
  power at 80%: **~392 games/arm** to resolve a 10-point difference, ~174 for
  15 points. Full table at 400 games/cell ≈ **3.3 h** CPU-only — cheap enough to
  run overnight, and the correct next step if this question matters.

- **What we learned (transferable):** a negative control is not bureaucracy —
  here it converted a table that *looked* like a result ("search + better prior
  wins, 0.680!") into a correct verdict of "underpowered, learn nothing." Two
  arms that are byte-identical in behaviour differed by 12 points; any claim
  smaller than that gap would have been noise dressed as signal. **Include an
  arm whose answer you already know, and size the experiment against the spread
  it reveals — not against the effect you hope to find.** Cheap and fast made it
  tempting to over-read; the control is what stopped that.

  Contrast with the same day's exploiter work, where effects were 50+ points and
  n=200 resolved them trivially. Sample size requirements scale with the effect
  you are chasing, and this effect is small.

- **Standing correctly, independent of this table:** `improved_prob_main` uses
  **no neural network at all**, and `agent_core_improved` in its shipped default
  loads the BC model and never queries it. Those are code facts, not
  measurements, and they are unaffected by the power problem.

## Follow-up: wide pool, 900 games per config — NOW POWERED

Rerun against 9 diverse opponents at 100 games each
(`scripts/pool_check_search_configs.sh` → `reports/pool_check_search_configs.json`).
Pooling the cells gives the sample size the 50-game ablation lacked.

| Opponent | heuristic only | search + all-days prior | Δ |
|---|---:|---:|---:|
| `random_legal` | 0.990 | 0.990 | +0.000 |
| `dedquoc_rule_engine` | 0.990 | 0.990 | +0.000 |
| `kiyotah_dragapult` | 0.700 | 0.660 | −0.040 |
| `il_agent` | 0.660 | 0.580 | −0.080 |
| `rule_baseline` | 0.620 | 0.630 | +0.010 |
| `improved_prob_main` | 0.480 | 0.510 | +0.030 |
| `makthanithin_improved_prob` | 0.430 | 0.460 | +0.030 |
| `ryotasueyoshi_alakazam` | 0.410 | 0.430 | +0.020 |
| `mechi22_alakazam` | 0.400 | 0.330 | −0.070 |
| **TOTAL (900 games each)** | **0.631** [0.599, 0.663] | **0.620** [0.588, 0.652] | **−0.011** |

**Difference: −1.1 points, 95% CI [−5.6, +3.4].**

- **Decision: turning search on with the all-days prior does NOT improve the
  agent.** Any true effect is bounded within roughly ±5 points, centred on zero.
  Unlike the 50-game table this is a real (null) result, not an absence of
  measurement — the aggregate CI excludes anything worth having. Per-opponent
  deltas range −0.08…+0.03 with no consistent sign, as expected when the
  underlying effect is ~0 and each cell carries ~±10 pts of noise.

- ⚠️ **What this does NOT isolate:** `USE_SEARCH=0→1` changes *two* things at
  once (search becomes live AND the prior becomes reachable). So this says the
  **package** doesn't pay; it cannot separate "search alone is worthless here"
  from "search helps but this prior hurts it by an equal amount". The
  `search_on_no_prior` arm at 900 games would settle that, and is the cheap next
  step if the question matters.

- **Bonus finding — where `agent_core_improved` actually stands.** The
  heuristic-only column is the first wide read on our strongest local agent:
  near-perfect vs the floor (`random_legal` 0.990, and `dedquoc_rule_engine`
  0.990 — that agent is *at* the random floor and is probably broken; worth
  checking before keeping it in the pool), comfortably ahead of our own
  `il_agent` (0.660), but **at or below even against four of the five external
  agents** (0.480, 0.430, 0.410, 0.400). That profile is far more consistent
  with a four-figure ladder rank than our local Glicko ordering is, and is
  probably the best local predictor we have.
