# Agent-pool comparison: 37-agent round-robin

**Run:** `uv run python scripts/benchmark_agents.py --agents <37 keys> --games 4`
**Date:** 2026-08-02 · **Wall clock:** 5295.7 s · **Pairings:** 703 · **Games:** ~5,900
**Raw:** `reports/agent_benchmark.json`, `reports/benchmark_full_pool_run.txt`,
ratings folded into `reports/glicko_ratings.json`
**Every agent played 296 games** (8 per cross-pairing, mirrored seats). Zero crashes.

## The headline: the trained model is not in this run

`il_agent` was excluded. `models/` is gitignored and empty in this checkout, and
`data/episodes/` is absent too, so there is no checkpoint to load and nothing to
retrain from. **This run therefore says nothing about the trained model.**

That exclusion was deliberate, because the alternative is worse than useless.
`agents/il_agent` honours the repo's never-crash contract: a missing `MODEL_DIR`
does not raise, it silently routes every decision to `_safe_choice`
(first-legal-option). Measured directly: **383 of 383 decisions were fallback**,
and the agent still posted a 66.7% win rate against `random_legal` — a number
that would have been written into the win matrix, `agent_benchmark.json`, and
the *compounding* `glicko_ratings.json` under the name `il_agent`. An untrained
policy's results wearing a trained model's name, with nothing in the artifacts
to reveal it.

`agents/il_agent/agent_core.py` now implements the `diag_reset`/`diag_snapshot`
contract (`weights_loaded`, `model_dir`, `ml_available`, per-path fallback
counts). `run_benchmark` checks it before playing a single game, prints a banner
naming any unweighted agent, and records them under `unweighted_agents` in the
result JSON. In this run that field is `[]` — every agent listed below ran its
real policy.

**To get the comparison actually asked for:** put a checkpoint in
`models/il_agent/` (or point `IL_MODEL_DIR` at one) and re-run with `il_agent`
added. The guard now makes it impossible to confuse a weightless run for a real one.

## Standings

Glicko RD is 30.0 for every agent — the implementation's floor, reached because
each played 296 games. Ratings closer than ±60 are not distinguishable.

| # | agent | win% | 95% CI | Glicko | GXE | src |
|--:|---|--:|---|--:|--:|---|
| 1 | pllinas_alakazam | 82.8 | [78.1, 86.6] | 1856.2 | 79.7 | pool |
| 2 | plamen06_steel | 74.7 | [69.4, 79.3] | 1770.0 | 73.8 | pool |
| 3 | mechi22_alakazam | 68.9 | [63.4, 73.9] | 1679.7 | 66.6 | pool |
| 4 | **improved_prob_main** | 67.9 | [62.4, 73.0] | 1680.3 | 66.7 | **ours** |
| 5 | **agent_core_improved** | 67.6 | [62.0, 72.6] | 1688.8 | 67.4 | **ours** |
| 6 | **proto** | 67.2 | [61.7, 72.3] | 1686.8 | 67.2 | **ours** |
| 7 | makthanithin_improved_prob | 67.2 | [61.7, 72.3] | 1662.6 | 65.1 | pool |
| 8 | tb_archaludon | 67.2 | [61.7, 72.3] | 1684.8 | 67.0 | pool |
| 9 | pixiux_lucario_v63 | 66.9 | [61.3, 72.0] | 1703.1 | 68.6 | pool |
| 10 | daniilkrasnovvv_conservative_prob | 66.6 | [61.0, 71.7] | 1688.8 | 67.4 | pool |
| 11 | romanrozen_strong_start | 64.9 | [59.3, 70.1] | 1668.9 | 65.7 | pool |
| 12 | kojimar_lucario | 64.5 | [58.9, 69.8] | 1666.8 | 65.5 | pool |
| 13 | makthanithin_1084_baseline | 64.5 | [58.9, 69.8] | 1661.8 | 65.1 | pool |
| 14 | biohack44_day2 | 62.2 | [56.5, 67.5] | 1626.2 | 61.9 | pool |
| 15 | prvsiyan_control_v11 | 61.5 | [55.8, 66.8] | 1627.5 | 62.0 | pool |
| 16 | tb_iono | 60.1 | [54.5, 65.5] | 1618.8 | 61.2 | pool |
| 17 | kiyotah_dragapult | 58.4 | [52.8, 63.9] | 1626.0 | 61.9 | pool |
| 18 | ryotasueyoshi_alakazam | 58.4 | [52.8, 63.9] | 1601.8 | 59.7 | pool |
| 19 | tb_dragapult | 58.4 | [52.8, 63.9] | 1605.8 | 60.0 | pool |
| 20 | tb_alakazam | 56.4 | [50.7, 61.9] | 1577.4 | 57.4 | pool |
| 21 | kiyotah_iono | 55.4 | [49.7, 61.0] | 1546.2 | 54.4 | pool |
| 22 | **rule_baseline** | 55.1 | [49.4, 60.6] | 1562.7 | 56.0 | **ours** |
| 23 | tb_abomasnow | 55.1 | [49.4, 60.6] | 1570.1 | 56.7 | pool |
| 24 | jek1wantaufik_lucario | 52.4 | [46.7, 58.0] | 1558.3 | 55.6 | pool |
| 25 | kiyotah_abomasnow | 51.7 | [46.0, 57.3] | 1538.1 | 53.7 | pool |
| 26 | tb_rulecore | 44.6 | [39.0, 50.3] | 1464.2 | 46.6 | pool |
| 27 | tb_lucario | 40.2 | [34.8, 45.9] | 1424.0 | 42.8 | pool |
| 28 | tb_search | 31.8 | [26.7, 37.3] | 1343.7 | 35.4 | pool |
| 29 | makimakiai_rl | 20.3 | [16.1, 25.2] | 1219.5 | 25.4 | pool (untrained weights) |
| 30 | dedquoc_rule_engine | 19.9 | [15.8, 24.9] | 1150.3 | 20.7 | pool |
| 31 | **grunt** | 19.6 | [15.5, 24.5] | 1145.4 | 20.4 | **ours** |
| 32 | prvsiyan_grimbelief_alakazam | 19.6 | [15.5, 24.5] | 1219.3 | 25.4 | pool |
| 33 | avikdas567_heuristic | 19.6 | [15.5, 24.5] | 1208.1 | 24.6 | pool |
| 34 | prvsiyan_templates_alakazam | 15.9 | [12.2, 20.5] | 1183.5 | 22.9 | pool |
| 35 | tb_heuristic | 15.9 | [12.2, 20.5] | 1177.9 | 22.5 | pool |
| 36 | tb_starmie | 14.9 | [11.3, 19.4] | 1165.1 | 21.6 | pool |
| 37 | random_legal | 10.1 | [7.2, 14.1] | 984.2 | 12.1 | floor |

## Findings

**1. Two public agents are clearly ahead of everything we have.**
`pllinas_alakazam` (82.8%) and `plamen06_steel` (74.7%) sit above our best by
margins whose CIs do not overlap ours. Head-to-head, our four real agents win
0–12.5% of games against `pllinas_alakazam` (0/8 for `rule_baseline` and
`grunt`; 1/8 for the other three). This is the first evidence in the repo of a
public agent that beats us decisively, and it is the obvious study target.

**2. Our three improved variants are statistically indistinguishable.**
`improved_prob_main` (67.9%), `agent_core_improved` (67.6%) and `proto` (67.2%)
have near-identical CIs and Glicko within 8 points of each other — far inside
the ±60 resolution. Treating any one as "the best of ours" is not supported.
They are also indistinguishable from four pool agents in the same band
(`mechi22_alakazam`, `makthanithin_improved_prob`, `tb_archaludon`,
`pixiux_lucario_v63`, `daniilkrasnovvv_conservative_prob`).

**3. The gain over `rule_baseline` is the heuristic, not search.** The improved
variants beat `rule_baseline` (55.1%) by ~13pp with non-overlapping CIs. That
gain is *not* attributable to the UCB1 re-ranker: `agent_core_improved` reported
`fallback_rate=100%` over 18,576 decisions, because `USE_SEARCH` is off by
default — a documented decision (`agent_core_improved.py:737`) after search was
benchmarked and lost. This run is the heuristic head alone, and it confirms the
heuristic is what carries the improvement. `improved_prob_main` ran its search on
58.4% of decisions and finished in a statistical tie with the search-free variant.

**4. `grunt` is ours and it is near the floor.** 19.6% overall, Glicko 1145 vs
the `random_legal` floor at 984 — and 0/8 against both top agents. It sits below
21 pool agents. Deck-agnostic greedy one-ply is not competitive here.

**5. `tb_*` mu does not transfer to this pool.** TomBombadyl's ladder ordering is
roughly inverted locally. His leader `tb_archaludon` (mu 1196.1) does rank top-8
here, but his best home-grown `tb_search` (mu 660.5) lands 28th at 31.8%, below
`tb_rulecore` (mu 535.6) and `tb_lucario`. Public mu and this pool measure
different fields; do not use his mu to rank opponents here.

## The deck confound — read before quoting any single number

`benchmark_agents.py` has no deck axis. `load_agent` fills `my_deck` from each
agent's own sibling `deck.csv`, so **deck is welded to agent identity** and every
cell above is a *policy + deck* result. `scripts/deck_fingerprints.py` groups the
pool: **37 agents, 21 distinct decks.**

So "we beat X" usually means "our policy on our deck beats their policy on their
deck" — which attributes the result to neither. The exception is a same-deck
group, where the deck cancels:

**Deck `76812200` — 7 agents, identical 60 cards (this is `il_agent`'s deck):**

| agent | win% | 95% CI | Glicko |
|---|--:|---|--:|
| makthanithin_improved_prob | 67.2 | [61.7, 72.3] | 1662.6 |
| daniilkrasnovvv_conservative_prob | 66.6 | [61.0, 71.7] | 1688.8 |
| romanrozen_strong_start | 64.9 | [59.3, 70.1] | 1668.9 |
| makthanithin_1084_baseline | 64.5 | [58.9, 69.8] | 1661.8 |
| kojimar_lucario | 64.5 | [58.9, 69.8] | 1666.8 |
| **grunt** (ours) | 19.6 | [15.5, 24.5] | 1145.4 |
| random_legal (floor) | 10.1 | [7.2, 14.1] | 984.2 |

The top five are a five-way statistical tie: five different policies on one deck
land within 3pp. **On this deck, policy choice is worth ~3pp among competent
agents, while `pllinas_alakazam` on a different deck is 18pp clear of all of
them.** That is a strong hint the archetype gap dominates the policy gap — and it
is the ordering the `deck-selection` skill warns about.

This is also the group `il_agent` belongs to. When a checkpoint exists, those
five are its clean comparison; the other 30 agents are confounded.

## Caveats

- **No common random seeds.** `cabt` exposes no RNG control from Python, so deck
  shuffles and opening hands are not held constant. All CIs are binomial only and
  understate run-to-run variance.
- **Single run, one arm per agent.** The repo's standing rule is ≥3 seeds; this
  is one. Treat sub-5pp differences as unresolved regardless of CI width.
- **`makimakiai_rl` runs the notebook's *default* (untrained) weights** — its
  learned bake was never reproduced. Its 29th place is not evidence about the
  method.
- **`improved_prob_main` and `proto` ship no `deck.csv`**; their decklists are
  module literals, so they are absent from the fingerprint grouping and their
  deck confound is unquantified here.
- The evaluator envelope (~1.6 vCPU, ~197.7 MiB) is not enforced by this harness.
  Nothing here bounds any agent's ladder latency or memory.
