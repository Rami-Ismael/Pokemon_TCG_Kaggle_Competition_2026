# A0-family pool benchmark: search + all-days BC prior vs the anchored pool

**Date:** 2026-08-06 (~02:45–03:26 local) · **Runtime:** 2463 s ·
**Harness:** `scripts/benchmark_agents.py --games 4` (8 games per unordered pair),
fresh Glicko file `reports/glicko_search_prior_pool.json` (everyone starts 1500 ± 350),
raw matrix in `reports/agent_benchmark_search_prior_pool.json`,
full log `reports/benchmark_search_prior_pool.log`.

## Question

Does `agent_core_improved`'s flat-UCB1 search, with the PUCT-style exploration
bonus served by our strongest BC clone, beat the search-off heuristic against
the ladder-anchored local pool?

## Arms

| Arm | Config |
|---|---|
| `search_prior_alldays` (new, `agents/search_arms/prior_alldays_lucario/`) | HEAD `agent_core_improved.py`, `USE_SEARCH=True`, `USE_BC_PRIOR=True` (BC_PRIOR_C=0.75), prior checkpoint `models/il_alldays_0804` (127,748 steps, 3 ep full hub corpus, eval_acc 0.7583 — byte-identical sha256 to `il_agent_full_0804`, the exploit-robust model). Mega Lucario ex deck. |
| `agent_core_improved` (control) | Same HEAD source, defaults (`USE_SEARCH=0`) = pure AdvancedPolicy heuristic, the 804.0-lineage behavior. Same deck. |

Prior liveness was verified pre-run: 64/64 `candidate_probs` calls non-null in a
smoke game; the control instance in the same process stayed search-off.

## Glicko-1 (this run only, single rating period)

| Agent | Rating | RD | GXE |
|---|---:|---:|---:|
| romanrozen_strong_start | 1723.9 | 60.3 | 70.2% |
| **agent_core_improved (control)** | **1696.0** | 60.3 | 67.9% |
| makthanithin_1084_baseline | 1682.0 | 60.3 | 66.7% |
| wmh_alakazam | 1654.0 | 60.3 | 64.3% |
| tb_archaludon | 1640.0 | 60.3 | 63.1% |
| tb_dragapult | 1612.0 | 60.3 | 60.5% |
| wmh_garchomp | 1598.0 | 60.3 | 59.2% |
| tb_heuristic | 1206.1 | 60.3 | 24.5% |
| **search_prior_alldays (arm)** | **1122.1** | 60.3 | 19.1% |
| random_legal | 1066.1 | 60.3 | 16.0% |

Gap: **573.9 Glicko points**, non-overlapping 95% intervals by ~4.8×. The arm
lands below `tb_heuristic` (ladder anchor 633.0) and is statistically separated
from `random_legal` only weakly (56 points < the ±120.6 intervals).

## Win rates (Wilson 95% CI)

- Arm overall: **9/72 = 12.5% [6.7, 22.1]** — only wins: 6-2 vs random_legal,
  1-7 vs tb_dragapult / wmh_garchomp / tb_heuristic; 0-8 vs everything else.
- Control overall: **50/72 = 69.4% [58.0, 78.8]**.
- Head-to-head, deck and code held fixed: **control beat the arm 8-0
  [67.6, 100.0]** — turning search+prior ON costs the head-to-head outright.

## Reading

1. Consistent with every prior measurement of this search
   (35% overall 2026-08-01 comment in `agent_core_improved.py`; 0-16 pooled in
   `reports/ablation_a0.json`): the one-turn `evaluate_state` rollout scorer
   overrides the domain-tuned heuristic and loses the trade. The stronger prior
   checkpoint does not rescue it — the prior only biases *exploration order*;
   values still decide, and the values are the broken part.
2. The prior's embeddings for the piloted deck are untrained (Riolu / Mega
   Lucario ex: 0/800 BC-corpus perspectives), so the PUCT bonus is at best
   noise here. This run does **not** rule out the prior helping on a deck the
   checkpoint saw (the A0'' Alakazam variant), but even there search_prior went
   1-9 (`reports/ablation_a0_alakazam.json`).
3. Caveats: machine was under load (~14 loadavg, 8 nice'd PPO workers) —
   benchmark ran un-niced for scheduling priority, but the 1.5 s wall-clock
   search budget still buys fewer simulations than on a quiet machine; direction
   of that bias is against the arm, magnitude unknown. Pool flatters
   search agents with unbudgeted think time (romanrozen at #1 despite ladder
   ~950 vs tb_archaludon 1196 is the known distortion), which if anything
   should have *helped* the arm.
4. Ladder framing: local numbers are hypotheses about the ladder. Neither this
   arm nor this exact control build has a fresh ladder read; the control's
   lineage anchor is 804.0 (2026-08-01, spread ±100). Given the arm cannot even
   hold 50% against the *local* pool the control dominates, submitting it has
   no support.

## Decision

Keep `USE_SEARCH=0` as the shipped default. Search-based improvement should go
through `mcts_il_agent` (IL-prior MCTS, ladder 600.0), whose search uses the
IL policy for *both* proposal and evaluation, rather than this flat bandit's
`evaluate_state` scorer.
