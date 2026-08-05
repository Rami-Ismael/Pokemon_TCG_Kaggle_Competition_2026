# Critic audit — critic_td

Eval rows: 820088 (4430 episodes, held-out day 2026-07-27)

## (i) Outcome prediction vs no-skill baselines
- MSE 0.3312 vs constant-0.5 0.2499 -> FAIL
- accuracy@0.5 0.5654 vs majority base rate 0.5379 (819870 decisive rows) -> PASS

## (ii) Advantage distribution (Â = outcome − V)
- mean +0.2471  σ 0.5198
- quantiles p1 -0.863 | p25 -0.060 | p50 +0.023 | p75 +0.775 | p99 +1.010
- verdict: FAIL (centered-at-0 and sane-spread check)
- figure: /Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026/.claude/worktrees/rewrite-kaggle-pokemon-tcg-prompt-c69997/reports/figures/critic_advantage_hist_critic_td.png

## (iii) Hand-read: top 10 positive and negative |Â| decisions
Human judgment required — the SIGN of each Â below must be defensible.

### Most positive advantages (critic says: better than expected)
episode 88474917 seat 1 turn 8 | outcome 1.0  V(s) -0.020  Â +1.020
  select CARD ctx 16 min 1 max 1 | 2 options
  my prizes left 4 deck 24 hand 3 | opp prizes 2 deck 23 hand 8
  CHOSE [1] (CARD area=5 index=2 playerIndex=1)
  options: [0](CARD area=5 index=1 playerIndex=1) [1](CARD area=5 index=2 playerIndex=1)

episode 88521797 seat 0 turn 11 | outcome 1.0  V(s) -0.020  Â +1.020
  select CARD ctx 22 min 0 max 3 | 3 options
  my prizes left 4 deck 21 hand 4 | opp prizes 1 deck 24 hand 6
  CHOSE [1] (CARD area=1 index=16 playerIndex=0)
  options: [0](CARD area=1 index=7 playerIndex=0) [1](CARD area=1 index=16 playerIndex=0) [2](CARD area=1 index=17 playerIndex=0)

episode 88368760 seat 1 turn 8 | outcome 1.0  V(s) -0.020  Â +1.020
  select CARD ctx 22 min 0 max 5 | 6 options
  my prizes left 5 deck 22 hand 7 | opp prizes 2 deck 31 hand 3
  CHOSE [6] DECLINE (empty pick)
  options: [0](CARD area=1 index=5 playerIndex=1) [1](CARD area=1 index=6 playerIndex=1) [2](CARD area=1 index=10 playerIndex=1) [3](CARD area=1 index=11 playerIndex=1) [4](CARD area=1 index=12 playerIndex=1) [5](CARD area=1 index=20 playerIndex=1)

episode 88354489 seat 1 turn 10 | outcome 1.0  V(s) -0.020  Â +1.020
  select CARD ctx 16 min 1 max 1 | 3 options
  my prizes left 4 deck 15 hand 7 | opp prizes 1 deck 25 hand 5
  CHOSE [0] (CARD area=4 index=0 playerIndex=1)
  options: [0](CARD area=4 index=0 playerIndex=1) [1](CARD area=5 index=1 playerIndex=1) [2](CARD area=5 index=2 playerIndex=1)

episode 88521797 seat 0 turn 11 | outcome 1.0  V(s) -0.019  Â +1.019
  select CARD ctx 22 min 0 max 3 | 3 options
  my prizes left 4 deck 21 hand 4 | opp prizes 1 deck 24 hand 6
  CHOSE [0] (CARD area=1 index=7 playerIndex=0)
  options: [0](CARD area=1 index=7 playerIndex=0) [1](CARD area=1 index=16 playerIndex=0) [2](CARD area=1 index=17 playerIndex=0)

episode 88368760 seat 1 turn 8 | outcome 1.0  V(s) -0.019  Â +1.019
  select CARD ctx 22 min 0 max 5 | 6 options
  my prizes left 5 deck 22 hand 7 | opp prizes 2 deck 31 hand 3
  CHOSE [4] (CARD area=1 index=12 playerIndex=1)
  options: [0](CARD area=1 index=5 playerIndex=1) [1](CARD area=1 index=6 playerIndex=1) [2](CARD area=1 index=10 playerIndex=1) [3](CARD area=1 index=11 playerIndex=1) [4](CARD area=1 index=12 playerIndex=1) [5](CARD area=1 index=20 playerIndex=1)

episode 88368760 seat 1 turn 8 | outcome 1.0  V(s) -0.019  Â +1.019
  select CARD ctx 22 min 0 max 5 | 6 options
  my prizes left 5 deck 22 hand 7 | opp prizes 2 deck 31 hand 3
  CHOSE [0] (CARD area=1 index=5 playerIndex=1)
  options: [0](CARD area=1 index=5 playerIndex=1) [1](CARD area=1 index=6 playerIndex=1) [2](CARD area=1 index=10 playerIndex=1) [3](CARD area=1 index=11 playerIndex=1) [4](CARD area=1 index=12 playerIndex=1) [5](CARD area=1 index=20 playerIndex=1)

episode 88521797 seat 0 turn 11 | outcome 1.0  V(s) -0.019  Â +1.019
  select CARD ctx 22 min 0 max 3 | 3 options
  my prizes left 4 deck 21 hand 4 | opp prizes 1 deck 24 hand 6
  CHOSE [3] DECLINE (empty pick)
  options: [0](CARD area=1 index=7 playerIndex=0) [1](CARD area=1 index=16 playerIndex=0) [2](CARD area=1 index=17 playerIndex=0)

episode 88416979 seat 0 turn 8 | outcome 1.0  V(s) -0.019  Â +1.019
  select CARD ctx 4 min 1 max 1 | 3 options
  my prizes left 6 deck 22 hand 14 | opp prizes 3 deck 34 hand 6
  CHOSE [2] (CARD area=5 index=2 playerIndex=0)
  options: [0](CARD area=5 index=0 playerIndex=0) [1](CARD area=5 index=1 playerIndex=0) [2](CARD area=5 index=2 playerIndex=0)

### Most negative advantages (critic says: worse than expected)
episode 88439704 seat 0 turn 13 | outcome 0.0  V(s) 1.026  Â -1.026
  select MAIN ctx 0 min 1 max 1 | 20 options
  my prizes left 5 deck 18 hand 7 | opp prizes 6 deck 10 hand 18
  CHOSE [8] (EVOLVE area=2 index=4 inPlayArea=4 inPlayIndex=0)
  options: [0](PLAY index=1) [1](PLAY index=2) [2](ATTACH area=2 index=3 inPlayArea=4 inPlayIndex=0) [3](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=0) [4](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=1) [5](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=2) [6](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=3) [7](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=4) [8](EVOLVE area=2 index=4 inPlayArea=4 inPlayIndex=0) [9](EVOLVE area=2 index=4 inPlayArea=5 inPlayIndex=4) ... (+10 more)

episode 88475289 seat 0 turn 13 | outcome 0.0  V(s) 1.025  Â -1.025
  select CARD ctx 7 min 0 max 1 | 6 options
  my prizes left 4 deck 20 hand 3 | opp prizes 5 deck 9 hand 15
  CHOSE [4] (CARD area=1 index=18 playerIndex=0)
  options: [0](CARD area=1 index=2 playerIndex=0) [1](CARD area=1 index=8 playerIndex=0) [2](CARD area=1 index=14 playerIndex=0) [3](CARD area=1 index=16 playerIndex=0) [4](CARD area=1 index=18 playerIndex=0) [5](CARD area=1 index=19 playerIndex=0)

episode 88475289 seat 0 turn 11 | outcome 0.0  V(s) 1.024  Â -1.024
  select MAIN ctx 0 min 1 max 1 | 10 options
  my prizes left 4 deck 26 hand 6 | opp prizes 5 deck 10 hand 15
  CHOSE [8] (ABILITY area=5 index=4)
  options: [0](PLAY index=0) [1](ATTACH area=2 index=1 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=1) [4](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=2) [5](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=3) [6](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=4) [7](PLAY index=5) [8](ABILITY area=5 index=4) [9](END)

episode 88475289 seat 0 turn 11 | outcome 0.0  V(s) 1.024  Â -1.024
  select MAIN ctx 0 min 1 max 1 | 9 options
  my prizes left 4 deck 26 hand 6 | opp prizes 5 deck 10 hand 15
  CHOSE [7] (PLAY index=5)
  options: [0](PLAY index=0) [1](ATTACH area=2 index=1 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=1) [4](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=2) [5](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=3) [6](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=4) [7](PLAY index=5) [8](END)

episode 88439704 seat 0 turn 13 | outcome 0.0  V(s) 1.024  Â -1.024
  select MAIN ctx 0 min 1 max 1 | 17 options
  my prizes left 5 deck 18 hand 6 | opp prizes 6 deck 10 hand 18
  CHOSE [15] (ABILITY area=4 index=0)
  options: [0](PLAY index=1) [1](PLAY index=2) [2](ATTACH area=2 index=3 inPlayArea=4 inPlayIndex=0) [3](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=0) [4](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=1) [5](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=2) [6](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=3) [7](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=4) [8](PLAY index=4) [9](ATTACH area=2 index=5 inPlayArea=4 inPlayIndex=0) ... (+7 more)

episode 88439704 seat 0 turn 11 | outcome 0.0  V(s) 1.024  Â -1.024
  select MAIN ctx 0 min 1 max 1 | 7 options
  my prizes left 5 deck 26 hand 4 | opp prizes 6 deck 12 hand 17
  CHOSE [0] (PLAY index=1)
  options: [0](PLAY index=1) [1](ATTACH area=2 index=2 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=1) [4](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=2) [5](PLAY index=3) [6](END)

episode 88475289 seat 0 turn 11 | outcome 0.0  V(s) 1.023  Â -1.023
  select MAIN ctx 0 min 1 max 1 | 12 options
  my prizes left 4 deck 26 hand 7 | opp prizes 5 deck 10 hand 15
  CHOSE [2] (EVOLVE area=2 index=0 inPlayArea=5 inPlayIndex=4)
  options: [0](EVOLVE area=2 index=0 inPlayArea=5 inPlayIndex=2) [1](EVOLVE area=2 index=0 inPlayArea=5 inPlayIndex=3) [2](EVOLVE area=2 index=0 inPlayArea=5 inPlayIndex=4) [3](PLAY index=1) [4](ATTACH area=2 index=2 inPlayArea=4 inPlayIndex=0) [5](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=0) [6](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=1) [7](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=2) [8](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=3) [9](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=4) ... (+2 more)

episode 88475289 seat 0 turn 11 | outcome 0.0  V(s) 1.023  Â -1.023
  select MAIN ctx 0 min 1 max 1 | 11 options
  my prizes left 4 deck 23 hand 7 | opp prizes 5 deck 10 hand 15
  CHOSE [9] (ABILITY area=7 index=0)
  options: [0](PLAY index=0) [1](ATTACH area=2 index=1 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=1) [4](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=2) [5](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=3) [6](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=4) [7](PLAY index=5) [8](PLAY index=6) [9](ABILITY area=7 index=0) ... (+1 more)

episode 88439704 seat 0 turn 13 | outcome 0.0  V(s) 1.023  Â -1.023
  select MAIN ctx 0 min 1 max 1 | 18 options
  my prizes left 5 deck 18 hand 6 | opp prizes 6 deck 10 hand 18
  CHOSE [3] (ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=0)
  options: [0](PLAY index=1) [1](PLAY index=2) [2](ATTACH area=2 index=3 inPlayArea=4 inPlayIndex=0) [3](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=0) [4](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=1) [5](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=2) [6](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=3) [7](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=4) [8](PLAY index=4) [9](ATTACH area=2 index=5 inPlayArea=4 inPlayIndex=0) ... (+8 more)

episode 88475289 seat 0 turn 11 | outcome 0.0  V(s) 1.023  Â -1.023
  select MAIN ctx 0 min 1 max 1 | 16 options
  my prizes left 4 deck 21 hand 9 | opp prizes 5 deck 10 hand 15
  CHOSE [1] (ATTACH area=2 index=1 inPlayArea=4 inPlayIndex=0)
  options: [0](PLAY index=0) [1](ATTACH area=2 index=1 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=1) [4](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=2) [5](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=3) [6](ATTACH area=2 index=1 inPlayArea=5 inPlayIndex=4) [7](PLAY index=5) [8](PLAY index=6) [9](ATTACH area=2 index=8 inPlayArea=4 inPlayIndex=0) ... (+6 more)

---

## Hand-read verdict (part iii) — recorded 2026-08-04 ~07:35

**FAIL.** The extreme positive advantages are critic artifacts, not
recognized good play: V(s) = −0.020 (outside the valid [0,1] outcome range)
recurs across unrelated mid-game states — a saturated floor for state
clusters that one epoch of TD(0) with ~27 hard target refreshes never
propagated terminal signal into (games run ~68 decisions per seat; value
information moves ~1 decision-step per refresh). The +0.247 mean advantage
(systematic under-prediction) and the fat +0.78/+1.01 p75/p99 tail are the
same under-propagation seen from the distribution side. The picked options
in the dumped rows are unremarkable CARD selects, indistinguishable from
their alternatives to a human reader — nothing about them justifies Â≈+1.

**Combined B2a verdict for critic_td: BLOCKED (i FAIL, ii FAIL, iii FAIL).**
Decision path: the registered MC ablation (direct outcome regression — no
propagation needed, targets bounded by construction) is queued; if it passes
all three parts, the advantage arms run with it and the TD-vs-MC
disagreement stands as a finding. If it also fails, the arms fall back to
the registered outcome-weighted E-fallback, loudly.
