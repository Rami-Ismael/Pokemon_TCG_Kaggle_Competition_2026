# Critic audit — critic_mc

Eval rows: 820088 (4430 episodes, held-out day 2026-07-27)

## (i) Outcome prediction vs no-skill baselines
(V clamped to [0,1] — the quantity advantage_weights consumes; raw out-of-range rate 22.20%)
- MSE 0.2679 vs constant-0.5 0.2499 -> FAIL
- accuracy@0.5 0.6496 vs majority base rate 0.5379 (819870 decisive rows) -> PASS

## (ii) Advantage distribution (Â = outcome − V)
- mean -0.0269  σ 0.5169
- quantiles p1 -1.000 | p25 -0.214 | p50 +0.000 | p75 +0.168 | p99 +0.971
- verdict: PASS (centered-at-0 and sane-spread check)
- figure: /Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026/.claude/worktrees/rewrite-kaggle-pokemon-tcg-prompt-c69997/reports/figures/critic_advantage_hist_critic_mc.png

## (iii) Hand-read: top 10 positive and negative |Â| decisions
Human judgment required — the SIGN of each Â below must be defensible.

### Most positive advantages (critic says: better than expected)
episode 88528789 seat 1 turn 12 | outcome 1.0  V(s) 0.000  Â +1.000
  select MAIN ctx 0 min 1 max 1 | 15 options
  my prizes left 3 deck 6 hand 20 | opp prizes 3 deck 35 hand 1
  CHOSE [6] (PLAY index=11)
  options: [0](PLAY index=0) [1](PLAY index=2) [2](PLAY index=6) [3](PLAY index=7) [4](PLAY index=8) [5](PLAY index=9) [6](PLAY index=11) [7](PLAY index=12) [8](PLAY index=15) [9](PLAY index=16) ... (+5 more)

episode 88528789 seat 1 turn 12 | outcome 1.0  V(s) 0.000  Â +1.000
  select MAIN ctx 0 min 1 max 1 | 10 options
  my prizes left 3 deck 6 hand 19 | opp prizes 3 deck 35 hand 1
  CHOSE [7] (ATTACK attack=1072)
  options: [0](PLAY index=0) [1](PLAY index=2) [2](PLAY index=9) [3](PLAY index=11) [4](PLAY index=15) [5](ABILITY area=5 index=0) [6](ABILITY area=5 index=2) [7](ATTACK attack=1072) [8](RETREAT) [9](END)

episode 88528800 seat 1 turn 7 | outcome 1.0  V(s) 0.000  Â +1.000
  select CARD ctx 4 min 1 max 1 | 1 options
  my prizes left 6 deck 40 hand 2 | opp prizes 3 deck 23 hand 7
  CHOSE [0] (CARD area=5 index=0 playerIndex=1)
  options: [0](CARD area=5 index=0 playerIndex=1)

episode 88528800 seat 1 turn 6 | outcome 1.0  V(s) 0.000  Â +1.000
  select MAIN ctx 0 min 1 max 1 | 3 options
  my prizes left 6 deck 40 hand 2 | opp prizes 4 deck 28 hand 7
  CHOSE [2] (END)
  options: [0](ATTACH area=2 index=0 inPlayArea=4 inPlayIndex=0) [1](ATTACH area=2 index=0 inPlayArea=5 inPlayIndex=0) [2](END)

### Most negative advantages (critic says: worse than expected)
episode 88528792 seat 0 turn 7 | outcome 0.0  V(s) 1.000  Â -1.000
  select MAIN ctx 0 min 1 max 1 | 8 options
  my prizes left 5 deck 32 hand 3 | opp prizes 4 deck 22 hand 8
  CHOSE [0] (PLAY index=0)
  options: [0](PLAY index=0) [1](PLAY index=1) [2](ATTACH area=2 index=2 inPlayArea=4 inPlayIndex=0) [3](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=0) [4](ATTACH area=2 index=2 inPlayArea=5 inPlayIndex=1) [5](ATTACK attack=937) [6](RETREAT) [7](END)

episode 88528795 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 5 min 0 max 2 | 3 options
  my prizes left 6 deck 43 hand 6 | opp prizes 6 deck 42 hand 6
  CHOSE [0] (CARD area=1 index=12 playerIndex=0)
  options: [0](CARD area=1 index=12 playerIndex=0) [1](CARD area=1 index=18 playerIndex=0) [2](CARD area=1 index=27 playerIndex=0)

episode 88528792 seat 0 turn 5 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 15 min 1 max 1 | 5 options
  my prizes left 6 deck 35 hand 3 | opp prizes 6 deck 26 hand 6
  CHOSE [4] (CARD area=5 index=4 playerIndex=1)
  options: [0](CARD area=5 index=0 playerIndex=1) [1](CARD area=5 index=1 playerIndex=1) [2](CARD area=5 index=2 playerIndex=1) [3](CARD area=5 index=3 playerIndex=1) [4](CARD area=5 index=4 playerIndex=1)

episode 88528795 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select MAIN ctx 0 min 1 max 1 | 8 options
  my prizes left 6 deck 44 hand 7 | opp prizes 6 deck 42 hand 6
  CHOSE [0] (PLAY index=0)
  options: [0](PLAY index=0) [1](ATTACH area=2 index=4 inPlayArea=4 inPlayIndex=0) [2](ATTACH area=2 index=4 inPlayArea=5 inPlayIndex=0) [3](ATTACH area=2 index=5 inPlayArea=4 inPlayIndex=0) [4](ATTACH area=2 index=5 inPlayArea=5 inPlayIndex=0) [5](PLAY index=6) [6](RETREAT) [7](END)

episode 88528800 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 7 min 1 max 1 | 6 options
  my prizes left 6 deck 38 hand 1 | opp prizes 6 deck 42 hand 1
  CHOSE [0] (CARD area=6 index=0 playerIndex=0)
  options: [0](CARD area=6 index=0 playerIndex=0) [1](CARD area=6 index=1 playerIndex=0) [2](CARD area=6 index=2 playerIndex=0) [3](CARD area=6 index=3 playerIndex=0) [4](CARD area=6 index=4 playerIndex=0) [5](CARD area=6 index=5 playerIndex=0)

episode 88528795 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 5 min 0 max 2 | 3 options
  my prizes left 6 deck 43 hand 6 | opp prizes 6 deck 42 hand 6
  CHOSE [1] (CARD area=1 index=18 playerIndex=0)
  options: [0](CARD area=1 index=12 playerIndex=0) [1](CARD area=1 index=18 playerIndex=0) [2](CARD area=1 index=27 playerIndex=0)

episode 88528800 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 15 min 1 max 1 | 3 options
  my prizes left 6 deck 38 hand 1 | opp prizes 6 deck 42 hand 1
  CHOSE [0] (CARD area=5 index=0 playerIndex=1)
  options: [0](CARD area=5 index=0 playerIndex=1) [1](CARD area=5 index=1 playerIndex=1) [2](CARD area=5 index=2 playerIndex=1)

episode 88528800 seat 0 turn 5 | outcome 0.0  V(s) 1.000  Â -1.000
  select MAIN ctx 0 min 1 max 1 | 5 options
  my prizes left 5 deck 29 hand 7 | opp prizes 6 deck 41 hand 1
  CHOSE [1] (EVOLVE area=2 index=6 inPlayArea=5 inPlayIndex=0)
  options: [0](PLAY index=1) [1](EVOLVE area=2 index=6 inPlayArea=5 inPlayIndex=0) [2](ATTACK attack=937) [3](RETREAT) [4](END)

episode 88528792 seat 0 turn 7 | outcome 0.0  V(s) 1.000  Â -1.000
  select MAIN ctx 0 min 1 max 1 | 9 options
  my prizes left 5 deck 34 hand 4 | opp prizes 4 deck 22 hand 8
  CHOSE [1] (EVOLVE area=2 index=1 inPlayArea=4 inPlayIndex=0)
  options: [0](PLAY index=0) [1](EVOLVE area=2 index=1 inPlayArea=4 inPlayIndex=0) [2](PLAY index=2) [3](ATTACH area=2 index=3 inPlayArea=4 inPlayIndex=0) [4](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=0) [5](ATTACH area=2 index=3 inPlayArea=5 inPlayIndex=1) [6](ATTACK attack=936) [7](RETREAT) [8](END)

episode 88528795 seat 0 turn 3 | outcome 0.0  V(s) 1.000  Â -1.000
  select CARD ctx 7 min 0 max 1 | 26 options
  my prizes left 6 deck 44 hand 6 | opp prizes 6 deck 42 hand 6
  CHOSE [5] (CARD area=1 index=9 playerIndex=0)
  options: [0](CARD area=1 index=1 playerIndex=0) [1](CARD area=1 index=3 playerIndex=0) [2](CARD area=1 index=4 playerIndex=0) [3](CARD area=1 index=5 playerIndex=0) [4](CARD area=1 index=6 playerIndex=0) [5](CARD area=1 index=9 playerIndex=0) [6](CARD area=1 index=11 playerIndex=0) [7](CARD area=1 index=12 playerIndex=0) [8](CARD area=1 index=14 playerIndex=0) [9](CARD area=1 index=16 playerIndex=0) ... (+16 more)

---

## Hand-read verdict (part iii) — recorded 2026-08-04 ~09:50

**Overall: FAIL — advantage arms BLOCKED per the registered B2a rule.**
(i) fails on calibration even after the consumption clamp (clamped MSE
0.2679 vs constant's 0.2499; raw out-of-range rate 22.2% — a fifth of all
rows, not a tail) while passing accuracy decisively (65.0% vs 53.8%).
(ii) passes. (iii): the extreme-Â rows are magnitude artifacts — e.g.
V=0.000 on a turn-12 prizes-3-3 position whose only real danger is a
6-cards-left deck-out risk; the direction is arguable, the certainty is
not, and E2 weights consume the magnitude.

Finding worth keeping: BOTH critic trainings (TD and MC) beat the accuracy
base rate but fail calibration on the held-out day. A 3.32M-param critic on
9,820 episodes ranks positions usefully but cannot calibrate — consistent
with the corpus being ~50x smaller than the Metamon paper's; their critic-
based objectives had two orders of magnitude more data to calibrate on.
Decision: E-fallback (outcome-weighted) arms run now; E3 runs as
efb-weight x skill-gate (the skill filter is orthogonal to the critic and
stays registered); E4 dies with its gate.
