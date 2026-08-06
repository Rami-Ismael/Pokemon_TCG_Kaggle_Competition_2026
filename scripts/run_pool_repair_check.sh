#!/bin/zsh
# Did adding tetsutani_grimmsnarl repair the pool?
#
# The test: does the local pool now rank OUR OWN submissions in the same order
# Kaggle did? Every agent below has a ladder score we read ourselves from
# reports/submission_ledger.jsonl (the verified tier -- not the self-reported
# third-party anchors). The load-bearing pair is the same checkpoint on two
# decks: il_alldays_0804 (418.0) vs il_alldays_0804@marnies_grimmsnarl_ex
# (314.1). Before the repair the pool ranked those 41st and 1st of 52 -- exactly
# backwards. If tetsutani_grimmsnarl can beat the Grimmsnarl arm, that arm
# should fall and Spearman rho should recover.
#
# LOAD GATE: several pool members run live search with unbudgeted think time
# (romanrozen_strong_start, prvsiyan_*, tb_search). Under CPU oversubscription
# they get fewer simulations per decision and measure weak for reasons that have
# nothing to do with policy -- which would corrupt a RANKING measurement. A
# concurrent PPO run spawned ~10 workers and drove loadavg to 42 on 12 cores.
set -u
cd "$(dirname "$0")/.."

NCPU=$(sysctl -n hw.ncpu)
GATE=$(( NCPU / 2 ))
MAX_WAIT=$(( 8 * 3600 ))
t0=$(date +%s)
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  [ "$load" -lt "$GATE" ] && break
  waited=$(( $(date +%s) - t0 ))
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "POOL-REPAIR ABORT: load ${load} still >= ${GATE} after ${waited}s" >&2
    exit 2
  fi
  echo "[gate] load ${load} >= ${GATE}, waiting 300s (waited ${waited}s)"
  sleep 300
done
echo "[gate] cleared at $(date). Launching pool-repair check."

# Ladder-anchored agents of ours (all with verified ledger scores) + the new
# Grimmsnarl opponent + enough field to give the anchors real opposition.
AGENTS="il_alldays_0804,il_alldays_0804@marnies_grimmsnarl_ex,il_agent,mcts_il_agent,selfplay_g1_ref430k,s2v2_e0_s43,ppo_u120832,s2_e1_s43,improved_prob_main,agent_core_improved,tetsutani_grimmsnarl,wmh_grimmsnarl,prvsiyan_grimbelief_alakazam,tb_archaludon,romanrozen_strong_start,makthanithin_1084_baseline,wmh_alakazam,wmh_garchomp,random_legal"

nice -n 5 uv run python scripts/benchmark_agents.py \
  --agents "$AGENTS" --games 10 --no-glicko-persist --no-tb \
  --out reports/pool_repair_check.json
rc=$?
echo "benchmark rc=$rc"
[ $rc -eq 0 ] && uv run python scripts/pool_predictiveness.py \
  --result reports/pool_repair_check.json
echo "POOL-REPAIR DONE at $(date)"
