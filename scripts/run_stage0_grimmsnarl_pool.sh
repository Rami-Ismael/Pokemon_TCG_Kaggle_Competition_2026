#!/bin/zsh
# Stage 0 of notes/experiments/2026-08-05-mcts-rl-deck-confound.md:
# rate the pool's 3 Grimmsnarl pilots against the rho=+0.929 anchored pool.
#
# LOAD GATE (not optional): three of these agents run live search with
# unbudgeted think time (romanrozen_strong_start, and both prvsiyan leaf-eval
# agents). Under CPU contention a search agent gets fewer simulations per
# decision, so its measured strength drops for reasons that have nothing to do
# with policy quality -- the pool would then misrate exactly the archetype
# Stage 0 is trying to measure. Wait for the box to be quiet.
set -u
cd "$(dirname "$0")/.."

NCPU=$(sysctl -n hw.ncpu)
GATE=$(( NCPU / 2 ))        # 1-min load average must fall below this
MAX_WAIT=$(( 6 * 3600 ))
t0=$(date +%s)

while :; do
  load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  [ "$load" -lt "$GATE" ] && break
  waited=$(( $(date +%s) - t0 ))
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "STAGE0 ABORT: load still ${load} (gate ${GATE}) after ${waited}s" >&2
    exit 2
  fi
  echo "[gate] load ${load} >= ${GATE}, waiting 300s (waited ${waited}s)"
  sleep 300
done

echo "[gate] load cleared at $(date). Launching Stage 0 tournament."

# 8-agent anchored pool (memory: anchored-pool-rho-0.93, rho +0.929 vs ladder)
# + the 3 pilots whose deck.csv resolves to Marnie's Grimmsnarl ex
# (reports/pool_archetype_census.json).
AGENTS="tb_archaludon,makthanithin_1084_baseline,romanrozen_strong_start,tb_dragapult,wmh_alakazam,wmh_garchomp,tb_heuristic,random_legal,prvsiyan_grimbelief_alakazam,prvsiyan_templates_alakazam,wmh_grimmsnarl"

nice -n 5 uv run python scripts/benchmark_agents.py \
  --agents "$AGENTS" \
  --games 10 \
  --glicko-path reports/glicko_stage0.json \
  --out reports/stage0_grimmsnarl_pool.json

echo "STAGE0 DONE rc=$? at $(date)"
