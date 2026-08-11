#!/bin/zsh
# Resume of run_bc_alldays52_pool_benchmark.sh after star 1 crashed loading
# grid_medium_comb (models/il_agent_medium_combined is an EMPTY dir; PR #56's
# strict-load raise turned its old silent fallback into a crash). Preflight
# found exactly two unloadable pool members; neither checkpoint exists in any
# worktree or in the Rami/ptcg-s2v2-arms HF backup, so they are EXCLUDED and
# the exclusion is recorded everywhere this result is quoted:
#   grid_medium_comb        (checkpoint never existed on this machine)
#   il_alldays_equalsteps   (worktree deleted; symlink dead)
# Bed = 91 agents (90 opponents per star). Wiring + fallback diagnostic
# already done by the first script (best ckpt step 620k, acc 0.761875; diag
# clean 0/392).
set -u
cd "$(dirname "$0")/.."
LOG=runs/bc_alldays52_pool_benchmark.log
CARD=notes/experiments/2026-08-10-bc-complete-53day-corpus.md
GAMES="${GAMES:-4}"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

AGENTS=$(uv run python -c "
import sys; sys.path.insert(0,'scripts')
import benchmark_agents as B
bad = {'grid_medium_comb','il_alldays_equalsteps'}
print(','.join(a for a in B.agent_groups()['all'] if a not in bad))")
n=$(echo "$AGENTS" | tr ',' '\n' | wc -l | tr -d ' ')
log "RESUME: star benchmarks on $n-agent bed (excluded: grid_medium_comb, il_alldays_equalsteps -- checkpoints lost, not restorable)"

log "star 1/2: bc_alldays52_jun16_aug07_seed42 vs ALL(-2)"
PTCG_DEVICE=cpu nice -n 10 uv run python scripts/benchmark_agents.py \
  --agents "$AGENTS" --focus bc_alldays52_jun16_aug07_seed42 --games "$GAMES" \
  --out reports/pool_star_bc_alldays52.json >> "$LOG" 2>&1 \
  || { log "ABORT: star 1 crashed"; exit 1; }
log "star 2/2: il_agent_v3_best vs ALL(-2) (paired baseline)"
PTCG_DEVICE=cpu nice -n 10 uv run python scripts/benchmark_agents.py \
  --agents "$AGENTS" --focus il_agent_v3_best --games "$GAMES" \
  --out reports/pool_star_il_agent_v3_best.json >> "$LOG" 2>&1 \
  || { log "ABORT: star 2 crashed"; exit 1; }

log "POOL BENCHMARK DONE"
{
  echo ""
  echo "### Raw pool-benchmark record (auto-appended $(date '+%Y-%m-%d %H:%M'))"
  echo '- wired checkpoint: best step 620000, eval acc 0.761875 (final was 0.760625)'
  echo '- fallback diagnostic: clean, 0/392 fallbacks, 0 illegal (reports/fallback_diag_bc_alldays52.json)'
  echo "- bed: $n agents; EXCLUDED grid_medium_comb + il_alldays_equalsteps (checkpoints lost -- searched worktrees + HF backup, 0 hits). Any number quoted from this run carries that caveat."
  echo '- star results: reports/pool_star_bc_alldays52.json vs reports/pool_star_il_agent_v3_best.json'
  echo '  (full log: runs/bc_alldays52_pool_benchmark.log)'
} >> "$CARD"
