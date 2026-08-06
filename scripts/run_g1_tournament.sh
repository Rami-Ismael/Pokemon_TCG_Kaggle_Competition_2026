#!/bin/zsh
# Gen-1 candidate tournament (v2 §3.3 gate leg 1 + §3.2b held-out transfer):
# both gen-1 candidates vs the three bases, the trained-on trio (in-sample
# diagnostic), FIVE never-trained-on publics (the transfer column), and the
# random floor. Isolated ratings.
set -u
cd "$(dirname "$0")/.."
LOG=runs/selfplay_g1.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "SELFPLAY G1 DONE" "$LOG" 2>/dev/null; do sleep 15; done
AGENTS="selfplay_g1_final,selfplay_g1_ref430k,il_alldays_0804,s2v2_e0_s43,il_agent,kiyotah_dragapult,mechi22_alakazam,plamen06_steel,tb_dragapult,tb_search,tb_alakazam,dedquoc_rule_engine,kojimar_lucario,random_legal"
log "STEP g1_tournament BEGIN (14 agents incl. 5 held-out publics)"
if nice -n 5 uv run python scripts/benchmark_agents.py \
    --agents "$AGENTS" --games 12 --no-glicko-persist \
    --out reports/selfplay_g1_tournament.json >>"$LOG" 2>&1; then
  log "STEP g1_tournament OK -> reports/selfplay_g1_tournament.json"
else
  log "CHAIN FAIL: g1 tournament crashed"
fi
log "G1 TOURNAMENT DONE"
