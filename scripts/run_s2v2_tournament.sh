#!/bin/zsh
# Local tournament over the restarted Stage-2 arms (v2 §2.B2 selection).
# Waits for the fallback arm queue, then round-robins: PRIOR + e0/efb/e3 x3
# seeds + the strong public trio + the random floor. 12 mirrored pairs per
# pairing (matches the v1-era arm tournament), Glicko NOT persisted (isolated
# run: this worktree's ratings file is stale vs main's and arm selection only
# needs within-run separation).
set -u
cd "$(dirname "$0")/.."
LOG=runs/stage2_restart.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "FALLBACK ARMS DONE" "$LOG" 2>/dev/null; do
  grep -q "CHAIN FAIL" "$LOG" 2>/dev/null && { log "TOURNAMENT ABORT: arm queue failed"; exit 1; }
  sleep 30
done
AGENTS="il_agent,s2v2_e0_s42,s2v2_e0_s43,s2v2_e0_s44,s2v2_efb_s42,s2v2_efb_s43,s2v2_efb_s44,s2v2_e3_s42,s2v2_e3_s43,s2v2_e3_s44,kiyotah_dragapult,mechi22_alakazam,plamen06_steel,random_legal"
log "STEP tournament BEGIN (14 agents, 12 mirrored pairs per pairing, isolated glicko)"
if nice -n 5 uv run python scripts/benchmark_agents.py \
    --agents "$AGENTS" --games 12 --no-glicko-persist \
    --out reports/s2v2_tournament.json >>"$LOG" 2>&1; then
  log "STEP tournament OK -> reports/s2v2_tournament.json"
else
  log "CHAIN FAIL: tournament crashed"; exit 1
fi
log "TOURNAMENT DONE"
