#!/bin/zsh
# Follow-up to run_stage2_restart.sh (v2 §2.B2a): the TD critic failed the
# built-in MSE gate (0.326 vs const-0.5's 0.250 — under-propagated in one
# epoch), so train the registered MC ablation and audit it. Runs only after
# the main chain prints CHAIN DONE (E0 arms finished, MPS free).
set -u
cd "$(dirname "$0")/.."
LOG=runs/stage2_restart.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "CHAIN DONE" "$LOG" 2>/dev/null; do
  grep -q "CHAIN FAIL" "$LOG" 2>/dev/null && { log "MC FOLLOWUP ABORT: main chain failed"; exit 1; }
  sleep 30
done
log "STEP critic_mc BEGIN (mc ablation, 1 epoch)"
if nice -n 5 uv run python scripts/train_critic.py \
    --target mc --data-source hub --hub-days 2026-07-01,2026-07-26 \
    --num-workers 4 --epochs 1 --out models/critic_mc >>"$LOG" 2>&1; then
  log "STEP critic_mc OK"
else
  log "CHAIN FAIL: critic_mc crashed"; exit 1
fi
log "STEP audit_mc BEGIN"
if nice -n 5 uv run python scripts/audit_critic.py \
    --critic-dir models/critic_mc >>"$LOG" 2>&1; then
  log "STEP audit_mc OK"
else
  log "CHAIN FAIL: audit_mc crashed"; exit 1
fi
log "MC FOLLOWUP DONE -- both critic audits ready for the (iii) hand read"
