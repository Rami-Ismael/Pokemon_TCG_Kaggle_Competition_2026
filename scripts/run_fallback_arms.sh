#!/bin/zsh
# B2 fallback arm queue (v2 §2.B2, critic-blocked path): efb x3 seeds
# (outcome-weighted, the registered contingency), then E3' = efb x skill
# gate at the measured Q75=1189.0. Same log/markers as the main chain.
set -u
cd "$(dirname "$0")/.."
LOG=runs/stage2_restart.log
DAYS="2026-07-01,2026-07-26"
STEPS=13000
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
log "FALLBACK ARMS BEGIN (critic blocked -> efb + e3-skill)"
for SEED in 42 43 44; do
  log "STEP efb_s${SEED} BEGIN"
  nice -n 5 uv run python scripts/train_il.py \
      --data-source hub --hub-days "$DAYS" --num-workers 4 \
      --init-from models/il_agent --total-steps $STEPS --lr 1e-4 \
      --weight-arm outcome --beta 1.0 \
      --seed $SEED --out models/s2v2/efb_s${SEED} \
      --run-dir runs/s2v2_efb_s${SEED} >>"$LOG" 2>&1 \
    && log "STEP efb_s${SEED} OK" || { log "CHAIN FAIL: efb_s${SEED}"; exit 1; }
done
for SEED in 42 43 44; do
  log "STEP e3_s${SEED} BEGIN"
  nice -n 5 uv run python scripts/train_il.py \
      --data-source hub --hub-days "$DAYS" --num-workers 4 \
      --init-from models/il_agent --total-steps $STEPS --lr 1e-4 \
      --weight-arm outcome --beta 1.0 --skill-min-score 1189.0 \
      --seed $SEED --out models/s2v2/e3_s${SEED} \
      --run-dir runs/s2v2_e3_s${SEED} >>"$LOG" 2>&1 \
    && log "STEP e3_s${SEED} OK" || { log "CHAIN FAIL: e3_s${SEED}"; exit 1; }
done
log "FALLBACK ARMS DONE -- e0/efb/e3 x3 ready for the local tournament"
