#!/bin/zsh
# Rami's full-corpus IL experiment (notes/experiments/2026-08-04-il-full-
# corpus-ladder.md): fresh 3-epoch BC on ALL Hub train days, PRIOR recipe.
# Waits for the arm tournament so MPS is uncontended.
set -u
cd "$(dirname "$0")/.."
LOG=runs/stage2_restart.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
until grep -q "TOURNAMENT DONE" "$LOG" 2>/dev/null; do
  grep -q "CHAIN FAIL: tournament" "$LOG" 2>/dev/null && { log "IL-FULL ABORT: tournament failed"; exit 1; }
  sleep 30
done
log "STEP il_full BEGIN (fresh 3 epochs, ALL hub train days, seed 42)"
if nice -n 5 uv run python scripts/train_il.py \
    --data-source hub --num-workers 4 \
    --epochs 3 --lr 3e-4 --batch-size 64 --seed 42 \
    --eval-every-steps 25000 \
    --out models/il_alldays_0804 \
    --run-dir runs/il_full_0804 >>"$LOG" 2>&1; then
  log "STEP il_full OK -> models/il_alldays_0804"
else
  log "CHAIN FAIL: il_full crashed"; exit 1
fi
log "IL-FULL DONE -- ready for fallback diagnostic + head-to-head + bundle"
