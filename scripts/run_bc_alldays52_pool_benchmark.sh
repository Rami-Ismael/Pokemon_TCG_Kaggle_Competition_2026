#!/bin/zsh
# Chained follow-up for the 2026-08-10 complete-corpus BC run (experiment card:
# notes/experiments/2026-08-10-bc-complete-53day-corpus.md). Rami's order:
# "when training is done benchmark it against the pool."
#
# Steps:
#   1. While train_il (pid $TRAIN_PID) runs: snapshot every is_best checkpoint
#      out of --out, because save_checkpoint() overwrites it at the final save
#      and v3's best (0.7528 @520k) beat its annealed final (0.7342) by 1.9pp.
#   2. On exit: pick best-vs-final by eval_accuracy, wire it into
#      agents/bc_alldays52_jun16_aug07_seed42/model/.
#   3. Fallback diagnostic, --assert-clean (verify the model really answers moves).
#   4. Wait for an idle machine (search-based pool members measure wrong under load).
#   5. Star benchmark vs the FULL pool (--agents all), then the SAME star for
#      il_agent_v3_best -- the paired baseline under an identical protocol.
set -u
cd "$(dirname "$0")/.."

TRAIN_PID=26481
OUT=models/bc_alldays52_jun16_aug07_seed42
BEST=models/bc_alldays52_jun16_aug07_seed42_best
AGENT_DIR=agents/bc_alldays52_jun16_aug07_seed42
CARD=notes/experiments/2026-08-10-bc-complete-53day-corpus.md
LOG=runs/bc_alldays52_pool_benchmark.log
GAMES="${GAMES:-4}"   # mirrored pairs per opponent = 8 games x ~92 opponents

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

meta() { # meta <dir> <field>
  python3 -c "import json,sys; print(json.load(open('$1/train_metadata.json'))[sys.argv[1]])" "$2" 2>/dev/null
}

log "watcher started (train pid $TRAIN_PID, games/opponent $((GAMES*2)))"

# -- 1. snapshot best checkpoints while training runs ------------------------
last_step=""
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if [ -f "$OUT/train_metadata.json" ]; then
    step=$(meta "$OUT" step)
    final=$(meta "$OUT" is_final_checkpoint)
    if [ -n "$step" ] && [ "$step" != "$last_step" ] && [ "$final" = "False" ]; then
      rm -rf "$BEST.tmp"; cp -a "$OUT" "$BEST.tmp" && { rm -rf "$BEST"; mv "$BEST.tmp" "$BEST"; }
      last_step="$step"
      log "snapshotted best checkpoint at step $step (acc $(meta "$BEST" eval_accuracy))"
    fi
  fi
  sleep 120
done
log "trainer exited"

# -- 2. verify + choose checkpoint -------------------------------------------
if [ ! -f "$OUT/model.safetensors" ] || [ "$(meta "$OUT" is_final_checkpoint)" != "True" ]; then
  log "ABORT: no final checkpoint in $OUT -- training did not finish cleanly"
  exit 1
fi
final_acc=$(meta "$OUT" eval_accuracy)
best_acc=$(meta "$BEST" eval_accuracy 2>/dev/null || echo 0)
pick="$OUT"; pick_label="final"
if [ -f "$BEST/model.safetensors" ] && \
   python3 -c "import sys; sys.exit(0 if float('$best_acc') > float('$final_acc') else 1)"; then
  pick="$BEST"; pick_label="best(step $(meta "$BEST" step))"
fi
log "checkpoints: final acc=$final_acc, best acc=$best_acc -> wiring $pick_label"
cp "$pick"/config.json "$pick"/model.safetensors "$pick"/train_metadata.json "$AGENT_DIR/model/"

# -- 3. fallback diagnostic ---------------------------------------
log "fallback diagnostic starting"
if PTCG_DEVICE=cpu nice -n 10 uv run python scripts/fallback_diagnostic.py \
    --agent bc_alldays52_jun16_aug07_seed42 --opponents rule_baseline \
    --pairs 3 --assert-clean \
    --out reports/fallback_diag_bc_alldays52.json >> "$LOG" 2>&1; then
  log "fallback diagnostic CLEAN"
else
  log "ABORT: fallback diagnostic NOT clean -- benchmark would measure _safe_choice, not the model"
  exit 1
fi

# -- 4. wait for an idle machine (same rationale as run_pool_repair_check.sh) ----------------
NCPU=$(sysctl -n hw.ncpu); GATE=$(( NCPU / 2 )); t0=$(date +%s)
while :; do
  load=$(sysctl -n vm.loadavg | awk '{print int($2)}')
  [ "$load" -lt "$GATE" ] && break
  [ $(( $(date +%s) - t0 )) -ge $(( 8 * 3600 )) ] && { log "ABORT: load $load never dropped below $GATE"; exit 2; }
  log "machine busy: load $load >= $GATE, waiting 300s"
  sleep 300
done
log "machine idle, starting benchmark"

# -- 5. star benchmarks: candidate, then paired v3 baseline -------------------
log "star 1/2: bc_alldays52_jun16_aug07_seed42 vs ALL"
PTCG_DEVICE=cpu nice -n 10 uv run python scripts/benchmark_agents.py \
  --agents all --focus bc_alldays52_jun16_aug07_seed42 --games "$GAMES" \
  --out reports/pool_star_bc_alldays52.json >> "$LOG" 2>&1 \
  || { log "ABORT: star 1 crashed"; exit 1; }
log "star 2/2: il_agent_v3_best vs ALL (paired baseline)"
PTCG_DEVICE=cpu nice -n 10 uv run python scripts/benchmark_agents.py \
  --agents all --focus il_agent_v3_best --games "$GAMES" \
  --out reports/pool_star_il_agent_v3_best.json >> "$LOG" 2>&1 \
  || { log "ABORT: star 2 crashed"; exit 1; }

log "POOL BENCHMARK DONE"
{
  echo ""
  echo "### Raw pool-benchmark record (auto-appended $(date '+%Y-%m-%d %H:%M'))"
  echo '- wired checkpoint: '"$pick_label"' (final acc '"$final_acc"', best acc '"$best_acc"')'
  echo '- fallback diagnostic: clean (reports/fallback_diag_bc_alldays52.json)'
  echo '- star results: reports/pool_star_bc_alldays52.json vs reports/pool_star_il_agent_v3_best.json'
  echo '  (full log: runs/bc_alldays52_pool_benchmark.log)'
} >> "$CARD"
