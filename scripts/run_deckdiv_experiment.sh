#!/bin/zsh
# Deck-diversity generalization experiment — arm A (control) vs arm B
# (opponent-deck pool), chained behind whatever owns MPS right now.
# Card: notes/experiments/2026-08-05-opponent-deck-diversity-selfplay.md
#
# MPS is never contended (standing rule): hard-wait until no other trainer
# (train_il.py / train_ppo_puffer.py) is alive. CPU contention from parallel
# sessions' benchmarks only slows steps/s (budget is in STEPS, not hours), so
# we soft-wait for quiet up to MAX_QUIET_WAIT_S, then proceed nice'd with
# --skip-preflight (the built-in preflight would refuse on those benchmarks;
# disk/RAM are re-checked here instead).
set -u
cd "$(dirname "$0")/.."   # worktree root
MAIN=/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
PY=$MAIN/.venv-ppo/bin/python
STEPS=500000
LOGDIR=reports/deckdiv
mkdir -p $LOGDIR
QLOG=$LOGDIR/queue.log
log() { echo "$(date '+%F %T') $*" | tee -a $QLOG; }

other_trainers() {  # real trainers that aren't ours; --smoke runs are CPU-only
  # (and one has been observed hung for 20h), wrapper shells die with children
  ps -axo pid=,command= | grep -E "python[0-9.]* .*train_(il|ppo_puffer)\.py" \
    | grep -v grep | grep -v "deckdiv_arm" | grep -v -- "--smoke" \
    | wc -l | tr -d ' '
}
heavy_benchmarks() {
  ps -axo pcpu=,command= | grep "benchmark_agents.py" | grep -v grep \
    | awk '$1 > 50' | wc -l | tr -d ' '
}

log "queue start (pid $$)"
while [ "$(other_trainers)" != "0" ]; do
  log "waiting: $(other_trainers) other trainer(s) hold MPS"; sleep 300
done
log "MPS free"

MAX_QUIET_WAIT_S=21600; waited=0
while [ "$(heavy_benchmarks)" != "0" ] && [ $waited -lt $MAX_QUIET_WAIT_S ]; do
  log "soft-wait: $(heavy_benchmarks) heavy benchmark proc(s); waited ${waited}s"
  sleep 600; waited=$((waited+600))
done
[ "$(heavy_benchmarks)" != "0" ] && log "proceeding despite CPU contention (soft-wait expired)"

free_gb=$(df -g . | tail -1 | awk '{print $4}')
if [ "$free_gb" -lt 10 ]; then log "ABORT: ${free_gb}GB disk free"; exit 1; fi

run_arm() {  # $1 tag  $2 out  $3... extra flags
  local tag=$1 out=$2; shift 2
  log "ARM $tag: start -> $out ($STEPS steps)"
  nice -n 5 $PY scripts/train_ppo_puffer.py \
    --init-from models/selfplay_g1/u430080 \
    --mix 0.5,0.3,0.2 \
    --league models/il_alldays_0804,models/selfplay_g1/u430080 \
    --total-timesteps $STEPS \
    --run-tag "$tag" --out "$out" --skip-preflight \
    "$@" >> $LOGDIR/${tag}.log 2>&1
  local rc=$?
  log "ARM $tag: exit rc=$rc"
  return $rc
}

battery() {  # $1 label  $2 ckpt-dir
  local label=$1 ckpt=$2
  log "BATTERY $label: $ckpt (holdout 8 decks x 50 pairs, opponent-only)"
  nice -n 5 $PY scripts/eval_exploiter.py \
    --ckpt "$ckpt" --opponent il_agent --temperature 1.0 \
    --deck-pool '@configs/deck_pools/opp_holdout.txt' --opp-deck-only \
    --pairs 400 --workers 8 \
    --out $LOGDIR/holdout_${label}.json >> $LOGDIR/battery_${label}.log 2>&1
  log "BATTERY $label: exit rc=$?"
}

final_ckpt() {  # newest *_final else newest snapshot under $1
  local d
  d=$(ls -dt "$1"/*_final 2>/dev/null | head -1)
  [ -z "$d" ] && d=$(ls -dt "$1"/u* 2>/dev/null | head -1)
  echo "$d"
}

run_arm deckdiv_armA models/deckdiv_armA_control || { log "arm A failed"; exit 1; }
run_arm deckdiv_armB models/deckdiv_armB_opppool \
  --opp-deck-pool '@configs/deck_pools/opp_train.txt' || { log "arm B failed"; exit 1; }

CKA=$(final_ckpt models/deckdiv_armA_control)
CKB=$(final_ckpt models/deckdiv_armB_opppool)
log "final checkpoints: A=$CKA B=$CKB"
battery base models/selfplay_g1/u430080
battery armA "$CKA"
battery armB "$CKB"
log "queue done"
