#!/bin/zsh
# Rerun arm B after the 2026-08-05 23:39 SIGKILL (rc=137, jetsam under memory
# pressure from parallel sessions), then run the three holdout batteries.
# Arm B restarts from scratch — resuming mid-PPO would give it a second
# critic cold-start arm A never had (protocol asymmetry). Gates: no other
# real trainer alive (MPS rule) AND >=6 GB reclaimable RAM. Retries the arm
# on nonzero exit up to 3 times.
set -u
cd "$(dirname "$0")/.."
MAIN=/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
PY=$MAIN/.venv-ppo/bin/python
LOGDIR=reports/deckdiv
QLOG=$LOGDIR/queue.log
log() { echo "$(date '+%F %T') $*" | tee -a $QLOG; }

other_trainers() {
  ps -axo pid=,command= | grep -E "python[0-9.]* .*train_(il|ppo_puffer)\.py" \
    | grep -v grep | grep -v "deckdiv_arm" | grep -v -- "--smoke" \
    | wc -l | tr -d ' '
}
avail_gb() {  # free + inactive pages, GB
  vm_stat | awk '/Pages free|Pages inactive/ {gsub(/\./,"",$NF); s+=$NF} END {printf "%d", s*16384/1e9}'
}
wait_for_headroom() {
  while [ "$(other_trainers)" != "0" ] || [ "$(avail_gb)" -lt 6 ]; do
    log "wait: trainers=$(other_trainers) avail=$(avail_gb)GB"; sleep 3600
  done
}

battery() {
  local label=$1 ckpt=$2
  log "BATTERY $label: $ckpt (holdout 8 decks x 50 pairs, opponent-only)"
  nice -n 5 $PY scripts/eval_exploiter.py \
    --ckpt "$ckpt" --opponent il_agent --temperature 1.0 \
    --deck-pool '@configs/deck_pools/opp_holdout.txt' --opp-deck-only \
    --pairs 400 --workers 8 \
    --out $LOGDIR/holdout_${label}.json >> $LOGDIR/battery_${label}.log 2>&1
  log "BATTERY $label: exit rc=$?"
}

log "armB-resume queue start (pid $$)"
tries=0
while [ $tries -lt 3 ]; do
  wait_for_headroom
  tries=$((tries+1))
  log "ARM deckdiv_armB: start attempt $tries (500000 steps, from scratch)"
  rm -rf models/deckdiv_armB_opppool
  nice -n 5 $PY scripts/train_ppo_puffer.py \
    --init-from models/selfplay_g1/u430080 \
    --mix 0.5,0.3,0.2 \
    --league models/il_alldays_0804,models/selfplay_g1/u430080 \
    --total-timesteps 500000 \
    --opp-deck-pool '@configs/deck_pools/opp_train.txt' \
    --run-tag deckdiv_armB --out models/deckdiv_armB_opppool \
    --skip-preflight >> $LOGDIR/deckdiv_armB.log 2>&1
  rc=$?
  log "ARM deckdiv_armB: exit rc=$rc (attempt $tries)"
  [ $rc -eq 0 ] && break
done
[ $rc -ne 0 ] && { log "arm B failed after $tries attempts"; exit 1; }

CKB=$(ls -dt models/deckdiv_armB_opppool/*_final 2>/dev/null | head -1)
log "arm B final: $CKB"
battery base models/selfplay_g1/u430080
battery armA "$(ls -dt models/deckdiv_armA_control/*_final | head -1)"
battery armB "$CKB"
log "armB-resume queue done"
