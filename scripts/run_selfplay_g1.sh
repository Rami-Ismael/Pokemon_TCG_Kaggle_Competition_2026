#!/bin/zsh
# Provisional self-play generation 1 (rl_pipeline_v2 §3, PRIOR-v2 re-base
# rule): base + KL teacher = all-days imitation (il_alldays_0804),
# provisional pending the ~08-07 settled base-model verdict. v2 defaults:
# lr 1e-4, gamma 0.997, entropy 0.01->0.001@50%, PFSP-lite on, anchor gate
# every 20 updates. Preflight-retry handles other sessions' CPU bursts.
set -u
cd "$(dirname "$0")/.."
MAIN=/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
LOG=runs/selfplay_g1.log
t0=$(date +%s)
while :; do
  start=$(date +%s)
  nice -n 5 $MAIN/.venv-ppo/bin/python scripts/train_ppo_puffer.py \
    --init-from models/il_alldays_0804 \
    --kl-prior models/il_alldays_0804 \
    --league models/il_agent,models/s2v2/e0_s43 \
    --total-timesteps 2000000 --max-seconds 14400 \
    --run-tag selfplay_g1_prov --out models/selfplay_g1 >>"$LOG" 2>&1
  rc=$?
  ran=$(( $(date +%s) - start ))
  [ $rc -eq 0 ] && { echo "SELFPLAY G1 DONE rc=0" >>"$LOG"; exit 0; }
  # quick failure = preflight contention assert; retry up to 90 min total
  if [ $ran -lt 120 ] && [ $(( $(date +%s) - t0 )) -lt 5400 ]; then
    echo "preflight/launch failed fast (rc=$rc after ${ran}s) -- retrying in 120s" >>"$LOG"
    sleep 120
    continue
  fi
  echo "SELFPLAY G1 FAILED rc=$rc after ${ran}s -- not retrying" >>"$LOG"
  exit $rc
done
