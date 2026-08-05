#!/bin/zsh
# Self-play generation 2 (v2 §3): continues from gen-1's final policy WITH
# the warmed critic (policy_full.pt - kills the cold-start tax), anchored to
# the gen-1 promoted teacher ref430k (best-attested policy), league grown to
# all four prior bases/candidates. 6h budget. Preflight-retry as gen 1.
set -u
cd "$(dirname "$0")/.."
MAIN=/Users/ramiismael/projects/kaggle/Pokemon_TCG_Kaggle_Competition_2026
LOG=runs/selfplay_g3.log
t0=$(date +%s)
while :; do
  start=$(date +%s)
  nice -n 5 $MAIN/.venv-ppo/bin/python scripts/train_ppo_puffer.py \
    --init-from models/selfplay_g2/u1293312_final \
    --init-policy-full models/selfplay_g2/policy_full.pt \
    --kl-prior models/selfplay_g1/refs/u430080 \
    --league models/il_agent,models/s2v2/e0_s43,models/il_agent_full_0804,models/selfplay_g1/refs/u430080 \
    --total-timesteps 2000000 --max-seconds 21600 \
    --run-tag selfplay_g3 --out models/selfplay_g3 >>"$LOG" 2>&1
  rc=$?
  ran=$(( $(date +%s) - start ))
  [ $rc -eq 0 ] && { echo "SELFPLAY G3 DONE rc=0" >>"$LOG"; exit 0; }
  if [ $ran -lt 120 ] && [ $(( $(date +%s) - t0 )) -lt 10800 ]; then
    echo "preflight/launch failed fast (rc=$rc after ${ran}s) -- retrying in 120s" >>"$LOG"
    sleep 120
    continue
  fi
  echo "SELFPLAY G3 FAILED rc=$rc after ${ran}s -- not retrying" >>"$LOG"
  exit $rc
done
