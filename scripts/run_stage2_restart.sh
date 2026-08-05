#!/bin/zsh
# Stage-2 restart overnight chain — rl_pipeline_v2.md work items B2a + B2(E0).
#
# Waits for the concurrent exploiter_v1 PPO run to release MPS (chain runs,
# never overlap — standing rule), then runs, sequentially:
#   1. critic (TD target, restored corpus, warm-started from PRIOR)
#   2. critic audit parts (i)/(ii) + the (iii) hand-read dump
#   3. arm E0 x 3 seeds (control, w=1 — independent of the critic verdict)
# and STOPS. E1/E2 launch only after a human (or the driving session) reads
# the audit's part (iii); a critic that fails (i)/(ii) is announced here and
# the advantage arms are then replaced by the registered E-fallback.
#
# Every step logs to runs/stage2_restart.log with BEGIN/OK/FAIL markers a
# Monitor can key on.
set -u
cd "$(dirname "$0")/.."
LOG=runs/stage2_restart.log
mkdir -p runs models/s2v2
DAYS="2026-07-01,2026-07-26"
STEPS=13000

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "CHAIN BEGIN (pid $$)"

# -- 0. wait for the machine ------------------------------------------------
while pgrep -f "train_ppo_puffer.py --opponent-module" >/dev/null 2>&1; do
  sleep 60
done
log "exploiter_v1 gone -- machine free"
sleep 10
FREE_GB=$(python3 -c "import shutil; print(int(shutil.disk_usage('.').free/1e9))")
if [ "$FREE_GB" -lt 5 ]; then
  log "CHAIN FAIL: only ${FREE_GB} GB disk free"
  exit 1
fi
log "preflight ok: ${FREE_GB} GB free"

# -- 1. critic (TD) ----------------------------------------------------------
log "STEP critic BEGIN (td, 1 epoch, days $DAYS)"
if nice -n 5 uv run python scripts/train_critic.py \
    --target td --data-source hub --hub-days "$DAYS" \
    --num-workers 4 --epochs 1 --out models/critic_td >>"$LOG" 2>&1; then
  log "STEP critic OK"
else
  log "CHAIN FAIL: critic training crashed"
  exit 1
fi

# -- 2. critic audit ---------------------------------------------------------
log "STEP audit BEGIN"
if nice -n 5 uv run python scripts/audit_critic.py \
    --critic-dir models/critic_td >>"$LOG" 2>&1; then
  log "STEP audit OK (verdicts above; part iii needs a human read)"
else
  log "CHAIN FAIL: audit crashed"
  exit 1
fi

# -- 3. E0 control arm, 3 seeds (critic-independent) --------------------------
for SEED in 42 43 44; do
  log "STEP e0_s${SEED} BEGIN"
  if nice -n 5 uv run python scripts/train_il.py \
      --data-source hub --hub-days "$DAYS" --num-workers 4 \
      --init-from models/il_agent --total-steps $STEPS --lr 1e-4 \
      --seed $SEED --out models/s2v2/e0_s${SEED} \
      --run-dir runs/s2v2_e0_s${SEED} >>"$LOG" 2>&1; then
    log "STEP e0_s${SEED} OK"
  else
    log "CHAIN FAIL: e0_s${SEED} crashed"
    exit 1
  fi
done

log "CHAIN DONE -- critic + audit + E0x3 complete; awaiting audit-(iii) review before E1/E2"
