#!/usr/bin/env bash
# Equal-steps arm: all-days corpus, but pinned to the ORIGINAL model's step
# count (38,562) so training length is held fixed and only the corpus varies.
# Card: notes/experiments/2026-08-04-equal-steps-data-vs-training-length.md
#
# Chains behind a running job rather than overlapping it (standing rule: MPS
# jobs never contend). Set WAIT_PID=0 to start immediately.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-38562}"          # models/il_agent/train_metadata.json -> step
OUT="${OUT:-models/il_alldays_equalsteps_0804}"
RUN_DIR="${RUN_DIR:-runs/il_alldays_equalsteps_0804}"
WAIT_PID="${WAIT_PID:-0}"

# ⚠️ MUST train with the SAME train_il.py the all-days model used. main's copy
# and this worktree's copy have OPPOSITE defaults for the Hub corpus when
# --hub-days is omitted:
#   main            hub_days = train_days -> ONLY 2026-07-26 (the small corpus)
#   this worktree   hub_days = None       -> ALL 10 days     (the all-days corpus)
# Running main's copy silently retrains on the ORIGINAL corpus and would make
# this experiment answer nothing. Caught on 2026-08-04 after a first launch did
# exactly that ("4554 episodes, days 2026-07-26" in the run header).
TRAIN_REPO="${TRAIN_REPO:-.claude/worktrees/rewrite-kaggle-pokemon-tcg-prompt-c69997}"

if [[ "$WAIT_PID" != "0" ]]; then
  echo "waiting for pid $WAIT_PID to exit before starting (MPS contention rule)..."
  while ps -p "$WAIT_PID" > /dev/null 2>&1; do sleep 30; done
  echo "pid $WAIT_PID gone; starting at $(date)"
fi

REPO_ROOT="$(pwd)"
echo "=== training: $STEPS steps on the all-days corpus -> $OUT ==="
echo "    (using $TRAIN_REPO/scripts/train_il.py -- the all-days-default copy)"
cd "$TRAIN_REPO"

# Same hyperparameters as BOTH comparison models; only --total-steps differs
# from the all-days-full run. --total-steps also drives the LR schedule, so
# this is a complete run at this budget, not a truncated one.
nice -n 5 uv run python scripts/train_il.py \
  --data-source hub \
  --num-workers 4 \
  --total-steps "$STEPS" \
  --lr 3e-4 \
  --batch-size 64 \
  --seed 42 \
  --eval-every-steps 10000 \
  --out "$OUT" \
  --run-dir "$RUN_DIR" 2>&1 | tee "$REPO_ROOT/reports/equal_steps_train.log"

CKPT_ABS="$(pwd)/$OUT"
if [[ ! -f "$CKPT_ABS/model.safetensors" ]]; then
  echo "TRAINING FAILED -- no checkpoint at $CKPT_ABS; not running the eval" >&2
  exit 1
fi

# Hard gate: refuse to evaluate a model that trained on the WRONG corpus.
# The run header names the days it streamed; the all-days arm must NOT be the
# single 2026-07-26 train day.
if grep -qE "train source:.*days 2026-07-26[,\"]?\s*$|4554 episodes, days 2026-07-26" \
     "$REPO_ROOT/reports/equal_steps_train.log"; then
  echo "ABORT -- trained on the single 2026-07-26 day, not the all-days corpus." >&2
  echo "That is the ORIGINAL corpus; this arm would answer nothing. See log." >&2
  exit 2
fi
grep -m1 "train source:" "$REPO_ROOT/reports/equal_steps_train.log" >&2 || true
cd "$REPO_ROOT"

echo "=== evaluating: exploiter vs the equal-steps model ==="
# NOTE: call eval_exploiter directly rather than the transfer script -- that
# script hardcodes the result name exploit_vs_alldays_imitation.json, which
# would CLOBBER the existing all-days-full result we are comparing against.
cd .claude/worktrees/ptcg-exploiter-agent-b80eab
IL_MODEL_DIR="$CKPT_ABS" PTCG_DEVICE=cpu PTCG_FALLBACK_TRACK=1 \
  nice -n 10 uv run python scripts/eval_exploiter.py \
    --ckpt models/ppo_exploiter_v1/u1565696_final \
    --opponent il_agent \
    --pairs 100 --workers 4 \
    --out reports/exploiter/transfer/exploit_vs_alldays_equalsteps.json 2>&1 \
  | grep -vE "OpenSpiel|Loading weights|resource_tracker|warnings.warn"

echo "=== done. Compare against:"
echo "  original imitation      0.955 [0.917, 0.976]"
echo "  plain imitation seeds   0.942  band [0.920, 0.970]"
echo "  all-days FULL (127748)  0.385 [0.320, 0.454]"
