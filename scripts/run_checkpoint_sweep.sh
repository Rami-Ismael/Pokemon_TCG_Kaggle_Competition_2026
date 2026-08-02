#!/usr/bin/env bash
# Gate 2 checkpoint sweep: 5 separate COMPLETE schedules (0.5/1/2/4/8 epochs
# over the full train day each), run sequentially so they don't compete for
# the same MPS device. Each writes to its own scratch models/il_agent_sweep/<X>ep/
# -- never the live models/il_agent/ path.
set -euo pipefail
cd "$(dirname "$0")/.."

for EPOCHS in 0.5 1 2 4 8; do
  OUT="models/il_agent_sweep/${EPOCHS}ep"
  RUNDIR="runs/sweep_${EPOCHS}ep"
  echo "=== starting epochs=${EPOCHS} -> ${OUT} ==="
  uv run python scripts/train_il.py \
    --epochs "${EPOCHS}" \
    --out "${OUT}" \
    --run-dir "${RUNDIR}" \
    --eval-batches 100
  echo "=== finished epochs=${EPOCHS} ==="
done

echo "=== sweep complete ==="
