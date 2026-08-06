#!/usr/bin/env bash
# Test candidates against the full validated opponent bed (configs/pool_v3.txt).
#
# Star topology (--focus): each candidate plays ONLY candidate-vs-pool, 42
# pairings instead of the 903 a round-robin would replay. The pool's internal
# ratings come from the one-off round-robin in
# reports/il_sweep/poolv3_rr/, and scripts/merge_il_sweep.py pools both so every
# agent lands on one Glicko scale.
#
# Usage: run_testbed.sh <out_subdir> <games_per_pair> <agent> [agent ...]
#   CONC=N to change parallelism (default 3)
set -uo pipefail

OUT_SUBDIR="$1"; shift
GAMES="$1"; shift
CANDIDATES=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/reports/il_sweep/$OUT_SUBDIR"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

POOL=$(tr '\n' ',' < "$REPO/configs/pool_v3.txt" | sed 's/,$//')
NPOOL=$(tr ',' '\n' <<< "$POOL" | grep -c .)
CONC="${CONC:-3}"

export PTCG_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

echo "testbed: ${#CANDIDATES[@]} candidate(s) vs $NPOOL opponents, ${GAMES} pairs/cell, conc=$CONC"
echo "out: $OUTDIR"

for c in "${CANDIDATES[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
  safe="${c//@/__}"
  out="$OUTDIR/$safe.json"
  if [ -s "$out" ]; then echo "SKIP  $c"; continue; fi
  echo "START $c"
  (
    nice -n 10 uv run python "$REPO/scripts/benchmark_agents.py" \
      --agents "$c,$POOL" --focus "$c" --games "$GAMES" \
      --no-glicko-persist --no-tb --out "$out" >"$LOGDIR/$safe.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then echo "DONE  $c"; else echo "FAIL  $c rc=$rc"; fi
  ) &
done

wait
echo "TESTBED-COMPLETE $OUT_SUBDIR"
