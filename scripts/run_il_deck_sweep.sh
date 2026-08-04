#!/usr/bin/env bash
# IL model x deck sweep driver.
#
# One nice'd, CPU-only, single-threaded benchmark process per (arm, deck) cell,
# at most $CONC at a time so the laptop stays usable and no MPS job is contended
# (this sweep never touches MPS -- PTCG_DEVICE=cpu is also what the Kaggle
# evaluator does, so it is the honest device to measure on).
#
# Each process benchmarks ONE arm against the shared opponent pool. Ratings are
# NOT taken from the per-process Glicko (each would start from its own 1500
# prior); scripts/merge_il_sweep.py recomputes Glicko once over the union of all
# games, where the shared pool links every arm onto a common scale.
#
# Usage: run_il_deck_sweep.sh <out_subdir> <games_per_pair> <arm@deck> [arm@deck ...]
set -uo pipefail

OUT_SUBDIR="$1"; shift
GAMES="$1"; shift
CELLS=("$@")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$REPO/reports/il_sweep/$OUT_SUBDIR"
LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

POOL="tb_archaludon,makthanithin_1084_baseline,romanrozen_strong_start,tb_dragapult,wmh_alakazam,wmh_garchomp,tb_heuristic,random_legal"
CONC="${CONC:-3}"

export PTCG_DEVICE=cpu
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "sweep: ${#CELLS[@]} cells, games_per_pair=$GAMES, concurrency=$CONC"
echo "pool:  $POOL"
echo "out:   $OUTDIR"

for cell in "${CELLS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 5; done
  safe="${cell//@/__}"
  out="$OUTDIR/$safe.json"
  if [ -s "$out" ]; then
    echo "SKIP  $cell (already have $out)"
    continue
  fi
  echo "START $cell"
  (
    nice -n 10 uv run python "$REPO/scripts/benchmark_agents.py" \
      --agents "$cell,$POOL" \
      --games "$GAMES" \
      --no-glicko-persist --no-tb \
      --out "$out" >"$LOGDIR/$safe.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then echo "DONE  $cell"; else echo "FAIL  $cell rc=$rc"; fi
  ) &
done

wait
echo "SWEEP-COMPLETE $OUT_SUBDIR"
