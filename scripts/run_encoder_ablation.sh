#!/bin/zsh
# Stage 1 of notes/experiments/2026-08-05-modernbert-vs-bert-encoder.md:
# BertModel vs ModernBERT-XXS, 2x2 factorial (block type x geometry), 3 seeds.
#
# Equal STEPS across every arm (repo rule 4), not equal epochs: --total-steps
# 12900 = one pass over the 4,554-episode 2026-07-26 train day at batch 64.
# Runs serially and nice'd so MPS is never contended.
set -u
cd "$(dirname "$0")/.."

STEPS=12900
DAY=2026-07-26
OUT=models/encoder_ablation
LOG=runs/encoder_ablation.log
mkdir -p "$OUT" runs

# arm|encoder_type|hidden|layers|heads|intermediate
ARMS=(
  "a_bert_base|bert|192|6|6|768"
  "b_mbert_matched|modernbert|192|6|6|512"
  "c_mbert_xxs|modernbert|256|7|4|384"
  "d_bert_xxsgeom|bert|256|7|4|1024"
)
SEEDS=(42 43 44)

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "ENCODER ABLATION BEGIN -- ${#ARMS[@]} arms x ${#SEEDS[@]} seeds, ${STEPS} steps each, train day ${DAY}"
for seed in $SEEDS; do
  for spec in $ARMS; do
    arm=${spec%%|*}; rest=${spec#*|}
    enc=${rest%%|*}; rest=${rest#*|}
    h=${rest%%|*}; rest=${rest#*|}
    L=${rest%%|*}; rest=${rest#*|}
    heads=${rest%%|*}; inter=${rest##*|}

    dir="$OUT/${arm}_s${seed}"
    if [[ -f "$dir/config.json" ]]; then
      log "SKIP $arm seed=$seed (already trained at $dir)"
      continue
    fi
    log "TRAIN $arm seed=$seed  enc=$enc h=$h L=$L heads=$heads inter=$inter"
    if nice -n 5 uv run python scripts/train_il.py \
        --data-source hub --hub-days "$DAY" --num-workers 4 \
        --encoder-type "$enc" --hidden-size "$h" --num-layers "$L" \
        --num-heads "$heads" --intermediate-size "$inter" \
        --total-steps "$STEPS" --lr 3e-4 --batch-size 64 --seed "$seed" \
        --eval-batches 20 \
        --out "$dir" --run-dir "runs/encoder_ablation/${arm}_s${seed}" \
        >>"$LOG" 2>&1; then
      log "OK $arm seed=$seed -> $dir"
    else
      log "FAIL $arm seed=$seed -- continuing with remaining arms"
    fi
  done
done
log "TRAINING DONE -- next: uv run python scripts/eval_encoder_ablation.py"
