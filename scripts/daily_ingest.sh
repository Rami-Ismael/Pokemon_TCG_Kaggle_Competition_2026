#!/usr/bin/env bash
# Daily rolling episode ingest (ADR-001 action item 3, made recurring
# 2026-08-05): fetch YESTERDAY's Kaggle episodes for the BC corpus, then
# pack -> local verify -> upload -> HUB byte-verify -> splits.json entry ->
# prune raw. Raw pruning of auto-ingested days has standing user approval
# (2026-08-05); the Hub copy is byte-recoverable via pack_episodes.py unpack.
#
# Scheduled by ~/Library/LaunchAgents/com.ptcg.daily-ingest.plist (07:30
# local). Safe to run manually or concurrently with other sessions: an
# already-on-Hub day is a clean skip, downloads resume, and the Kaggle
# replay endpoint's rate limit (429 above ~0.3 req/s, see
# notes/adr001_hf_streaming_corpus.md) is handled by progress-aware backoff.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONUNBUFFERED=1

D="${1:-$(date -v-1d '+%Y-%m-%d')}"
RAW="data/episodes/splits/train-$D"
echo "=== DAILY INGEST $D: start $(date '+%F %T') ==="

free_gb=$(df -g / | tail -1 | awk '{print $4}')
if [ "$free_gb" -lt 30 ]; then
  echo "=== DAILY INGEST $D: SKIPPED, only ${free_gb} GB free (need 30) ==="
  exit 1
fi

# Already on the Hub (e.g. another session ingested it) -> clean skip.
if uv run python scripts/ingest_episodes.py status 2>/dev/null | grep -q "^$D .*uploaded"; then
  echo "=== DAILY INGEST $D: already on the Hub; nothing to do ==="
  exit 0
fi

# Download with progress-aware backoff: a throttled attempt that still grew
# the folder resets the retry counter; only 5 consecutive zero-progress
# attempts (~15h spread) abort.
attempt=0
while true; do
  n_before=$(ls "$RAW" 2>/dev/null | wc -l | tr -d ' ')
  DL_OUT=$(uv run python scripts/ingest_episodes.py download --date "$D" --split train --sleep 1.0 2>&1)
  rc=$?
  echo "$DL_OUT"
  [ $rc -eq 0 ] && break
  # "0 episodes found" is a LEGITIMATELY EMPTY day (the episode listing
  # cannot reach days older than the current top teams' submissions), not a
  # throttle -- skip immediately instead of burning 5 backoff cycles (~15h)
  # on a day that will never yield anything. Observed live on 2026-07-20.
  if echo "$DL_OUT" | grep -q "^0 episodes found"; then
    echo "=== DAILY INGEST $D: 0 episodes reachable for this day -- skipping ==="
    exit 0
  fi
  n_after=$(ls "$RAW" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n_after" -gt "$n_before" ]; then
    echo "=== DAILY INGEST $D: throttled mid-harvest but progressed ($n_before -> $n_after); retry counter reset ==="
    attempt=1
  else
    attempt=$((attempt+1))
  fi
  if [ $attempt -ge 5 ]; then
    echo "=== DAILY INGEST $D: DEAD-STOPPED after 5 zero-progress attempts -- raw kept ==="
    exit 1
  fi
  wait_s=$((3600 * attempt))
  echo "=== DAILY INGEST $D: backing off $((wait_s/60)) min (attempt $attempt) ==="
  sleep $wait_s
done

if [ ! -d "$RAW" ] || [ -z "$(ls "$RAW" 2>/dev/null)" ]; then
  echo "=== DAILY INGEST $D: no episodes downloaded (empty day?) -- nothing to upload ==="
  exit 0
fi

OUT=$(uv run python scripts/ingest_episodes.py run --split-dir "$RAW" 2>&1)
rc=$?
echo "$OUT"
if [ $rc -ne 0 ] || ! echo "$OUT" | grep -q "verified faithful"; then
  echo "=== DAILY INGEST $D: INGEST/VERIFY FAILED (exit $rc) -- raw kept ==="
  exit 1
fi

# splits.json entry with the exact Hub-verified count (footers only).
uv run python - "$D" <<'PYEOF'
import json, sys
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

day = sys.argv[1]
fs = HfFileSystem()
n = 0
for s in fs.glob(f"datasets/Rami/ptcg-episodes/train/day={day}/*.parquet"):
    with fs.open(s, "rb") as f:
        n += pq.ParquetFile(f).metadata.num_rows
p = "data/episodes/splits/splits.json"
d = json.loads(open(p).read())
d[f"train-{day}"] = {
    "date": day,
    "episodes": n,
    "folder": f"splits/train-{day}",
    "note": "Daily rolling ingest; raw pruned after Hub-verify (standing "
            "user approval 2026-08-05). Hub shards are the only copy. "
            "NOT in any train union until one names it.",
}
open(p, "w").write(json.dumps(d, indent=2) + "\n")
print(f"splits.json: train-{day} = {n} episodes")
PYEOF

echo "=== DAILY INGEST $D: hub-verified; pruning raw $(du -sh "$RAW" | cut -f1) (standing approval) ==="
rm -rf "$RAW"
echo "=== DAILY INGEST $D: DONE $(date '+%F %T') ==="
