#!/usr/bin/env bash
# Refresh submissions/mega_lucario_improved/ from current source + checkpoint,
# then pack submission.tar.gz. Same flat-bundle pattern as
# scripts/build_il_submission.sh, but for agent_core_improved.py -- which
# now also needs bc_prior.py and the il_agent checkpoint bundled, since
# SEARCH_ALGO's UCB1 loop uses the trained policy as a PUCT-style prior
# (see agents/mega_lucario/bc_prior.py). Degrades to plain search with no
# prior if that model/bc_prior import fails for any reason -- never
# required for this agent to run, only to run slightly better.
#
# main.py and deck.csv have no separate agents/ source for this bundle
# (unlike agent_core_improved.py/bc_prior.py) -- they're preserved from
# whatever's already in submissions/mega_lucario_improved/ before it gets
# wiped, so the __file__-loading fix already applied there survives a
# rebuild. If that directory doesn't exist yet, write main.py/deck.csv by
# hand first.
set -euo pipefail
cd "$(dirname "$0")/.."

BUNDLE=submissions/mega_lucario_improved
CHECKPOINT=models/il_agent
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "no checkpoint at $CHECKPOINT -- run scripts/train_il.py first" >&2
  exit 1
fi
if [ ! -f "$BUNDLE/main.py" ] || [ ! -f "$BUNDLE/deck.csv" ]; then
  echo "$BUNDLE is missing main.py/deck.csv -- nothing to preserve, write them first" >&2
  exit 1
fi

cp "$BUNDLE/main.py" "$BUNDLE/deck.csv" "$STAGE/"

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/model" "$BUNDLE/pokemon_tcg"

cp "$STAGE/main.py" "$STAGE/deck.csv" "$BUNDLE/"
cp agents/mega_lucario/agent_core_improved.py agents/mega_lucario/bc_prior.py "$BUNDLE/"
cp -R data/external/cg-lib/cg "$BUNDLE/cg"
cp "$CHECKPOINT/config.json" "$CHECKPOINT/model.safetensors" "$BUNDLE/model/"
cp src/pokemon_tcg/config.py src/pokemon_tcg/device.py src/pokemon_tcg/il_dataset.py src/pokemon_tcg/il_model.py "$BUNDLE/pokemon_tcg/"
touch "$BUNDLE/pokemon_tcg/__init__.py"

find "$BUNDLE" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

TARBALL="$BUNDLE/submission.tar.gz"
rm -f "$TARBALL"
tar -czf "$TARBALL" -C "$BUNDLE" \
  main.py agent_core_improved.py bc_prior.py deck.csv cg model pokemon_tcg

echo "built: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
