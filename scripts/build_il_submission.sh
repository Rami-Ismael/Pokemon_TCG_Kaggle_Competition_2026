#!/usr/bin/env bash
# Refresh submissions/il_agent/ from the current source + checkpoint, then
# pack it into submission.tar.gz -- mirrors the layout of
# submissions/mega_lucario_improved/ (main.py + agent_core.py + deck.csv +
# cg/ engine), plus this agent's own model/ and pokemon_tcg/ (the encoder +
# model code it needs, since this is the only agent in the repo with an ML
# dependency at all).
#
# This script only builds the tarball. It does NOT submit it -- that's a
# separate, explicit step (see the `kaggle competitions submit` command
# printed at the end, or scripts/submit_il_agent.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

# Both overridable so a non-default checkpoint can be bundled without editing
# this script, e.g.
#   CHECKPOINT=models/il_alldays_0804 BUNDLE=submissions/il_alldays_0804 \
#     ./scripts/build_il_submission.sh
BUNDLE="${BUNDLE:-submissions/il_agent}"
CHECKPOINT="${CHECKPOINT:-models/il_agent}"

if [ ! -f "$CHECKPOINT/model.safetensors" ]; then
  echo "no checkpoint at $CHECKPOINT -- run scripts/train_il.py first" >&2
  exit 1
fi

rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/model" "$BUNDLE/pokemon_tcg"

cp agents/il_agent/agent_core.py agents/il_agent/main.py agents/il_agent/deck.csv "$BUNDLE/"
cp -R data/external/cg-lib/cg "$BUNDLE/cg"
cp "$CHECKPOINT/config.json" "$CHECKPOINT/model.safetensors" "$BUNDLE/model/"
cp src/pokemon_tcg/config.py src/pokemon_tcg/device.py src/pokemon_tcg/il_dataset.py src/pokemon_tcg/il_model.py "$BUNDLE/pokemon_tcg/"
touch "$BUNDLE/pokemon_tcg/__init__.py"

find "$BUNDLE" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

TARBALL="$BUNDLE/submission.tar.gz"
rm -f "$TARBALL"
tar -czf "$TARBALL" -C "$BUNDLE" \
  agent_core.py main.py deck.csv cg model pokemon_tcg

echo "built: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
echo ""
echo "NOT submitted. To submit (spends one of your 5 daily / 2-scored slots):"
echo "  uv run kaggle competitions submit pokemon-tcg-ai-battle \\"
echo "    -f $TARBALL -m \"il_agent v1: 3.3M-param BC, 73.0% top-1 offline (vs 38.1% majority), loses to rule_baseline/agent_core_improved on Rung 2\""
