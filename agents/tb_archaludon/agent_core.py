"""TomBombadyl Archaludon ex / Cinderace rule pilot (his ladder LEADER, mu 1196.1).

Thin shim into the vendored `agents/tb_shared/agent` package. The real, byte-verbatim
source is `agent/archaludon_agent.py` from github.com/TomBombadyl/kaggle_pokemon
(community v5 + his R7 empty-bench guard). We only (a) point the agent's own deck
resolver at *this* dir's deck.csv via the ARCHALUDON_DECK env override it already
checks first -- our benchmark harness runs from the repo root, which ships a decoy
deck.csv -- and (b) put tb_shared on sys.path so its `from agent.*` imports resolve.

Provenance / safety review: notebooks/reference/tombombadyl/INDEX.md.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ARCHALUDON_DECK", os.path.join(_HERE, "deck.csv"))
_SHARED = os.path.abspath(os.path.join(_HERE, os.pardir, "tb_shared"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from agent.archaludon_agent import agent, my_deck  # noqa: E402,F401
