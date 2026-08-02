"""TomBombadyl Alakazam best5 rule pilot (mu 659.0).

Thin shim into the vendored `agents/tb_shared/agent` package. The real, byte-verbatim
source is `agent/alakazam_agent.py` from github.com/TomBombadyl/kaggle_pokemon (ryotasueyoshi best5 rules + dragapult matchup levers).
We only (a) point the agent's own deck resolver at *this* dir's deck.csv via the
ALAKAZAM_DECK env override it already checks first -- our benchmark harness runs from the
repo root, which ships a decoy deck.csv -- and (b) put tb_shared on sys.path so its
`from agent.*` imports resolve.

Provenance / safety review: notebooks/reference/tombombadyl/INDEX.md.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ALAKAZAM_DECK", os.path.join(_HERE, "deck.csv"))
_SHARED = os.path.abspath(os.path.join(_HERE, os.pardir, "tb_shared"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from agent.alakazam_agent import agent, my_deck  # noqa: E402,F401
