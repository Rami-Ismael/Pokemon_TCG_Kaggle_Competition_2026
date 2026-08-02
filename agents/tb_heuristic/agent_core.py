"""TomBombadyl HeuristicScorer scorer brain on real_dragapult_ex (his catalog deck a2_kyogre_33_energy, mu 633.0, is not in-repo; using real_dragapult_ex).

Vendored from github.com/TomBombadyl/kaggle_pokemon. His scorer brains are packaged by
`scripts/package_submission.py --scorer heuristic`, which bundles the whole `agent/`
package and generates a main.py doing `build_agent(scorer=...)`. This wrapper mirrors
that generated main.py: it boots the vendored `agents/tb_shared/agent` package onto
sys.path, resolves the deck next to this file (our harness runs from the repo root),
and builds the same scorer. `heuristic` is a deck-agnostic hand-rules brain (hand-tuned MAIN priorities, no search).

Provenance / safety review: notebooks/reference/tombombadyl/INDEX.md.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SHARED = os.path.abspath(os.path.join(_HERE, os.pardir, "tb_shared"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)
_DECK = os.path.join(_HERE, "deck.csv")

from agent.agent import build_agent  # noqa: E402

_AGENT = build_agent(seed=0, deck_path=_DECK)


def _read_deck() -> list:
    with open(_DECK, "r", encoding="utf-8") as fh:
        rows = [r for r in fh.read().split("\n") if r.strip()]
    return [int(rows[i]) for i in range(60)]


my_deck = _read_deck()


def agent(obs_dict: dict) -> list:
    if obs_dict.get("select") is None:
        return my_deck
    return _AGENT.act(obs_dict)
