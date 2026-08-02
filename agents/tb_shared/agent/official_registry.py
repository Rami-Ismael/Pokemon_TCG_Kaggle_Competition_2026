"""Official Kaggle rule-based sample agents + imported community pilots.

Organizer samples (kiyotah kernels):
  - Mega Lucario ex   -> agent/lucario_policy.py (LucarioScorer)
  - Dragapult ex      -> agent/dragapult_agent.py
  - Iono's deck       -> agent/iono_agent.py
  - Mega Abomasnow ex -> agent/abomasnow_agent.py

Community import (659 mu reference):
  - Alakazam best5    -> agent/alakazam_agent.py (ryotasueyoshi notebook)

Decks without a wired pilot (e.g. Trevenant) use random only when opted in.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path

from agent.cg_bootstrap import ensure_cg_engine  # noqa: E402
from agent.lucario_policy import is_lucario_deck

# Kaggle organizer rule samples (kiyotah) — one pilot per archetype.
KERNEL_URLS = {
    "mega_lucario_ex": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-lucario-ex-deck",
    "dragapult_ex": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-dragapult-ex-deck",
    "iono": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-iono-s-deck",
    "mega_abomasnow_ex": "https://www.kaggle.com/code/kiyotah/a-sample-rule-based-agent-mega-abomasnow-ex-deck",
}

# Card-id signatures for archetype detection (deck list, not board).
_LUCARIO_LINE = frozenset({673, 674, 676, 677, 678})
_DRAGAPULT_LINE = frozenset({119, 120, 121})
_ABOMASNOW_LINE = frozenset({722, 723})
_IONO_LINE = frozenset({265, 268, 269, 270, 271})
_ALAKAZAM_LINE = frozenset({741, 742, 743})
_STARMIE_LINE = frozenset({1030, 1031, 860, 861})

# Slugs match extract_public_agents.FALLBACK_DECK keys / kaggle kernel names.
KERNEL_SLUGS = {
    "mega_lucario_ex": "a-sample-rule-based-agent-mega-lucario-ex-deck",
    "dragapult_ex": "a-sample-rule-based-agent-dragapult-ex-deck",
    "iono": "a-sample-rule-based-agent-iono-s-deck",
    "mega_abomasnow_ex": "a-sample-rule-based-agent-mega-abomasnow-ex-deck",
}

_STANDALONE_ENV = {
    "dragapult_ex": "DRAGAPULT_DECK",
    "iono": "IONO_DECK",
    "mega_abomasnow_ex": "ABOMASNOW_DECK",
    "alakazam_psychic": "ALAKAZAM_DECK",
    "starmie_water": "STARMIE_DECK",
}

_STANDALONE_MODULE = {
    "dragapult_ex": "agent.dragapult_agent",
    "iono": "agent.iono_agent",
    "mega_abomasnow_ex": "agent.abomasnow_agent",
    "alakazam_psychic": "agent.alakazam_agent",
    "starmie_water": "agent.starmie_agent",
}

# Human-readable pilot source for logs / verify_official_opponents.py
OFFICIAL_PILOT_MODULES: dict[str, str] = {
    "mega_lucario_ex": "agent.lucario_policy.LucarioScorer",
    "alakazam_psychic": "agent.alakazam_agent",
    "starmie_water": "agent.starmie_agent",
    **{k: v for k, v in _STANDALONE_MODULE.items()},
}

# Default field training: one CSV per archetype with an official Kaggle rule pilot.
OFFICIAL_FIELD_DECK_STEMS = [
    "dragapult_ex_sample",
    "real_mega_lucario_ex",
    "real_dragapult_ex",
    "real_mega_abomasnow_ex",
    "real_iono",
    "top_mined_dragapult_ex",
    "top_mined_iono",
    "top_mined_mega_abomasnow_ex",
    "top_mined_mega_lucario_ex",
    "top_mined_alakazam",
    "ryotasueyoshi_alakazam_best5",
    "starmie_froslass_ashleysandlin",
]

# No organizer sample — excluded unless training opts into random.
RANDOM_ONLY_DECK_STEMS = [
    "top_mined_trevenant",
]

_standalone_cache: dict[tuple[str, str], Callable[[dict], list[int]]] = {}
_scorer_cache: dict[str, Callable[[dict], list[int]]] = {}


def detect_official_archetype(deck_ids: list[int]) -> str | None:
    """Return archetype key if this deck matches a known official sample."""
    ids = set(deck_ids)
    if _LUCARIO_LINE.issubset(ids):
        return "mega_lucario_ex"
    if _DRAGAPULT_LINE.issubset(ids):
        return "dragapult_ex"
    if _ABOMASNOW_LINE.issubset(ids):
        return "mega_abomasnow_ex"
    if ids & _IONO_LINE:
        return "iono"
    if _ALAKAZAM_LINE.issubset(ids):
        return "alakazam_psychic"
    if _STARMIE_LINE.issubset(ids):
        return "starmie_water"
    return None


def official_archetype_for_opponent(opp_name: str, deck_ids: list[int]) -> str | None:
    """Name hint + deck signature."""
    stem = opp_name.lower()
    if "lucario" in stem:
        return "mega_lucario_ex"
    if "dragapult" in stem:
        return "dragapult_ex"
    if "abomasnow" in stem:
        return "mega_abomasnow_ex"
    if "iono" in stem:
        return "iono"
    if "starmie" in stem or "froslass" in stem:
        return "starmie_water"
    if "alakazam" in stem:
        return "alakazam_psychic"
    return detect_official_archetype(deck_ids)


def _lucario_act(deck_path: str, seed: int = 0) -> Callable[[dict], list[int]]:
    key = str(Path(deck_path).resolve())
    if key not in _scorer_cache:
        from agent.agent import build_agent
        from agent.lucario_policy import LucarioScorer

        agent = build_agent(
            seed=seed,
            deck_path=deck_path,
            scorer=LucarioScorer(deck_path=deck_path),
        )
        _scorer_cache[key] = agent.act
    return _scorer_cache[key]


def _standalone_act(archetype: str, deck_path: str) -> Callable[[dict], list[int]]:
    abs_path = str(Path(deck_path).resolve())
    cache_key = (archetype, abs_path)
    if cache_key in _standalone_cache:
        return _standalone_cache[cache_key]

    env_var = _STANDALONE_ENV.get(archetype)
    module_name = _STANDALONE_MODULE.get(archetype)
    if not env_var or not module_name:
        raise ValueError(f"no standalone module for {archetype}")

    os.environ[env_var] = abs_path
    ensure_cg_engine()

    # Prefer repo module (dragapult_agent.py); fall back to extracted kernel main.py.
    try:
        mod = importlib.import_module(module_name)
        mod = importlib.reload(mod)
        fn = mod.agent
        _standalone_cache[cache_key] = fn
        return fn
    except ModuleNotFoundError:
        from agent.kernel_loader import load_kernel_agent

        slug = KERNEL_SLUGS[archetype]
        fn = load_kernel_agent(slug)
        _standalone_cache[cache_key] = fn
        return fn


def make_official_opponent(
    deck_path: str,
    deck_ids: list[int],
    *,
    opp_name: str = "",
    seed: int = 0,
) -> tuple[Callable[[dict], list[int]], str]:
    """Return (act_fn, archetype_label). Raises FileNotFoundError if sample missing."""
    archetype = official_archetype_for_opponent(opp_name, deck_ids)
    if archetype is None:
        raise ValueError(
            f"no official Kaggle rule sample for deck {opp_name or deck_path} "
            f"(not Lucario/Dragapult/Iono/Abomasnow/Alakazam/Starmie). Use --non-official-brain random to opt in."
        )
    if archetype == "mega_lucario_ex":
        return _lucario_act(deck_path, seed), archetype
    return _standalone_act(archetype, deck_path), archetype


def list_missing_official_samples(
    opponents: dict[str, list[int]],
    *,
    allow_non_official: set[str],
) -> list[str]:
    """Opponent stems that lack an official sample and are not in allow_non_official."""
    missing = []
    for name, deck in opponents.items():
        if name in allow_non_official:
            continue
        if official_archetype_for_opponent(name, deck) is None:
            missing.append(name)
    return missing
