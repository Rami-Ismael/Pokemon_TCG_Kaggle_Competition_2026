from __future__ import annotations

import os
from typing import Any

from .handwritten_policy_core import choose
from .manual_guards import apply as apply_manual_guards
from .strategic_memory import StrategicMemory

DECK = [
    7,7,7,7,7,7,7,7,7,7,104,104,112,112,112,112,646,646,646,646,
    647,647,647,648,648,648,860,860,1079,1079,1079,1080,1086,1086,
    1086,1086,1097,1097,1097,1122,1137,1152,1152,1152,1152,1182,
    1182,1219,1219,1219,1219,1227,1227,1227,1227,1231,1259,1259,1259,1259,
]

# Only short action history is kept. No table, model, fitted parameter, replay index,
# search tree, rollout, or learned artifact is used.
_HISTORY: list[dict[str, int]] = []
_STRATEGIC_MEMORY = StrategicMemory()


def _card_id(x: Any) -> int:
    return int(x.get("id", 0) or 0) if isinstance(x, dict) else 0


def _serial(x: Any) -> int:
    return int(x.get("serial", -1) if isinstance(x, dict) else -1)


def _zone_list(player: dict, area: int) -> list:
    return {
        2: player.get("hand") or [],
        3: player.get("discard") or [],
        4: player.get("active") or [],
        5: player.get("bench") or [],
        6: player.get("prize") or [],
    }.get(area, [])


def _relative(player_index: int, your_index: int) -> int:
    return 0 if int(player_index) == int(your_index) else 1


def _resolve_source(obs: dict, option: dict) -> tuple[dict | None, int, int]:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    your = int(current.get("yourIndex", 0) or 0)
    players = current.get("players") or [{}, {}]
    pidx = int(option.get("playerIndex", your) if option.get("playerIndex") is not None else your)
    if not 0 <= pidx < len(players):
        pidx = your
    area = option.get("area")
    idx = option.get("index")
    obj = None
    zone = 0
    if area == 1:
        arr = select.get("deck") or []
        zone = 1
        if isinstance(idx, int) and 0 <= idx < len(arr):
            obj = arr[idx]
    elif area in (2, 3, 4, 5, 6, 7):
        arr = _zone_list(players[pidx], int(area))
        zone = int(area)
        if isinstance(idx, int) and 0 <= idx < len(arr):
            obj = arr[idx]
    elif isinstance(idx, int):
        # MAIN options without an explicit area refer to the acting player's hand.
        arr = players[your].get("hand") or []
        if 0 <= idx < len(arr):
            obj = arr[idx]
            zone = 2
    return obj, zone, _relative(pidx, your)


def _resolve_target(obs: dict, option: dict) -> tuple[dict | None, int, int]:
    current = obs.get("current") or {}
    your = int(current.get("yourIndex", 0) or 0)
    players = current.get("players") or [{}, {}]
    area = option.get("inPlayArea")
    idx = option.get("inPlayIndex")
    if area is None and option.get("type") == 10:
        area = option.get("area")
        idx = option.get("index")
    pidx = int(option.get("playerIndex", your) if option.get("playerIndex") is not None else your)
    if not 0 <= pidx < len(players):
        pidx = your
    arr = _zone_list(players[pidx], int(area)) if area in (4, 5) else []
    obj = arr[idx] if isinstance(idx, int) and 0 <= idx < len(arr) else None
    return obj, int(area or 0), _relative(pidx, your)


def _semantic(obs: dict, option: dict) -> dict[str, int]:
    src, source_zone, source_rel = _resolve_source(obs, option)
    tgt, target_area, target_rel = _resolve_target(obs, option)
    return {
        "type": int(option.get("type", -1) if option.get("type") is not None else -1),
        "source_id": _card_id(src),
        "source_serial": _serial(src),
        "source_zone": source_zone,
        "source_rel": source_rel,
        "target_id": _card_id(tgt),
        "target_serial": _serial(tgt),
        "target_area": target_area,
        "target_rel": target_rel,
        "attack_id": int(option.get("attackId", 0) or 0),
        "area": int(option.get("area", 0) or 0),
        "index": int(option.get("index", -1) if option.get("index") is not None else -1),
        "inplay_area": int(option.get("inPlayArea", 0) or 0),
        "inplay_index": int(option.get("inPlayIndex", -1) if option.get("inPlayIndex") is not None else -1),
    }


def _card_brief(card: Any) -> dict | None:
    if not isinstance(card, dict):
        return None
    return {
        "id": _card_id(card),
        "hp": int(card.get("hp", 0) or 0),
        "maxHp": int(card.get("maxHp", 0) or 0),
        "en": len(card.get("energyCards") or []),
        "appear": bool(card.get("appearThisTurn")),
        "serial": _serial(card),
        "preEvolution": [
            {"id": _card_id(x), "serial": _serial(x)} for x in (card.get("preEvolution") or [])
        ],
        "tools": [{"id": _card_id(x), "serial": _serial(x)} for x in (card.get("tools") or [])],
    }


def _state(obs: dict) -> dict:
    current = obs.get("current") or {}
    select = obs.get("select") or {}
    your = int(current.get("yourIndex", 0) or 0)
    players = current.get("players") or [{}, {}]
    me = players[your] if 0 <= your < len(players) else {}
    opponent = players[1 - your] if len(players) > 1 else {}

    def player_view(player: dict, private: bool = False) -> dict:
        view = {
            "deck": int(player.get("deckCount", 0) or 0),
            "hand_n": int(player.get("handCount", len(player.get("hand") or [])) or 0),
            "prize_n": len(player.get("prize") or []),
            "active": [_card_brief(c) for c in player.get("active") or []],
            "bench": [_card_brief(c) for c in player.get("bench") or []],
            "discard_ids": [_card_id(c) for c in player.get("discard") or []],
        }
        if private:
            view["hand_ids"] = [_card_id(c) for c in player.get("hand") or []]
        return view

    first = current.get("firstPlayer", -1)
    return {
        "turn": int(current.get("turn", 0) or 0),
        "tac": int(current.get("turnActionCount", 0) or 0),
        "first_rel": -1 if first == -1 else (0 if int(first) == your else 1),
        "flags": {
            "energy": bool(current.get("energyAttached")),
            "supporter": bool(current.get("supporterPlayed")),
            "stadium": bool(current.get("stadiumPlayed")),
            "retreated": bool(current.get("retreated")),
        },
        "stadium": _card_id((current.get("stadium") or [None])[0]) if current.get("stadium") else 0,
        "me": player_view(me, True),
        "op": player_view(opponent, False),
        "context": int(select.get("context", -1) if select.get("context") is not None else -1),
        "stype": int(select.get("type", -1) if select.get("type") is not None else -1),
        "min": int(select.get("minCount", 0) or 0),
        "max": int(select.get("maxCount", 0) or 0),
        "effect": _card_id(select.get("effect")),
        "context_card": _card_id(select.get("contextCard")),
        "remain_damage": int(select.get("remainDamageCounter", 0) or 0),
        "remain_energy": int(select.get("remainEnergyCost", 0) or 0),
    }


def read_deck_csv() -> list[int]:
    path = os.path.join(os.path.dirname(__file__), "deck.csv")
    if not os.path.exists(path):
        path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, encoding="utf-8") as f:
        deck = [int(x.strip()) for x in f if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain 60 cards, got {len(deck)}")
    return deck


def agent(obs_dict: dict) -> list[int]:
    global _HISTORY, _STRATEGIC_MEMORY
    select = obs_dict.get("select")
    if select is None:
        _HISTORY = []
        _STRATEGIC_MEMORY.reset()
        return read_deck_csv()
    raw_options = select.get("option") or []
    if not raw_options:
        return []
    options = [_semantic(obs_dict, option) for option in raw_options]
    state = _state(obs_dict)
    if state.get("turn") == 0 and state.get("context") == 41:
        _STRATEGIC_MEMORY.reset()
    action = apply_manual_guards(obs_dict)
    if action is None:
        action = choose(state, options, _HISTORY, _STRATEGIC_MEMORY)

    # Hard legality guard. It is deterministic and only protects against malformed/novel contexts.
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    valid = (
        minimum <= len(action) <= maximum
        and len(set(action)) == len(action)
        and all(isinstance(i, int) and 0 <= i < len(raw_options) for i in action)
    )
    if not valid:
        action = list(range(min(maximum, len(raw_options))))
        if len(action) < minimum:
            action = list(range(min(minimum, len(raw_options))))

    _STRATEGIC_MEMORY.record([options[i] for i in action])
    for i in action:
        _HISTORY.append(options[i])
    _HISTORY = _HISTORY[-8:]
    return action
