"""Fast hand-written board value for search leaves and RL shaping."""

from __future__ import annotations

from agent.agent import _get


def _player(current, idx):
    players = _get(current, "players", []) or []
    return players[idx] if 0 <= idx < len(players) else {}


def _prize_count(player) -> int:
    prize = _get(player, "prize", None)
    if prize is not None:
        return len(prize)
    return int(_get(player, "prizeCount", 0) or 0)


def _active_hp(player) -> tuple[int, int]:
    active = (_get(player, "active", []) or [])
    if not active:
        return 0, 0
    mon = active[0]
    hp = int(_get(mon, "hp", 0) or 0)
    max_hp = int(_get(mon, "maxHp", hp) or hp)
    return hp, max_hp


def _bench_strength(player) -> float:
    score = 0.0
    for mon in _get(player, "bench", []) or []:
        if mon is None:
            continue
        hp = int(_get(mon, "hp", 0) or 0)
        energies = len(_get(mon, "energies", []) or [])
        score += hp + 25 * energies
    return score


def board_value(obs_dict: dict) -> float:
    """Scalar board evaluation from our perspective (higher = better).

    Combines prize differential, active/bench HP and energy, deck tempo, and
    turn-based pressure. Safe on partial observations.
    """
    current = obs_dict.get("current") or {}
    your_idx = int(_get(current, "yourIndex", 0) or 0)
    opp_idx = 1 - your_idx if your_idx in (0, 1) else 1

    me = _player(current, your_idx)
    opp = _player(current, opp_idx)

    our_prize = _prize_count(me)
    opp_prize = _prize_count(opp)
    value = 120.0 * (opp_prize - our_prize)

    me_hp, me_max = _active_hp(me)
    opp_hp, opp_max = _active_hp(opp)
    if me_max:
        value += 40.0 * (me_hp / me_max)
    if opp_max:
        value -= 35.0 * (opp_hp / opp_max)
    if opp_hp <= 0 and opp_max > 0:
        value += 80.0

    our_deck = int(_get(me, "deckCount", 0) or 0)
    opp_deck = int(_get(opp, "deckCount", 0) or 0)
    value += 0.4 * (opp_deck - our_deck)

    value += 0.08 * _bench_strength(me)
    value -= 0.06 * _bench_strength(opp)

    turn = int(_get(current, "turn", 0) or 0)
    our_bench = len(_get(me, "bench", []) or [])
    if our_bench == 0:
        # Empty bench is a fast-loss vector on ladder (no promotion after active KO).
        value -= 180.0 if turn <= 25 else 90.0

    if our_prize < opp_prize:
        value += 2.0 * turn
    elif our_prize > opp_prize:
        value -= 1.5 * turn

    if bool(_get(current, "energyAttached", False)):
        value += 8.0
    if bool(_get(current, "supporterUsed", False)):
        value += 5.0

    return value
