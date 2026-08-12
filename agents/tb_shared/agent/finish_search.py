"""cg search_* helpers built on the real cg.api wrapper.

History note (2026-08-11): earlier versions of this module and of
search_policy._CgSearchMixin called lib.SearchBegin directly with an
invented C signature (battle_ptr, begin_input, remaining_ms, option-index
and output arrays, n_opts). The real signature (cg/sim.py argtypes) takes
an AgentStart() handle, the search_begin_input string plus its length, and
six predicted-hidden-card arrays. Every call failed, the bare except
swallowed the error, and callers got None — both search paths were silent
no-ops. Any past result measured "with search" through try_cg_search or
SearchScorer therefore never ran a search.

Everything here now goes through cg.api.search_begin/search_step — the
same seam lucario_mcts_runtime.py and mega_lucario/agent_core_improved.py
already use correctly. Hidden zones (decks, prizes, opponent hand/active)
are filled with samples from the known 60-card lists, exactly as the
working runtime does.

Audit: every attempt appends a JSON line to $SEARCH_AUDIT_LOG (or the
audit_log/audit arguments) with event "search_result" and engine
"cg.api", so a live game can prove the search actually fired.
"""

from __future__ import annotations

import json
import os
import time

_WIN = 1e9
_LOSS = -1e9

_CARD_TABLE: dict | None = None


def _audit(event: str, audit_log: str | None = None, **payload) -> None:
    path = audit_log or os.environ.get("SEARCH_AUDIT_LOG")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")
    except Exception:
        pass


def _card_table() -> dict:
    global _CARD_TABLE
    if _CARD_TABLE is None:
        from cg.api import all_card_data

        _CARD_TABLE = {c.cardId: c for c in all_card_data()}
    return _CARD_TABLE


def _deck_basic_ids(deck: list[int]) -> list[int]:
    from cg.api import CardType

    table = _card_table()
    return [
        cid
        for cid in deck
        if (data := table.get(cid)) is not None
        and data.cardType == CardType.POKEMON
        and getattr(data, "basic", False)
    ]


def _sample_hidden(rng, deck: list[int], n: int) -> list[int]:
    if n <= 0 or not deck:
        return []
    if n <= len(deck):
        return rng.sample(deck, n)
    return (deck * (n // len(deck) + 1))[:n]


def _sample_hidden_deck(rng, deck: list[int], n: int) -> list[int]:
    """Deck belief for search_begin: cg requires >=1 Basic during setup."""
    if n <= 0:
        return []
    basics = _deck_basic_ids(deck)
    if not basics:
        return _sample_hidden(rng, deck, n)
    if n == 1:
        return [rng.choice(basics)]
    sample = _sample_hidden(rng, deck, n)
    if any(cid in basics for cid in sample):
        return sample
    sample[rng.randrange(n)] = rng.choice(basics)
    return sample


def _stub_pokemon_id(deck: list[int]) -> int:
    from cg.api import CardType

    basics = _deck_basic_ids(deck)
    if basics:
        return basics[0]
    table = _card_table()
    for cid in deck:
        data = table.get(cid)
        if data is not None and data.cardType == CardType.POKEMON:
            return cid
    return deck[0] if deck else 677


def _stub_energy_id(deck: list[int]) -> int:
    from cg.api import CardType

    table = _card_table()
    for cid in deck:
        data = table.get(cid)
        if data is not None and data.cardType in (
            CardType.BASIC_ENERGY,
            CardType.SPECIAL_ENERGY,
        ):
            return cid
    return 6


def _begin_search(obs, your_deck: list[int], opp_deck: list[int]):
    """cg.api.search_begin with sampled beliefs for every hidden zone."""
    import random

    from cg.api import search_begin

    rng = random.Random()
    state = obs.current
    me = state.yourIndex
    my_ps = state.players[me]
    opp_ps = state.players[1 - me]
    active = opp_ps.active
    return search_begin(
        obs,
        your_deck=_sample_hidden_deck(rng, your_deck, my_ps.deckCount),
        your_prize=_sample_hidden(rng, your_deck, len(my_ps.prize)),
        opponent_deck=_sample_hidden_deck(rng, opp_deck, opp_ps.deckCount),
        opponent_prize=_sample_hidden(rng, opp_deck, len(opp_ps.prize)),
        opponent_hand=[_stub_energy_id(opp_deck)] * opp_ps.handCount,
        opponent_active=(
            [_stub_pokemon_id(opp_deck)]
            if len(active) > 0 and active[0] is None
            else []
        ),
    )


def _board_value(cur, me: int) -> float:
    """Terminal ±inf, else prize differential with a damage tie-break."""
    if cur.result >= 0:
        return _WIN if cur.result == me else _LOSS
    my_ps = cur.players[me]
    opp_ps = cur.players[1 - me]
    # Taking a prize removes it from MY pile, so fewer own prizes is good.
    val = 100.0 * (len(opp_ps.prize) - len(my_ps.prize))
    for ps, sign in ((opp_ps, 1.0), (my_ps, -1.0)):
        for poke in list(ps.active) + list(ps.bench):
            if poke is not None:
                val += sign * float(poke.maxHp - poke.hp) * 0.1
    return val


def _probe_win(state, me: int, deadline: float, *, max_depth: int = 6,
               max_nodes: int = 256) -> bool:
    """DFS over our own consecutive selects for a line ending result == me.

    Stops at the opponent's select (our turn ended without the win), at the
    deadline, or at the node cap. Mandatory multi-picks (minCount > 1) are
    followed with the first minCount indices only.
    """
    from cg.api import search_step

    nodes = 0

    def dfs(node, depth: int) -> bool:
        nonlocal nodes
        cur = node.observation.current
        if cur.result >= 0:
            return cur.result == me
        if depth >= max_depth or cur.yourIndex != me:
            return False
        sel = node.observation.select
        opts = (sel.option or []) if sel is not None else []
        if not opts:
            return False
        min_c = int(sel.minCount or 0)
        if min_c > 1:
            picks = [list(range(min(min_c, len(opts))))]
        else:
            picks = [[j] for j in range(len(opts))]
        for pick in picks:
            if nodes >= max_nodes or time.monotonic() >= deadline:
                return False
            nodes += 1
            try:
                child = search_step(node.searchId, pick)
            except (ValueError, RuntimeError):
                continue
            if dfs(child, depth + 1):
                return True
        return False

    return dfs(state, 0)


def cg_search_pick(
    obs_dict: dict,
    options: list,
    your_deck: list[int] | None,
    *,
    opp_deck: list[int] | None = None,
    budget_ms: float = 200.0,
    require_win: bool = False,
    audit=None,
) -> list[int] | None:
    """Pick one option index via the engine's real search seam; None = no verdict.

    One-ply lookahead over the engine's own forward model: search_begin with
    sampled hidden-info beliefs, search_step each candidate option, score the
    resulting state with _board_value. With require_win=True (finish mode)
    a child only counts if it terminates as our win or _probe_win finds a
    same-turn winning line through it.

    `options` must be the engine's full select["option"] array — the option
    index passed to search_step is the engine's index, not a subset's.
    """
    log = audit if audit is not None else _audit
    if not options or not your_deck:
        log("search_result", fired=False, reason="no_options_or_deck", engine="cg.api")
        return None
    if not obs_dict.get("search_begin_input"):
        log("search_result", fired=False, reason="missing_begin_input", engine="cg.api")
        return None
    deadline = time.monotonic() + budget_ms / 1000.0
    try:
        from cg.api import search_end, search_step, to_observation_class
    except Exception:
        log("search_result", fired=False, reason="cg_import_failed", engine="cg.api")
        return None
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        log("search_result", fired=False, reason="obs_parse_failed", engine="cg.api")
        return None
    sel = obs.select
    if sel is None or obs.current is None:
        log("search_result", fired=False, reason="no_select_state", engine="cg.api")
        return None
    if int(sel.minCount or 0) > 1 or int(sel.maxCount or 0) < 1:
        log("search_result", fired=False, reason="multi_pick_select", engine="cg.api")
        return None
    me = obs.current.yourIndex
    try:
        root = _begin_search(obs, list(your_deck), list(opp_deck or your_deck))
    except Exception as exc:
        log(
            "search_result",
            fired=False,
            reason="search_begin_failed",
            error=f"{type(exc).__name__}: {exc}"[:200],
            engine="cg.api",
        )
        return None
    best_i: int | None = None
    best_v: float | None = None
    win_i: int | None = None
    evaluated = 0
    try:
        for i in range(len(options)):
            if time.monotonic() >= deadline:
                break
            try:
                child = search_step(root.searchId, [i])
            except (ValueError, RuntimeError):
                continue
            evaluated += 1
            v = _board_value(child.observation.current, me)
            if require_win and v < _WIN and _probe_win(child, me, deadline):
                v = _WIN
            if best_v is None or v > best_v:
                best_i, best_v = i, v
            if v >= _WIN:
                win_i = i
                break
    finally:
        try:
            search_end()
        except Exception:
            pass
    if require_win:
        fired = win_i is not None
        log(
            "search_result",
            fired=fired,
            reason=None if fired else "no_win_found",
            pick=win_i,
            evaluated=evaluated,
            options=len(options),
            engine="cg.api",
        )
        return [win_i] if fired else None
    if best_i is None:
        log(
            "search_result",
            fired=False,
            reason="no_child_evaluated",
            evaluated=evaluated,
            options=len(options),
            engine="cg.api",
        )
        return None
    log(
        "search_result",
        fired=True,
        pick=best_i,
        value=round(best_v, 3),
        evaluated=evaluated,
        options=len(options),
        engine="cg.api",
    )
    return [best_i]


def try_cg_search(
    obs_dict: dict,
    options: list,
    *,
    your_deck: list[int] | None = None,
    budget_ms: float = 200,
    audit_log: str | None = None,
) -> list[int] | None:
    """Finish-mode lethal verification.

    Returns [option_index] only when the engine's search finds a same-turn
    winning line through that option; None otherwise (caller keeps its
    heuristic pick). Callers must pass their deck list (`your_deck`) — the
    search cannot fill hidden zones without it.
    """
    if your_deck is None:
        _audit(
            "search_result",
            audit_log=audit_log,
            fired=False,
            reason="no_options_or_deck",
            engine="cg.api",
        )
        return None

    def log(event: str, **payload) -> None:
        _audit(event, audit_log=audit_log, **payload)

    return cg_search_pick(
        obs_dict,
        options,
        your_deck,
        budget_ms=budget_ms,
        require_win=True,
        audit=log,
    )
