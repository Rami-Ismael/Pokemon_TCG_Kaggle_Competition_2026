"""v6 lineage self-play game workers: direct-engine games between our own
checkpoints, recorded as corpus-format episodes (rl_pipeline_v6.md §1.2).

The game loop is promotion._play_direct with recording added: every accepted
(observation, action) pair is kept per seat, and the finished game is
assembled into the SAME JSON shape the Kaggle replay corpus uses --
``{"steps": [[seat0, seat1], ...], "rewards": [r0, r1]}`` -- so
``il_dataset.iter_episode_decisions`` consumes a self-play episode unchanged.

The one non-obvious part is the pairing quirk that iterator documents: the
response to ``decisions[i]``'s select is read from ``decisions[i+1]``'s action
field. Assembly therefore shifts each seat's actions one entry later and
appends a trailing entry (duplicate of the last observation) to carry the
final action; the duplicate is never itself emitted as a decision because the
iterator's range stops before it.

Zero external agents (v6 §1.3): every seat is a SamplingPolicy over one of
OUR checkpoint dirs. This module must never grow a code path that seats a
public-pool or heuristic agent.

Process model: the cg/cabt native engine keeps ONE battle state per process,
so parallelism is process-level via a spawn Pool, one sequential game per
worker (same constraint as promotion.py).
"""

from __future__ import annotations

import json
import os
import time

# Direct-engine runaway safety net; the engine ends real games itself.
_MAX_SELECTS = 5000

# Per-process cache (spawn context): policies keyed by checkpoint dir.
_LW: dict = {}


def worker_init(current_ckpt: str, temperature: float = 1.0) -> None:
    import torch

    torch.set_num_threads(1)
    try:
        os.nice(5)  # keep the laptop responsive (standing rule)
    except OSError:
        pass
    _LW["current"] = current_ckpt
    _LW["temperature"] = temperature
    _LW["policies"] = {}
    _get_policy(current_ckpt)


def _get_policy(ckpt: str):
    from .selfplay import SamplingPolicy

    if ckpt not in _LW["policies"]:
        _LW["policies"][ckpt] = SamplingPolicy(
            ckpt, temperature=_LW["temperature"], record=False
        )
    return _LW["policies"][ckpt]


def _assemble_episode(per_seat, rewards: list, info: dict) -> dict:
    """Corpus-format episode from per-seat [(obs, action), ...] lists.

    Action pairing: entry i carries the action that answered entry i-1's
    select (see module docstring); a trailing duplicate-observation entry
    carries the final action.
    """
    entries: list[list[dict]] = [[], []]
    for s in (0, 1):
        dec = per_seat[s]
        for i, (obs, _act) in enumerate(dec):
            entries[s].append(
                {"observation": obs, "action": dec[i - 1][1] if i > 0 else None}
            )
        if dec:
            entries[s].append({"observation": dec[-1][0], "action": dec[-1][1]})
    n = max(len(entries[0]), len(entries[1]))
    filler = {"observation": {}, "action": None}
    steps = [
        [
            entries[0][i] if i < len(entries[0]) else dict(filler),
            entries[1][i] if i < len(entries[1]) else dict(filler),
        ]
        for i in range(n)
    ]
    return {"steps": steps, "rewards": rewards, "info": info}


def play_one_recorded_game(task: dict) -> dict:
    """Worker entry: one direct-engine game, both seats recorded.

    ``task`` keys: game_idx, opponent_ckpt (None = mirror = current weights),
    current_seat, decks ([deck0_ids, deck1_ids]), info (provenance merged
    into the episode).

    Returns episode_json (bytes, corpus format) plus stats; episode_json is
    None when a deck was rejected at battle_start (no game happened).
    """
    from cg.game import battle_finish, battle_select, battle_start

    from .selfplay import _safe_choice

    t0 = time.time()
    game_idx = task["game_idx"]
    cur_seat = task["current_seat"]
    opp_ckpt = task["opponent_ckpt"]
    decks = task["decks"]

    cur = _get_policy(_LW["current"])
    opp = cur if opp_ckpt is None else _get_policy(opp_ckpt)
    agents = [None, None]
    agents[cur_seat] = cur
    agents[1 - cur_seat] = opp

    per_seat: tuple[list, list] = ([], [])
    fallbacks = [0, 0]
    rejects = 0
    forfeiter: int | None = None

    obs, start = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:  # rejected deck: no game, report which seat's deck
        return {
            "game_idx": game_idx,
            "deck_rejected_seat": int(start.errorPlayer),
            "episode_json": None,
            "seconds": time.time() - t0,
        }

    result: int | None = None
    try:
        for _ in range(_MAX_SELECTS):
            cur_view = obs["current"]
            if cur_view["result"] >= 0:
                result = int(cur_view["result"])
                break
            seat = int(cur_view["yourIndex"])
            sel = obs.get("select")
            # Deep copy via JSON round-trip: guarantees the recorded obs is
            # both detached from any engine-reused buffers and serializable.
            obs_copy = json.loads(json.dumps(obs))
            pol = agents[seat]
            fb0 = pol.fallbacks
            try:
                act = pol.agent(obs)
            except Exception:
                act = _safe_choice(sel) if sel else []
            fallbacks[seat] += pol.fallbacks - fb0
            try:
                nxt = battle_select(act)
            except Exception:
                rejects += 1
                act = _safe_choice(sel) if sel else []
                try:
                    nxt = battle_select(act)
                except Exception:
                    # unrecoverable rejection: the acting seat forfeits; its
                    # unanswered decision is dropped (action None convention)
                    forfeiter = seat
                    result = 1 - seat
                    break
            if sel:
                per_seat[seat].append((obs_copy, list(act)))
            obs = nxt
        else:
            result = 2  # safety-net cap: score it a draw
    finally:
        battle_finish()

    # Corpus reward convention (_seat_outcome): positive = win, negative =
    # loss, zero = draw, None = errored seat. [1, 0] would mislabel the
    # loser's whole episode as a draw.
    if result == 2:
        rewards: list = [0, 0]
        outcome = 0.5
    else:
        rewards = [-1, -1]
        rewards[result] = 1
        if forfeiter is not None:
            rewards[forfeiter] = None
        outcome = 1.0 if result == cur_seat else 0.0

    episode = _assemble_episode(per_seat, rewards, dict(task.get("info") or {}))
    return {
        "game_idx": game_idx,
        "deck_rejected_seat": None,
        "episode_json": json.dumps(episode).encode(),
        "outcome": outcome,
        "result": result,
        "forfeiter": forfeiter,
        "decisions": [len(per_seat[0]), len(per_seat[1])],
        "fallbacks": fallbacks,
        "engine_rejects": rejects,
        "seconds": time.time() - t0,
    }
