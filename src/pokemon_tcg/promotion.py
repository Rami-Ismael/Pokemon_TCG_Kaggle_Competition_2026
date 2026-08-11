"""Checkpoint-promotion gate for KL-anchored self-play (the ratchet).

Plays live pi_theta vs the frozen reference pi_ref as MIRRORED PAIRS -- each
pair is two games with the seat order swapped, the same first-player-advantage
cancellation scripts/benchmark_agents.py::play_match uses. Both sides SAMPLE
at temperature 1.0 (the distribution PPO actually optimizes).

THE GATE'S DECKS MUST MATCH THE TRAINING DISTRIBUTION. With no deck pool both
sides pilot selfplay.load_deck, as before. With a pool, each game draws the two
seats' decks the same way the env does (independently, or as one draw under
mirror_deck), seeded by game index so a gate is reproducible. Left on the
single deck it would ratchet the KL reference on one matchup while the policy
trains on many -- promoting whichever checkpoint happens to be best at Mega
Lucario ex, which is the failure this whole change exists to end.

Games run in spawned worker processes: the cg/cabt native engine keeps ONE
battle state per process (the train_ppo_puffer.py singleton constraint), so
parallelism is process-level, one sequential game at a time per worker. Both
checkpoints are read from DISK dirs -- the caller snapshots the live actor
first -- so workers never touch the MPS learner.

Decision rule (Orbit Wars 1st place / Lux AI S1): promote iff the live model
wins strictly more than `threshold` of DECISIVE games (draws excluded; a
crashed seat counts as that seat's loss, matching play_match / winning_agent).
Fewer than MIN_DECISIVE decisive games is an automatic no-promote: a gate
that mostly draws has not shown superiority.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

# Below this many decisive games the win-rate estimate is too noisy to
# ratchet on (20 of the default 100 games; binomial sigma at p=0.7, n=20 is
# ~0.10, already generous).
MIN_DECISIVE = 20

# Per-process cache for the eval workers (spawn context).
_EW: dict = {}


def _eval_worker_init(live_dir: str, ref_dir: str,
                      deck_pool=None, mirror_deck: bool = False,
                      seed: int = 0) -> None:
    import torch

    torch.set_num_threads(1)
    from kaggle_environments import make as kaggle_make  # slow import, once per worker

    from .deck_pool import deck_override_agent
    from .selfplay import SamplingPolicy, as_env_agent

    _EW["make"] = kaggle_make
    _EW["decks"] = {0: None, 1: None}  # per-game draw, set by _play_eval_game
    live = as_env_agent(
        SamplingPolicy(live_dir, temperature=1.0, record=False).agent
    )
    ref = as_env_agent(
        SamplingPolicy(ref_dir, temperature=1.0, record=False).agent
    )
    if deck_pool is not None:
        # Seat 0 / seat 1 getters, not live/ref getters: the deck belongs to
        # the SEAT, and mirrored pairs swap which policy sits in which seat.
        live = deck_override_agent(live, lambda: _EW["decks"][_EW["live_seat"]])
        ref = deck_override_agent(ref, lambda: _EW["decks"][1 - _EW["live_seat"]])
    _EW["live"], _EW["ref"] = live, ref
    _EW["pool"], _EW["mirror_deck"], _EW["seed"] = deck_pool, mirror_deck, seed


def _play_eval_game(task) -> float:
    """One game; returns the LIVE side's outcome in {1.0, 0.5, 0.0}.

    ``task`` is (game_idx, live_seat). A ``None`` reward means that seat
    crashed/errored out = that seat's loss (same convention as
    benchmark_agents.play_match / selfplay.play_one_game).
    """
    import random as _random

    game_idx, live_seat = task
    _EW["live_seat"] = live_seat
    if _EW["pool"] is not None:
        # Seeded per game index: the two games of a mirrored PAIR (indices
        # 2k, 2k+1) get the same rng stream, so the pair really is the same
        # deck matchup with the seats swapped rather than two unrelated games.
        # STRING seed, not a tuple -- random.Random rejects tuples (TypeError,
        # which in a worker process is a crashed seat = a silent gate loss).
        rng = _random.Random(f"{_EW['seed']}:{game_idx // 2}")
        _, d0 = _EW["pool"].sample(rng)
        d1 = d0 if _EW["mirror_deck"] else _EW["pool"].sample(rng)[1]
        _EW["decks"] = {0: d0, 1: d1}
    agents = [None, None]
    agents[live_seat] = _EW["live"]
    agents[1 - live_seat] = _EW["ref"]
    env = _EW["make"]("cabt")
    trace = env.run(agents)
    final = trace[-1]
    rewards = [final[i]["reward"] for i in range(2)]
    scored = [float("-inf") if r is None else r for r in rewards]
    mine, theirs = scored[live_seat], scored[1 - live_seat]
    return 0.5 if mine == theirs else (1.0 if mine > theirs else 0.0)


def evaluate_gate(
    live_dir: str | Path,
    ref_dir: str | Path,
    pairs: int = 50,
    workers: int = 8,
    threshold: float = 0.70,
    deck_pool=None,
    mirror_deck: bool = False,
    seed: int = 0,
) -> dict:
    """Play ``pairs`` mirrored pairs (2*pairs games) of live vs ref.

    ``deck_pool`` (a DeckPool) makes the gate sample decks exactly as the
    training env does; pass the SAME pool and ``mirror_deck`` the run trains
    with, or the ratchet selects for a matchup the policy is not training on.

    Returns a decision dict: wins/losses/draws are from the LIVE side's
    perspective, ``win_rate`` is over decisive games only, and ``promote``
    is the ratchet verdict (win_rate strictly > threshold AND at least
    MIN_DECISIVE decisive games).
    """
    tasks = [(i, i % 2) for i in range(2 * pairs)]  # mirrored pairs
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=min(workers, len(tasks)),
        initializer=_eval_worker_init,
        initargs=(str(live_dir), str(ref_dir), deck_pool, mirror_deck, seed),
    ) as pool:
        outcomes = pool.map(_play_eval_game, tasks)
    verdict = gate_verdict(outcomes, threshold)
    verdict.update({
        "live": str(live_dir),
        "ref": str(ref_dir),
        "pairs": pairs,
        "deck_pool_k": len(deck_pool) if deck_pool is not None else 1,
        "mirror_deck": mirror_deck,
        "seconds": time.time() - t0,
    })
    return verdict


def gate_verdict(outcomes: list[float], threshold: float = 0.70) -> dict:
    """The ratchet's decision rule over per-game live outcomes {1.0, 0.5, 0.0}.

    Promote iff live won STRICTLY more than ``threshold`` of decisive
    (non-draw) games AND at least MIN_DECISIVE games were decisive.
    """
    wins = sum(1 for o in outcomes if o == 1.0)
    losses = sum(1 for o in outcomes if o == 0.0)
    draws = sum(1 for o in outcomes if o == 0.5)
    decisive = wins + losses
    win_rate = wins / decisive if decisive else 0.0
    return {
        "games": len(outcomes),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "decisive": decisive,
        "win_rate": win_rate,
        "threshold": threshold,
        "min_decisive": MIN_DECISIVE,
        "promote": decisive >= MIN_DECISIVE and win_rate > threshold,
    }
