"""Glicko-1 rating system (Glickman, 1999).

Why Glicko-1 instead of plain win-rate or Elo:
  * Elo/win-rate treats every game as equally informative and ignores how
    much evidence we've actually accumulated about an agent's skill.
  * Glicko-1 tracks a rating deviation (RD) alongside the rating -- a
    confidence interval that shrinks as an agent plays more games and grows
    back when an agent goes idle (skipped in a later tournament run). That
    makes ratings comparable across agents with very different game counts,
    and lets ratings persist and compound across repeated benchmark runs
    instead of being recomputed from scratch each time.

Reference: http://www.glicko.net/glicko/glicko.pdf

This is a from-scratch, dependency-free implementation of the *batch*
(rating-period) update: all games an agent played against a given opponent
within one rating period use that opponent's rating/RD as it stood at the
*start* of the period (standard Glicko-1 semantics -- ratings don't leak
within a period).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Q = math.log(10) / 400.0  # 0.0057565...

DEFAULT_RATING = 1500.0
DEFAULT_RD = 350.0  # "unrated" starting deviation
MIN_RD = 30.0  # floor so a long history can't collapse confidence to zero
MAX_RD = 350.0  # ceiling reached by a fully inactive player

# Typical-play constant: RD grows back to ~MAX_RD after roughly this many
# consecutive idle rating periods (Glickman's own example uses ~50).
PERIODS_TO_MAX_RD = 50.0
C = math.sqrt((MAX_RD**2 - MIN_RD**2) / PERIODS_TO_MAX_RD)


@dataclass
class Rating:
    rating: float = DEFAULT_RATING
    rd: float = DEFAULT_RD


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * Q**2 * rd**2 / math.pi**2)


def _e(rating: float, opp_rating: float, opp_rd: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-_g(opp_rd) * (rating - opp_rating) / 400.0))


def apply_inactivity(r: Rating, idle_periods: int = 1) -> Rating:
    """Widen RD for a player who sat out `idle_periods` rating periods."""
    if idle_periods <= 0:
        return r
    new_rd = min(MAX_RD, math.sqrt(r.rd**2 + (C**2) * idle_periods))
    return Rating(rating=r.rating, rd=new_rd)


def update_rating(player: Rating, opponents: list[tuple[Rating, float]]) -> Rating:
    """Batch-update `player` given all their games in one rating period.

    `opponents` is a list of (opponent_rating_at_start_of_period, score) with
    score in {0.0, 0.5, 1.0} (loss/draw/win). Returns the new post-period
    Rating; does not mutate the input.
    """
    if not opponents:
        return apply_inactivity(player, idle_periods=1)

    rd = player.rd
    d_sq_inv = 0.0
    sum_term = 0.0
    for opp, score in opponents:
        g_opp = _g(opp.rd)
        e = _e(player.rating, opp.rating, opp.rd)
        d_sq_inv += (g_opp**2) * e * (1.0 - e)
        sum_term += g_opp * (score - e)
    d_sq_inv *= Q**2

    if d_sq_inv == 0.0:
        return apply_inactivity(player, idle_periods=1)

    d_sq = 1.0 / d_sq_inv
    denom = 1.0 / rd**2 + 1.0 / d_sq
    new_rating = player.rating + (Q / denom) * sum_term
    new_rd = max(MIN_RD, math.sqrt(1.0 / denom))
    return Rating(rating=new_rating, rd=new_rd)


def run_rating_period(
    current: dict[str, Rating],
    games: list[tuple[str, str, float]],
) -> dict[str, Rating]:
    """Run one Glicko-1 rating period over a batch of games.

    `games` is a list of (player_a, player_b, score_a) with score_a in
    {0.0, 0.5, 1.0} from player_a's perspective. Every game is mirrored for
    player_b automatically. Agents present in `current` but absent from
    `games` are treated as idle for one period (RD widens).

    Returns a new dict of post-period Ratings (does not mutate `current`).
    """
    snapshot = dict(current)  # ratings/RDs as of the START of this period
    per_player: dict[str, list[tuple[Rating, float]]] = {}
    played = set()
    for a, b, score_a in games:
        ra = snapshot.setdefault(a, Rating())
        rb = snapshot.setdefault(b, Rating())
        per_player.setdefault(a, []).append((rb, score_a))
        per_player.setdefault(b, []).append((ra, 1.0 - score_a))
        played.add(a)
        played.add(b)

    updated: dict[str, Rating] = {}
    for name, player in snapshot.items():
        if name in per_player:
            updated[name] = update_rating(player, per_player[name])
        else:
            updated[name] = apply_inactivity(player, idle_periods=1)
    return updated
