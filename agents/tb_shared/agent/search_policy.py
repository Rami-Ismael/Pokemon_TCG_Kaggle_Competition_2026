"""Time-budgeted search-augmented OptionScorer with heuristic fallback.

Track A design: keep the full heuristic policy as the default brain and only
layer shallow search on high-leverage card picks (promotion / switch). The prior
evalfn rerank on MAIN skipped EVOLVE/ATTACH and regressed badly vs heuristic.

LucarioSearchScorer merges SearchScorer's cg search_* layer with LucarioScorer
MAIN/meta (668 mu search + SmartBench mirror gains).
"""

from __future__ import annotations

import json
import os
import time

from agent.agent import (
    CTX_SETUP_ACTIVE_POKEMON,
    CTX_SETUP_BENCH_POKEMON,
    CTX_SWITCH,
    CTX_TO_ACTIVE,
    CTX_TO_HAND,
    CTX_TO_DECK,
    CTX_TO_DECK_BOTTOM,
    CTX_ATTACH_FROM,
    HeuristicScorer,
    SEL_CARD,
    load_deck,
)
from agent.prize_tracker import PrizeTracker

try:
    from agent.finish_search import cg_search_pick
except ImportError:  # package-less layout on Kaggle
    from finish_search import cg_search_pick

SEARCH_BUDGET_MS = 200
HIGH_LEVERAGE_CONTEXTS = {
    CTX_TO_ACTIVE,
    CTX_SWITCH,
    CTX_SETUP_ACTIVE_POKEMON,
    CTX_SETUP_BENCH_POKEMON,
}
# Lucario hybrid: search on promote/switch/setup-active (SETUP_BENCH is bench_guard → Lucario).
# Search picks must land in Lucario top-2 or we keep LucarioScorer (mirror guard).
LUCARIO_SEARCH_CONTEXTS = {
    CTX_TO_ACTIVE,
    CTX_SWITCH,
    CTX_SETUP_ACTIVE_POKEMON,
}
SEARCH_GUARD_TOP_K = 2
PRIZE_DECK_CONTEXTS = {
    CTX_TO_HAND,
    CTX_TO_DECK,
    CTX_TO_DECK_BOTTOM,
    CTX_ATTACH_FROM,
}


class _PrizeTrackerMixin:
    """Update PrizeTracker and penalize prized cards in deck search picks."""

    def _init_prize_tracker(self, deck_path: str | None = None) -> None:
        path = deck_path or os.path.join(
            os.path.dirname(__file__), "deck.csv"
        )
        try:
            deck_ids = load_deck(path)
            self._prize_tracker = PrizeTracker(deck_ids) if len(deck_ids) == 60 else None
        except Exception:
            self._prize_tracker = None

    def _update_prize_tracker(self, obs_dict) -> None:
        if self._prize_tracker is None:
            return
        try:
            from cg.api import to_observation_class

            obs = to_observation_class(obs_dict)
            self._prize_tracker.update(obs, obs_dict)
        except Exception:
            pass

    def _prize_adjust_card_score(self, score: float, card_id: int, select) -> float:
        if self._prize_tracker is None or select is None:
            return score
        context = select.get("context")
        if context not in PRIZE_DECK_CONTEXTS:
            return score
        prized = self._prize_tracker.is_prized(card_id)
        if prized is True:
            return score - 1e9
        return score


class _CgSearchMixin:
    """Shared cg search_* wrapper for high-leverage card picks.

    History note (2026-08-11): until now this mixin called lib.SearchBegin
    with an invented C signature; the call always failed, the bare except
    swallowed it, and every "search" decision was the heuristic fallback.
    It now routes through finish_search.cg_search_pick (cg.api seam). Past
    results measured through this path never ran a search.
    """

    def _init_search(
        self,
        budget_ms: float = SEARCH_BUDGET_MS,
        *,
        search_contexts: set | None = None,
        deck_path: str | None = None,
    ) -> None:
        self._budget_ms = budget_ms
        self._search_contexts = search_contexts or HIGH_LEVERAGE_CONTEXTS
        # The real search needs the 60-card list to fill hidden zones.
        path = deck_path or os.path.join(os.path.dirname(__file__), "deck.csv")
        try:
            deck = load_deck(path)
            self._search_deck = deck if len(deck) == 60 else None
        except Exception:
            self._search_deck = None

    def _audit_search(self, event: str, **payload) -> None:
        path = os.environ.get("SEARCH_AUDIT_LOG")
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": event, **payload}, sort_keys=True) + "\n")
        except Exception:
            pass

    def _try_search(self, obs_dict, select, options) -> list[int] | None:
        if not options:
            return None
        try:
            context = select.get("context")
            eligible = (
                select.get("type") == SEL_CARD
                and context in self._search_contexts
                and int(select.get("minCount", 1) or 0) <= 1
            )
            self._audit_search(
                "try_search",
                context=context,
                eligible=eligible,
                has_begin=bool(obs_dict.get("search_begin_input")),
                options=len(options),
                select_type=select.get("type"),
            )
            if eligible:
                deadline = time.monotonic() + self._budget_ms / 1000.0
                return self._cg_api_search(obs_dict, options, deadline)
        except Exception:
            pass
        return None

    def _cg_api_search(self, obs_dict, options, deadline) -> list[int] | None:
        """One-ply lookahead via finish_search.cg_search_pick; None on failure."""
        budget_ms = (deadline - time.monotonic()) * 1000.0
        if budget_ms <= 0:
            self._audit_search("search_result", fired=False, reason="engine_or_budget")
            return None
        return cg_search_pick(
            obs_dict,
            options,
            self._search_deck,
            budget_ms=budget_ms,
            audit=self._audit_search,
        )


class SearchScorer(_CgSearchMixin, _PrizeTrackerMixin, HeuristicScorer):
    """Heuristic baseline + optional cg search_* on promotion/switch picks."""

    def __init__(
        self,
        rng=None,
        budget_ms: float = SEARCH_BUDGET_MS,
        deck_path: str | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self._init_search(budget_ms, deck_path=deck_path)
        self._init_prize_tracker(deck_path)
        self._active_select = None

    def choose(self, obs_dict, select, current, options):
        self._update_prize_tracker(obs_dict)
        self._active_select = select
        if not options:
            return []
        picked = self._try_search(obs_dict, select, options)
        if picked is not None:
            return picked
        return super().choose(obs_dict, select, current, options)

    def _card_id_score(self, card_id, current):
        base = super()._card_id_score(card_id, current)
        return self._prize_adjust_card_score(base, card_id, self._active_select)


class LucarioSearchScorer(_CgSearchMixin, _PrizeTrackerMixin):
    """Lucario meta MAIN + cg search_* on setup/switch/to-active picks."""

    def __init__(
        self,
        rng=None,
        deck_path: str | None = None,
        budget_ms: float = SEARCH_BUDGET_MS,
    ) -> None:
        from agent.lucario_policy import LucarioScorer

        path = deck_path
        self._lucario = LucarioScorer(rng=rng, deck_path=path)
        self._init_search(
            budget_ms, search_contexts=LUCARIO_SEARCH_CONTEXTS, deck_path=path
        )
        self._init_prize_tracker(path)
        self._active_select = None

    def choose(self, obs_dict, select, current, options):
        self._update_prize_tracker(obs_dict)
        self._active_select = select
        if not options:
            return []
        lucario_pick = self._lucario.choose(obs_dict, select, current, options)
        picked = self._try_search(obs_dict, select, options)
        if picked is None or len(picked) != 1:
            return lucario_pick
        ranked = self._lucario.rank_options(obs_dict, select, current, options)
        if not ranked:
            return lucario_pick
        top_k = set(ranked[: min(SEARCH_GUARD_TOP_K, len(ranked))])
        if picked[0] in top_k:
            self._audit_search("lucario_guard", accepted=True, pick=picked[0], top_k=sorted(top_k))
            return picked
        self._audit_search("lucario_guard", accepted=False, pick=picked[0], top_k=sorted(top_k))
        return lucario_pick
