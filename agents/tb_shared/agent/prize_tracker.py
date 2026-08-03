"""Conservative prize-card deduction for search-backed agents.

Gold-medal Starmie lesson: forward search with wrong hidden deck info causes
NOMATCH failures (search finds lines using cards that were actually prized).
Wrong inference is worse than no inference — return unknown when ambiguous.

In-flight effect cards (e.g. Hilda resolving) must be subtracted via
``obs.select.effect`` while the card is neither in hand nor discard.
"""

from __future__ import annotations

from collections import Counter


class PrizeTracker:
    """Track inferred prize cards from public zones + full-deck searches."""

    def __init__(self, decklist: list[int]) -> None:
        self._decklist = list(decklist)
        self._prized: Counter[int] | None = None
        self._last_prize_count: int | None = None
        self._last_hand_by_serial: dict[int, int] = {}

    def update(self, obs, obs_dict: dict | None = None) -> None:
        yi = obs.current.yourIndex
        player = obs.current.players[yi]
        prize_count = len(player.prize)
        hand_by_serial = {
            card.serial: card.id
            for card in player.hand or []
            if card is not None and getattr(card, "serial", None) is not None
        }

        if (
            self._prized is not None
            and self._last_prize_count is not None
            and prize_count < self._last_prize_count
        ):
            taken = self._last_prize_count - prize_count
            card_ids = self._prize_to_hand(obs_dict, yi)
            if len(card_ids) != taken:
                card_ids = [
                    cid
                    for serial, cid in hand_by_serial.items()
                    if serial not in self._last_hand_by_serial
                ]
            if len(card_ids) != taken or not self._remove(card_ids):
                self._prized = None

        self._last_prize_count = prize_count
        self._last_hand_by_serial = hand_by_serial

        if self._prized is not None:
            return

        if obs.select is None or obs.select.deck is None:
            return
        if len(obs.select.deck) != player.deckCount:
            return

        inferred = self._deduce(obs, player, yi)
        if inferred is not None:
            self._prized = inferred

    def _deduce(self, obs, player, player_index: int) -> Counter[int] | None:
        remaining = Counter(self._decklist)

        def sub(card) -> None:
            if card is not None:
                remaining[card.id] -= 1

        for card in obs.select.deck:
            sub(card)
        for card in player.hand or []:
            sub(card)
        for pokemon in list(player.active or []) + list(player.bench or []):
            if pokemon is None:
                continue
            sub(pokemon)
            for c in getattr(pokemon, "preEvolution", None) or []:
                sub(c)
            for c in getattr(pokemon, "energyCards", None) or []:
                sub(c)
            for c in getattr(pokemon, "tools", None) or []:
                sub(c)
        for card in player.discard or []:
            sub(card)
        for card in obs.current.stadium or []:
            if card is not None and getattr(card, "playerIndex", None) == player_index:
                remaining[card.id] -= 1

        effect = getattr(obs.select, "effect", None)
        if effect is not None and getattr(effect, "playerIndex", None) == player_index:
            if remaining.get(effect.id, 0) > 0:
                remaining[effect.id] -= 1

        if any(count < 0 for count in remaining.values()):
            return None
        inferred = Counter({cid: count for cid, count in remaining.items() if count > 0})
        if sum(inferred.values()) != len(player.prize):
            return None
        return inferred

    def _remove(self, card_ids: list[int]) -> bool:
        if self._prized is None:
            return False
        removals = Counter(card_ids)
        if any(self._prized.get(cid, 0) < count for cid, count in removals.items()):
            return False
        self._prized.subtract(removals)
        self._prized += Counter()
        return True

    def _prize_to_hand(self, obs_dict: dict | None, player_index: int) -> list[int]:
        if not isinstance(obs_dict, dict):
            return []
        return [
            log["cardId"]
            for log in obs_dict.get("logs", [])
            if log.get("playerIndex") == player_index
            and log.get("fromArea") in (6, "PRIZE", "Prize")
            and log.get("toArea") in (2, "HAND", "Hand")
            and log.get("cardId") is not None
        ]

    def is_prized(self, card_id: int) -> bool | None:
        """True / False when known, otherwise None."""
        if self._prized is None:
            return None
        return self._prized.get(card_id, 0) > 0

    def prized_cards(self) -> Counter[int] | None:
        if self._prized is None:
            return None
        return self._prized.copy()
