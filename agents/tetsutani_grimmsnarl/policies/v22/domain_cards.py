from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_BASE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CardProfile:
    card_id: int
    name: str
    stage: str
    rule: str
    hp: int
    text: str
    moves: str

    @property
    def is_rulebox(self) -> bool:
        return self.rule not in ("", "n/a")

    @property
    def is_hand_size_punisher(self) -> bool:
        return "for each card in your hand" in self.text

    @property
    def blocks_ex_attack_damage(self) -> bool:
        return (
            "prevent all damage done to this pokémon by attacks from your opponent’s pokémon {ex}"
            in self.text
        )

    @property
    def is_repeatable_draw_engine(self) -> bool:
        return "once during your turn" in self.text and (
            "draw 2 cards" in self.text
            or "draw 3 cards" in self.text
            or "draw cards" in self.text
        )

    @property
    def is_repeatable_search_engine(self) -> bool:
        return "once during your turn" in self.text and "search your deck" in self.text

    @property
    def is_energy_engine(self) -> bool:
        return "once during your turn" in self.text and (
            "attach a basic energy" in self.text or "attach up to" in self.text
        )

    @property
    def is_spread_engine(self) -> bool:
        return "damage counter" in self.text or "benched pokémon" in self.text


@lru_cache(maxsize=1)
def _profiles() -> dict[int, CardProfile]:
    rows: dict[int, dict[str, object]] = {}
    with (_BASE.parent.parent / "EN_Card_Data.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            card_id = int(row["Card ID"])
            entry = rows.setdefault(
                card_id,
                {
                    "name": row["Card Name"],
                    "stage": row["Stage (Pokémon)/Type (Energy and Trainer)"],
                    "rule": row["Rule"],
                    "hp": int(row["HP"]) if str(row["HP"]).isdigit() else 0,
                    "text": [],
                    "moves": [],
                },
            )
            entry["text"].append((row["Effect Explanation"] or "").lower())
            entry["moves"].append((row["Move Name"] or "").lower())
    return {
        card_id: CardProfile(
            card_id=card_id,
            name=str(data["name"]),
            stage=str(data["stage"]),
            rule=str(data["rule"]),
            hp=int(data["hp"]),
            text=" ".join(data["text"]),
            moves=" ".join(data["moves"]),
        )
        for card_id, data in rows.items()
    }


def profile(card_id: int) -> CardProfile:
    card_id = int(card_id or 0)
    return _profiles().get(card_id, CardProfile(card_id, "", "", "", 0, "", ""))
