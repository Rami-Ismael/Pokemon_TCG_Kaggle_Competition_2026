from __future__ import annotations

import os
from cg.api import to_observation_class  # preserve official package compatibility
from .handwritten_policy import BUILD_ID, DECK, PLAYER_NAME, decide, reset, view_from_raw


def read_deck_csv() -> list[int]:
    path = os.path.join(os.path.dirname(__file__), "deck.csv")
    if not os.path.exists(path):
        path = "deck.csv"
    if not os.path.exists(path):
        path = "/kaggle_simulations/agent/deck.csv"
    with open(path, "r", encoding="utf-8") as f:
        deck = [int(x.strip()) for x in f if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck.csv must contain exactly 60 cards, got {len(deck)}")
    return deck


def agent(obs_dict: dict) -> list[int]:
    if not obs_dict or obs_dict.get("select") is None:
        reset()
        return read_deck_csv()
    return decide(view_from_raw(obs_dict))
