"""Deck pools — the swept axis of the deck-diversity exploitability study
(notes/experiments/2026-08-04-deck-diversity-exploitability-sweep.md).

Until now the training/eval env played ONE hardcoded deck
(`selfplay.load_deck()` reading a fixed `DECK_PATH`), which is why the v1
exploiter diagnostic is scoped to the Mega Lucario ex mirror. This module adds
a pool that is sampled per episode.

MIRROR SEMANTICS, and why they are not optional. A module opponent supplies its
own deck when the engine asks for one (`select is None`), so widening only the
learner's pool would silently produce *cross-deck* matchups — the learner on a
sampled deck, the opponent forever on its bundled one. That measures deck
matchup, not policy exploitability, and it is exactly the confound the v1 report
controlled for by using one deck on both sides. `mirror_deck_agent` therefore
intercepts the opponent's deck step and substitutes the episode's deck, so both
seats always play the same list. A pool of size 1 holding il_agent's own deck
reproduces the v1 setup exactly.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from . import config

DECK_LISTS_DIR = config.PROJECT_ROOT / "configs" / "deck_lists"
AGENTS_DIR = config.PROJECT_ROOT / "agents"

DECK_SIZE = 60


def parse_deck(path: Path) -> list[int]:
    """Same parse as selfplay.load_deck: whitespace-separated ids, first 60."""
    return [int(x) for x in path.read_text().split() if x.strip()][:DECK_SIZE]


def resolve_deck_ref(ref: str) -> tuple[str, Path]:
    """Resolve a deck reference to (name, path).

    Accepted, in order: an existing path; a `configs/deck_lists/<ref>.csv`
    stem; an `agents/<ref>/deck.csv` agent name.
    """
    p = Path(ref)
    if p.exists() and p.is_file():
        name = p.parent.name if p.name == "deck.csv" else p.stem
        return name, p
    cand = DECK_LISTS_DIR / f"{ref}.csv"
    if cand.exists():
        return ref, cand
    cand = AGENTS_DIR / ref / "deck.csv"
    if cand.exists():
        return ref, cand
    raise FileNotFoundError(
        f"deck ref {ref!r} not found as a path, a configs/deck_lists stem, "
        f"or an agents/<name>/deck.csv")


def _dedup_by_content(pairs: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    """Keep one entry per distinct deck content, first name wins.

    agents/ holds 47 deck.csv files but only 33 distinct lists; counting
    duplicates as pool members would overstate K.
    """
    seen: dict[str, tuple[str, Path]] = {}
    for name, path in sorted(pairs):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        seen.setdefault(digest, (name, path))
    return sorted(seen.values())


def _expand_spec(spec: str) -> list[tuple[str, Path]]:
    if spec == "all:decklists":
        return sorted((p.stem, p) for p in DECK_LISTS_DIR.glob("*.csv"))
    if spec == "all:agents":
        return _dedup_by_content(
            [(p.parent.name, p) for p in AGENTS_DIR.glob("*/deck.csv")])
    if spec.startswith("@"):
        manifest = Path(spec[1:])
        refs = [ln.split("#", 1)[0].strip()
                for ln in manifest.read_text().splitlines()]
        return [resolve_deck_ref(r) for r in refs if r]
    return [resolve_deck_ref(r) for r in spec.split(",") if r.strip()]


class DeckPool:
    """An ordered, named set of decks sampled from once per episode.

    Order is sorted by name so a given (spec, seed) picks the same decks on
    any machine — nested pools in the sweep must be reproducible.
    """

    def __init__(self, entries: list[tuple[str, list[int]]], spec: str = "") -> None:
        if not entries:
            raise ValueError("empty deck pool")
        self.entries = entries
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: str, limit: int | None = None,
                  seed: int | None = None,
                  pin: list[str] | None = None) -> "DeckPool":
        """Build a pool.

        `limit` (K) takes a subset; `seed` chooses WHICH subset. Subsets are
        NESTED: the seed fixes one permutation of the whole spec and K takes a
        prefix of it, so P_4 ⊂ P_16 ⊂ P_33 for a given seed. Nesting is what
        makes K the only variable that moves between arms of the sweep — an
        independent `random.sample` per K would confound "more decks" with
        "different decks".

        `pin` forces refs to the front of that permutation, so K=1 can be held
        to a chosen deck (the sweep pins il_agent's, its K=1 baseline).
        """
        pairs = _expand_spec(spec)
        if not pairs:
            raise ValueError(f"deck pool spec {spec!r} matched no decks")
        if pin:
            # Match by deck CONTENT, not filename: 'all:agents' is deduped by
            # content and keeps one name per distinct list, so a pinned ref
            # (e.g. il_agent) is often present under some other agent's name.
            # The pinned entry replaces that duplicate and keeps its own name.
            head = [resolve_deck_ref(r) for r in pin]
            pinned_digests = {hashlib.md5(p.read_bytes()).hexdigest()
                              for _, p in head}
            rest = [(n, p) for n, p in pairs
                    if hashlib.md5(p.read_bytes()).hexdigest() not in pinned_digests]
        else:
            head, rest = [], list(pairs)
        if seed is not None:
            random.Random(seed).shuffle(rest)
        pairs = head + rest
        if limit is not None:
            if limit > len(pairs):
                raise ValueError(f"K={limit} exceeds pool size {len(pairs)}")
            pairs = pairs[:limit]
        entries = [(name, parse_deck(path)) for name, path in pairs]
        short = [e for e in entries if len(e[1]) != DECK_SIZE]
        if short:
            raise ValueError(
                f"decks with != {DECK_SIZE} cards: {[n for n, _ in short]}")
        return cls(entries, spec=spec)

    def sample(self, rng: random.Random) -> tuple[str, list[int]]:
        return self.entries[rng.randrange(len(self.entries))]

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"DeckPool(K={len(self)}, {', '.join(self.names)})"


def mirror_deck_agent(fn, deck_getter):
    """Wrap an opponent so its deck step returns the episode's pooled deck.

    `deck_getter` is a zero-arg callable read at deck-submission time, not a
    fixed list, because the deck changes every episode. The wrapper is a plain
    1-arg function: kaggle_environments inspects `__code__.co_argcount`, and a
    bound method would be called with (obs, config) and crash the seat (see
    selfplay.as_env_agent).
    """

    def agent_fn(obs):
        if obs.get("select") is None:
            return list(deck_getter())
        return fn(obs)

    return agent_fn
