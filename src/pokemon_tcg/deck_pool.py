"""Deck pools — the diversity axis of self-play.

Until 2026-08-05 the training env played ONE hardcoded deck
(`selfplay.load_deck()` reading a fixed `DECK_PATH`) on BOTH seats, so three
self-play generations learned a single mirror matchup rather than the game.
This module supplies a pool that is sampled per episode.

WHO SUPPLIES A DECK. When the engine asks a seat for its list (`select is
None`), the learner's side is answered by the env (`PTCGGym.deck`) but the
opponent answers for itself: a checkpoint opponent returns
`SamplingPolicy.deck` (= `load_deck()`, one fixed list) and a module opponent
returns its own bundled `deck.csv`. So widening only the learner's pool leaves
every opponent on the same deck forever. `deck_override_agent` intercepts that
step and substitutes whatever the caller's getter returns.

TWO MODES, both built on that one wrapper:
  independent (default with a pool) — learner deck and opponent deck are two
    separate uniform draws per episode. This is the diversity treatment.
  mirror (`--mirror-deck`) — one draw, both seats. The same-deck-both-seats
    control, and what the 2026-08-04 exploitability sweep needs to keep deck
    MATCHUP out of an exploitability measurement.

SAMPLING IS UNIFORM over pool members (`DeckPool.sample`), deliberately. We
have no ladder meta-share data, so uniform is the honest choice and it targets
worst-case robustness. A policy trained this way has NOT been "trained on the
meta distribution" — meta-weighting is a separate experiment.

Pool members must be cg-LEGAL: an illegal list returns INVALID at deck
submission and hands away every episode it is drawn for. Gate a pool through
`scripts/probe_deck_legality.py` and feed the resulting manifest back as
`@configs/deck_pools/<name>.txt`.
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

    Accepted, in order: an existing path (absolute, cwd-relative, or
    PROJECT_ROOT-relative — manifests store repo-relative paths so they do not
    bake in the worktree that generated them); a `configs/deck_lists/<ref>.csv`
    stem; an `agents/<ref>/deck.csv` agent name.
    """
    for p in (Path(ref), config.PROJECT_ROOT / ref):
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


def deck_digest(path: Path) -> str:
    """Identity of a deck = its SORTED 60-card multiset, not its file bytes.

    Hashing the raw bytes (what this did until 2026-08-05) treats lists that
    differ only in card ORDER or whitespace as distinct. scripts/census_deck_pool.py
    found four such groups among the 35 legality-passed lists — e.g. tb_starmie
    and wmh_froslass are the same 60 cards at jaccard 1.000. Left in, they
    inflate K and hand their archetype extra sampling weight under a flag whose
    whole point is a controlled uniform draw.
    """
    return hashlib.md5(str(sorted(parse_deck(path))).encode()).hexdigest()


def _dedup_by_content(pairs: list[tuple[str, Path]],
                      keep_order: bool = False) -> list[tuple[str, Path]]:
    """Keep one entry per distinct deck content; duplicates would overstate K.

    By default entries are sorted first, so the alphabetically-first name wins
    and the result is machine-independent. `keep_order=True` respects the
    caller's order instead — used by the union spec so a curated
    configs/deck_lists name wins over an agent directory holding the same list.
    """
    seen: dict[str, tuple[str, Path]] = {}
    for name, path in (pairs if keep_order else sorted(pairs)):
        seen.setdefault(deck_digest(path), (name, path))
    return list(seen.values()) if keep_order else sorted(seen.values())


def _decklists() -> list[tuple[str, Path]]:
    return sorted((p.stem, p) for p in DECK_LISTS_DIR.glob("*.csv"))


def _agent_decks() -> list[tuple[str, Path]]:
    return _dedup_by_content(
        [(p.parent.name, p) for p in AGENTS_DIR.glob("*/deck.csv")])


def _expand_spec(spec: str) -> list[tuple[str, Path]]:
    if spec == "all:decklists":
        return _decklists()
    if spec == "all:agents":
        return _agent_decks()
    if spec == "all:decks":
        # Union of both sources, content-deduped across them. Curated lists go
        # first so their human-readable stems survive dedup and become the
        # wrd_/wrm_ stat slugs.
        return _dedup_by_content(_decklists() + _agent_decks(), keep_order=True)
    if spec.startswith("@"):
        manifest = Path(spec[1:])
        if not manifest.exists():
            manifest = config.PROJECT_ROOT / spec[1:]
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
            pinned_digests = {deck_digest(p) for _, p in head}
            rest = [(n, p) for n, p in pairs
                    if deck_digest(p) not in pinned_digests]
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


def deck_override_agent(fn, deck_getter):
    """Wrap an opponent so its deck step returns a caller-chosen deck.

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
