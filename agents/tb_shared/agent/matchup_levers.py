"""Opponent archetype detection + per-matchup lever specs (Phase 2, R11).

Visible-board signatures only — no hidden hand inference.
Consumed by lucario_policy / rule_core; one lever change at a time, re-gate each.

See data/MATCHUP_PLAYBOOK.md for strategy research and lever rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Signatures: any ID on opponent active/bench --------------------------------

ARCHETYPE_SIGNATURES: dict[str, frozenset[int]] = {
    "lucario_mirror": frozenset({677, 678, 673, 674, 676, 675}),
    "alakazam_psychic": frozenset({741, 742, 743, 245}),
    "dragapult_psychic": frozenset({119, 120, 121}),
    "iono_lightning": frozenset({265, 268, 269, 270, 271}),
    "abomasnow_water": frozenset({722, 723}),
    "kyogre_water": frozenset({721}),
    "trevenant_control": frozenset({878, 879, 304}),
    "crustle_wall": frozenset({344, 345}),
    "dunsparce_engine": frozenset({65, 66}),
}

# Strongest signal wins when multiple match (ordered priority).
ARCHETYPE_PRIORITY: tuple[str, ...] = (
    "crustle_wall",
    "trevenant_control",
    "abomasnow_water",
    "kyogre_water",
    "iono_lightning",
    "alakazam_psychic",
    "dragapult_psychic",
    "lucario_mirror",
    "dunsparce_engine",
)


def detect_archetypes(opp_board_ids: set[int] | frozenset[int]) -> frozenset[str]:
    """Tags visible on opponent board. Multiple tags allowed."""
    ids = set(opp_board_ids)
    found = {tag for tag, sig in ARCHETYPE_SIGNATURES.items() if ids & sig}
    return frozenset(found)


def primary_archetype(opp_board_ids: set[int] | frozenset[int]) -> str | None:
    """Single best tag for lever dispatch (highest-priority match)."""
    found = detect_archetypes(opp_board_ids)
    for tag in ARCHETYPE_PRIORITY:
        if tag in found:
            return tag
    return None


@dataclass(frozen=True)
class LeverDeltas:
    """Additive score bonuses for lucario_policy contexts (tune one archetype at a time)."""

    # setup / bench
    prefer_solrock_open: float = 0.0
    prefer_riolu_open: float = 0.0
    bench_hariyama_bonus: float = 0.0
    # attack plan
    solrock_vs_single_prize: float = 0.0
    hariyama_vs_ex: float = 0.0
    mega_brave_vs_ex: float = 0.0
    skip_mega_brave_vs_bulk_single_prize: float = 0.0
    # supporters
    boss_orders: float = 0.0
    switch_after_mega_brave: float = 0.0
    premium_power_pro: float = 0.0
    lillie_early: float = 0.0
    gravity_mountain: float = 0.0
    # target selection
    gust_setup_pokemon: float = 0.0
    avoid_ko_trevenant_setup: float = 0.0


# Board IDs we want to gust before they evolve (setup pressure).
GUST_SETUP_IDS = frozenset({119, 120, 677, 722, 741, 878})  # Dreepy/Drakloak, Riolu, Snover, Abra, Phantump
TREVENANT_REVENGE_SETUP_IDS = frozenset({878, 304})  # Phantump, Hop's Snorlax
SNOVER_ID = 722
# Values are starting hypotheses — each row must pass L1 re-gate before trust.
LUCARIO_LEVERS: dict[str, LeverDeltas] = {
    "lucario_mirror": LeverDeltas(
        prefer_riolu_open=1.0,
        mega_brave_vs_ex=100.0,
        boss_orders=800.0,
        gust_setup_pokemon=400.0,
        premium_power_pro=500.0,
    ),
    "alakazam_psychic": LeverDeltas(
        prefer_solrock_open=2.0,
        solrock_vs_single_prize=200.0,
        skip_mega_brave_vs_bulk_single_prize=500.0,
        boss_orders=600.0,
        gust_setup_pokemon=500.0,  # Abra before evolve
    ),
    "dragapult_psychic": LeverDeltas(
        solrock_vs_single_prize=150.0,
        boss_orders=900.0,  # Tuned 2026-06-26: +200 (was 700)
        gust_setup_pokemon=600.0,  # Dreepy/Drakloak
        switch_after_mega_brave=300.0,
    ),
    "iono_lightning": LeverDeltas(
        lillie_early=500.0,
        boss_orders=900.0,
        gust_setup_pokemon=500.0,
        solrock_vs_single_prize=100.0,
    ),
    "abomasnow_water": LeverDeltas(
        prefer_solrock_open=3.0,
        solrock_vs_single_prize=400.0,
        hariyama_vs_ex=300.0,
        skip_mega_brave_vs_bulk_single_prize=800.0,
        boss_orders=1000.0,
        gust_setup_pokemon=800.0,  # Snover before Hammer-lanche online
        premium_power_pro=-500.0,  # race before they flip high damage
    ),
    "kyogre_water": LeverDeltas(
        solrock_vs_single_prize=300.0,
        boss_orders=800.0,
        gust_setup_pokemon=400.0,
    ),
    "trevenant_control": LeverDeltas(
        boss_orders=1200.0,
        gust_setup_pokemon=900.0,  # Phantump / Snorlax
        avoid_ko_trevenant_setup=2000.0,
        solrock_vs_single_prize=250.0,
        skip_mega_brave_vs_bulk_single_prize=400.0,
    ),
    "crustle_wall": LeverDeltas(
        bench_hariyama_bonus=500.0,
        hariyama_vs_ex=400.0,
        mega_brave_vs_ex=-1000.0,  # ex into wall immune
    ),
    "dunsparce_engine": LeverDeltas(
        boss_orders=500.0,
        gust_setup_pokemon=300.0,
        solrock_vs_single_prize=150.0,
    ),
}


def merge_lever_table(
    overrides: dict[str, LeverDeltas] | None = None,
) -> dict[str, LeverDeltas]:
    """Return LUCARIO_LEVERS with optional per-archetype replacements."""
    if not overrides:
        return LUCARIO_LEVERS
    return {**LUCARIO_LEVERS, **overrides}


def levers_for_lucario(
    opp_board_ids: set[int] | frozenset[int],
    lever_table: dict[str, LeverDeltas] | None = None,
) -> LeverDeltas:
    """Merge deltas for all detected tags (sum). Prefer primary_archetype for first impl."""
    table = lever_table if lever_table is not None else LUCARIO_LEVERS
    tags = detect_archetypes(opp_board_ids)
    if not tags:
        return LeverDeltas()
    merged = LeverDeltas()
    for tag in tags:
        if tag not in table:
            continue
        d = table[tag]
        for f in merged.__dataclass_fields__:
            object.__setattr__(
                merged,
                f,
                getattr(merged, f) + getattr(d, f),
            )
    return merged


def levers_for_deck(deck_name: str, opp_board_ids: set[int] | frozenset[int]) -> LeverDeltas:
    if deck_name == "mega_lucario_ex":
        return levers_for_lucario(opp_board_ids)
    return LeverDeltas()
