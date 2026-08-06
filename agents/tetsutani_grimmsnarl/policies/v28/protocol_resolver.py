from __future__ import annotations

from dataclasses import dataclass

from .resource_ledger import GRIMMSNARL, ResourceLedger


@dataclass(frozen=True)
class ProtocolDecision:
    indices: tuple[int, ...]
    job: str
    reason: str
    mode: str


def _last_grim_evolution(ledger: ResourceLedger) -> dict:
    return next(
        (
            action
            for action in reversed(ledger.history)
            if int(action.get("type", -1)) == 9
            and int(action.get("source_id", 0) or 0) == GRIMMSNARL
        ),
        {},
    )


def _evolved_target_energy(ledger: ResourceLedger, evolution: dict) -> int:
    area = int(evolution.get("target_area", 0) or 0)
    index = int(evolution.get("inplay_index", -1) if evolution.get("inplay_index") is not None else -1)
    cards = (ledger.active,) if area == 4 else ledger.bench if area == 5 else tuple()
    if 0 <= index < len(cards):
        return int(cards[index].get("en", 0) or 0)
    return 0


def plan_protocol(
    ledger: ResourceLedger,
    options: list[dict],
    baseline: list[int],
    *,
    context: int,
) -> ProtocolDecision | None:
    if not options:
        return None

    # Punk Up activation is a post-evolution protocol job.  If the newly evolved
    # Bench attacker is already attack-ready and the board already owns three ready
    # attackers plus another reserved resource, the energy-loading job is complete.
    if context == 43:
        evolution = _last_grim_evolution(ledger)
        area = int(evolution.get("target_area", 0) or 0)
        target_energy = _evolved_target_energy(ledger, evolution)
        ready = sum(
            int(card.get("id", 0) or 0) == GRIMMSNARL and int(card.get("en", 0) or 0) >= 2
            for card in ledger.own_cards()
        )
        froslass_count = int(ledger.board[104])
        if area == 5 and target_energy >= 2 and ready >= 3 and (
            ledger.hand[7] >= 1 or froslass_count >= 2 or ledger.discard[7] >= 3
        ):
            decline = next((i for i, option in enumerate(options) if int(option.get("type", -1)) == 2), None)
            if decline is not None:
                return ProtocolDecision(
                    (decline,),
                    "decline_completed_punk_up_job",
                    "the newly evolved Bench attacker and the existing attacker pool are already fully funded",
                    "committed_change" if [decline] != baseline else "committed_confirm",
                )
        return ProtocolDecision(
            tuple(baseline),
            "resolve_punk_up_activation",
            "the post-evolution ability protocol is resolved from the visible energy ledger",
            "structural_confirm",
        )

    # These contexts are engine protocol decisions rather than strategic card
    # ordering.  The validated result remains the deterministic resolver output.
    if context in (1, 2, 38, 40, 41):
        return ProtocolDecision(
            tuple(baseline),
            f"resolve_protocol_context_{context}",
            "the engine prompt is resolved by its deterministic legal-option protocol",
            "structural_confirm",
        )
    return None
