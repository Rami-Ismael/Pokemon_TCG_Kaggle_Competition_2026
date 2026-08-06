from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from . import validated_rule_profile as primitive
from .domain import DARK, FROSLASS, GRIMMSNARL, IMPIDIMP, MORGREM, MUNKIDORI, NIGHT_STRETCHER, POKE_PAD
from .state_model import TurnState


class SelectionObjective(Enum):
    COMPLETE_ATTACK_LINE = auto()
    BUILD_BACKUP_ATTACK_LINE = auto()
    COMPLETE_FROSLASS_LINE = auto()
    SCALE_DAMAGE_ENGINE = auto()
    RESTORE_ENERGY = auto()
    RESTORE_POKEMON = auto()
    BANK_SUPPORTER = auto()
    COMPATIBILITY = auto()


@dataclass(frozen=True)
class SelectionIntent:
    objective: SelectionObjective
    desired_card_id: int | None
    priority: int
    confidence: int
    reason: str


class EffectObjectivePlanner:
    """Plan non-MAIN card-effect selections from the same resource ledger.

    The planner never sees replay identifiers and never indexes a state table.
    It receives only public board/hand resources and the current legal cards.
    """

    def plan_to_hand(self, state: TurnState, effect: int, baseline_card_id: int, available: set[int]) -> list[SelectionIntent]:
        l = state.ledger
        h = l.hand
        out: list[SelectionIntent] = []

        # Spikemuth: once one ready attacker exists and a Morgrem is already
        # visible, a missing Basic is the limiting resource for the next line.
        if (
            effect == primitive.SPIKEMUTH and baseline_card_id == GRIMMSNARL
            and l.ready_attackers == 1 and l.hand[IMPIDIMP] == 0
            and l.hand[MORGREM] == 0 and l.bench_free >= 1
            and l.hand[GRIMMSNARL] == 0 and IMPIDIMP in available
            and any(p.get("id") == MORGREM for p in primitive._all_inplay(state.view["me"]))
            and not any(p.get("id") == IMPIDIMP for p in primitive._all_inplay(state.view["me"]))
        ):
            out.append(SelectionIntent(
                SelectionObjective.BUILD_BACKUP_ATTACK_LINE, IMPIDIMP, 930, 94,
                "one ready attacker plus a visible Morgrem: reserve the missing Basic for the backup line",
            ))

        # Petrel: two or more unpowered Munkidori make Night Stretcher the
        # highest-leverage trainer because it restores Dark Energy access.
        if (
            effect == primitive.PETREL and baseline_card_id == POKE_PAD
            and l.munkidori_unpowered >= 2 and NIGHT_STRETCHER in available
        ):
            out.append(SelectionIntent(
                SelectionObjective.RESTORE_ENERGY, NIGHT_STRETCHER, 920, 93,
                "multiple unpowered Munkidori: recover the energy loop before another broad Pokemon search",
            ))

        # Dawn: after the first powered Munkidori is online, the next copy
        # scales the damage-transfer engine more than another Basic attacker.
        if (
            effect == primitive.DAWN and baseline_card_id == IMPIDIMP
            and l.munkidori_powered >= 1 and MUNKIDORI in available
        ):
            out.append(SelectionIntent(
                SelectionObjective.SCALE_DAMAGE_ENGINE, MUNKIDORI, 910, 92,
                "one powered Munkidori is online: add the second damage-transfer body",
            ))

        # Night Stretcher: if Grimmsnarl is already retained in hand, recover
        # Dark Energy instead of a duplicate stage-two card.
        if (
            effect == primitive.NIGHT_STRETCHER and baseline_card_id == GRIMMSNARL
            and h[GRIMMSNARL] >= 1 and DARK in available
        ):
            out.append(SelectionIntent(
                SelectionObjective.RESTORE_ENERGY, DARK, 900, 94,
                "stage-two attacker already banked in hand: restore the energy resource",
            ))

        # Night Stretcher: a visible Snorunt without Froslass is an incomplete
        # engine line; recover the evolution rather than redundant Energy.
        if (
            effect == primitive.NIGHT_STRETCHER and baseline_card_id == DARK
            and any(p.get("id") == primitive.SNORUNT for p in primitive._all_inplay(state.view["me"]))
            and not any(p.get("id") == FROSLASS for p in primitive._all_inplay(state.view["me"]))
            and FROSLASS in available
        ):
            out.append(SelectionIntent(
                SelectionObjective.COMPLETE_FROSLASS_LINE, FROSLASS, 890, 90,
                "visible Snorunt without its evolution: complete the passive damage line",
            ))

        out.append(SelectionIntent(
            SelectionObjective.COMPATIBILITY, baseline_card_id, -1, 0,
            "validated primitive card-effect choice",
        ))
        return sorted(out, key=lambda x: (-x.priority, -x.confidence, x.objective.value))
