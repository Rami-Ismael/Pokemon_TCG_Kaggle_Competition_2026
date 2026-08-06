from __future__ import annotations

"""Persistent within-turn objective and process-queue state.

Only public observations and semantic roles selected by this policy are stored.
There is no replay identifier table, state hash lookup, fitted parameter,
rollout, search tree, or learned model.
"""

from dataclasses import dataclass, field
from typing import Any

from .planner import PlannedObjective, TurnObjective
from .state_model import TurnState


_LEDGER_FIELDS = (
    "hand_size", "deck_size", "bench_used", "bench_free",
    "attack_lines", "ready_attackers", "froslass_lines",
    "munkidori_total", "munkidori_powered", "munkidori_unpowered",
    "dark_in_hand", "stadium_live", "supporter_used", "energy_attached",
    "active_id", "active_energy", "active_damage",
)


@dataclass
class PersistentTurnPlan:
    """Carry an unfinished deterministic plan across prompts in one turn."""

    episode: int | None = None
    turn: int | None = None
    last_action_count: int | None = None
    executed_roles: list[str] = field(default_factory=list)
    executed_objectives: list[str] = field(default_factory=list)
    last_ledger: dict[str, Any] = field(default_factory=dict)
    last_resource_delta: dict[str, Any] = field(default_factory=dict)
    previous_turn: int | None = None
    previous_roles: tuple[str, ...] = ()
    previous_objectives: tuple[str, ...] = ()
    previous_ledger: dict[str, Any] = field(default_factory=dict)
    active_objective: str = "COMPATIBILITY"
    active_origin_role: str = "NONE"

    def reset(self) -> None:
        self.episode = None
        self.turn = None
        self.last_action_count = None
        self.executed_roles.clear()
        self.executed_objectives.clear()
        self.last_ledger.clear()
        self.last_resource_delta.clear()
        self.previous_turn = None
        self.previous_roles = ()
        self.previous_objectives = ()
        self.previous_ledger.clear()
        self.active_objective = "COMPATIBILITY"
        self.active_origin_role = "NONE"

    @staticmethod
    def _snapshot(state: TurnState) -> dict[str, Any]:
        ledger = state.ledger
        return {name: getattr(ledger, name) for name in _LEDGER_FIELDS}

    def sync(self, state: TurnState) -> None:
        ep = state.view.get("ep")
        snapshot = self._snapshot(state)
        if ep != self.episode or self.turn != state.turn:
            same_episode = ep == self.episode
            if same_episode and self.executed_roles:
                self.previous_turn = self.turn
                self.previous_roles = tuple(self.executed_roles)
                self.previous_objectives = tuple(self.executed_objectives)
                self.previous_ledger = dict(self.last_ledger)
            elif not same_episode:
                self.previous_turn = None
                self.previous_roles = ()
                self.previous_objectives = ()
                self.previous_ledger = {}
            self.episode = ep
            self.turn = state.turn
            self.last_action_count = state.action_count
            self.executed_roles = []
            self.executed_objectives = []
            self.active_objective = "COMPATIBILITY"
            self.active_origin_role = "NONE"
            self.last_ledger = snapshot
            self.last_resource_delta = {}
            return

        # A new public prompt in the same turn contains the engine-confirmed
        # resource result of the previous selection.  Preserve that delta for
        # planning and diagnostics; repeated reads of the same prompt do not
        # fabricate another update.
        if self.last_action_count != state.action_count:
            delta: dict[str, Any] = {}
            for name, value in snapshot.items():
                old = self.last_ledger.get(name)
                if value != old:
                    if isinstance(value, bool):
                        delta[name] = {"from": old, "to": value}
                    elif isinstance(value, int) and isinstance(old, int):
                        delta[name] = value - old
                    else:
                        delta[name] = {"from": old, "to": value}
            self.last_resource_delta = delta
            self.last_ledger = snapshot
            self.last_action_count = state.action_count

    @property
    def last_role(self) -> str:
        return self.executed_roles[-1] if self.executed_roles else "TURN_START"

    def record(self, state: TurnState, role: str, objective: str, origin_role: str) -> None:
        self.sync(state)
        self.executed_roles.append(role)
        self.executed_objectives.append(objective)
        self.active_objective = objective
        self.active_origin_role = origin_role


    def horizon_objectives(self, state: TurnState, provisional_role: str) -> list[PlannedObjective]:
        """Re-plan unfinished strategic work across the opponent's intervening turn.

        The horizon uses only this policy's prior semantic roles and the public
        resource ledgers observed at the end of its prior turn and the start of
        the current turn.  No replay key or state lookup is used.
        """
        self.sync(state)
        if self.executed_roles or not self.previous_roles or self.previous_turn is None:
            return []
        # The target player's decisions are separated by one opponent turn.
        if state.turn - self.previous_turn != 2:
            return []

        enabled = {"H1","H2","H3","H4","H5","H6","H7","H8","H9","H10"}
        l = state.ledger
        legal = state.legal_roles
        p = self.previous_ledger
        prev_roles = set(self.previous_roles)
        prev_last = self.previous_roles[-1]
        prev_ready = int(p.get("ready_attackers", 0) or 0)
        prev_lines = int(p.get("attack_lines", 0) or 0)
        prev_frost = int(p.get("froslass_lines", 0) or 0)
        delta_ready = l.ready_attackers - prev_ready
        delta_lines = l.attack_lines - prev_lines
        out: list[PlannedObjective] = []

        # H1: the opponent removed the only ready attacker.  Rebuild the
        # exposed Morgrem before spending the turn on Adrena-Brain.
        if (
            "H1" in enabled and provisional_role == "ABILITY_ADRENA"
            and not l.stadium_live and delta_ready <= -1 and l.ready_attackers == 0
            and "EVOLVE_GRIMMSNARL" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.RECOVER_LOST_ATTACKER, 1100, 100,
                "the opponent removed a ready attacker; restore the exposed attack line first",
            ))

        # H2: a stable three-line / one-Froslass board ended the prior turn,
        # but the persistent search stadium is now gone.  Restore the engine
        # before optional damage movement while the hand is compact.
        if (
            "H2" in enabled and provisional_role == "ABILITY_ADRENA"
            and prev_lines == 3 and prev_frost == 1 and l.hand_size <= 6
            and not l.stadium_live and "PLAY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.RESTORE_SEARCH_ENGINE, 1099, 100,
                "stable pressure board lost its persistent search engine; restore it before optional damage movement",
            ))

        # H3: after the prior turn completed a Grimmsnarl, the next long-horizon
        # objective is the passive Froslass engine rather than another Adrena.
        if (
            "H3" in enabled and provisional_role == "ABILITY_ADRENA"
            and "EVOLVE_GRIMMSNARL" in prev_roles and state.turn >= 9
            and l.hand_size <= 5 and "EVOLVE_FROSLASS" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.COMPLETE_PASSIVE_ENGINE, 1098, 100,
                "the prior turn completed the attacker; finish the passive damage engine next",
            ))

        # H4: an opening turn ended without an attack and left a developing
        # line.  Continue that line before opening the damage-movement queue.
        if (
            "H4" in enabled and provisional_role == "ABILITY_ADRENA"
            and state.turn <= 3 and not any(x.startswith("ATTACK_") for x in self.previous_roles)
            and prev_last == "END" and "EVOLVE_MORGREM" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.CONTINUE_ATTACK_DEVELOPMENT, 1097, 99,
                "the opening attack objective remained unfinished across the opponent turn",
            ))

        # H5: with both players still on six prizes and exactly two attack
        # lines, the previous turn did not consume broad search.  Use the live
        # exact stadium search before committing the visible evolution.
        if (
            "H5" in enabled and provisional_role == "EVOLVE_GRIMMSNARL"
            and l.opponent_prizes == 6 and l.attack_lines <= 2
            and "PLAY_POKE_PAD" not in prev_roles and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.CONTINUE_EXACT_SEARCH, 1096, 100,
                "opening board remains shallow; consume the banked exact search before committing evolution",
            ))

        # H6: the prior turn invested in an Impidimp line.  With a compact hand,
        # continue exact attacker search before powering the damage engine.
        if (
            "H6" in enabled and provisional_role == "ATTACH_MUNKIDORI"
            and "PLAY_IMPIDIMP" in prev_roles and l.hand_size <= 4
            and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.CONTINUE_EXACT_SEARCH, 1095, 100,
                "continue the attacker investment from the prior turn before powering a side engine",
            ))

        # H7: an attacker line and its ready state disappeared during the
        # opponent turn.  Exact stadium search precedes broad Poke Pad recovery.
        if (
            "H7" in enabled and provisional_role == "PLAY_POKE_PAD"
            and delta_ready <= -1 and delta_lines <= -1
            and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.CONTINUE_EXACT_SEARCH, 1094, 100,
                "the opponent removed an attack line; use exact recovery search before broad search",
            ))

        # H8: in late development, if the prior turn did not add another
        # Impidimp, complete the existing Morgrem before adding Munkidori.
        if (
            "H8" in enabled and provisional_role == "PLAY_MUNKIDORI"
            and state.turn >= 8 and "PLAY_IMPIDIMP" not in prev_roles
            and "EVOLVE_MORGREM" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.CONTINUE_ATTACK_DEVELOPMENT, 1093, 100,
                "complete the carried attacker line before opening another damage-engine slot",
            ))

        # H9: if the prior turn did not divert into a new Munkidori, preserve
        # the direct Rare Candy completion of the exposed Active Impidimp.
        if (
            "H9" in enabled and provisional_role == "EVOLVE_MORGREM"
            and l.active_id == 646 and l.active_energy == 0
            and "PLAY_MUNKIDORI" not in prev_roles and "PLAY_RARE_CANDY" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.ACCELERATE_ATTACKER, 1092, 100,
                "carry the direct attacker-completion plan across turns instead of inserting the intermediate stage",
            ))

        # H10: a prior turn that ended without a ready attacker and now exposes
        # Active Impidimp with Froslass online needs targeted trainer selection,
        # not generic draw.
        if (
            "H10" in enabled and provisional_role == "PLAY_LILLIE"
            and prev_ready == 0 and l.active_id == 646 and l.froslass_lines >= 1
            and "PLAY_PETREL" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.TARGETED_RECOVERY, 1091, 100,
                "stalled attacker with passive engine online: select the exact trainer needed to resume pressure",
            ))

        return sorted(out, key=lambda x: (-x.priority, -x.confidence, x.objective.value))

    def continuation_objectives(self, state: TurnState, provisional_role: str) -> list[PlannedObjective]:
        """Return only objectives whose resource work is demonstrably unfinished."""
        self.sync(state)
        l = state.ledger
        legal = state.legal_roles
        last = self.last_role
        out: list[PlannedObjective] = []

        # Complete the attacker rotation opened by the previous evolution.
        if (
            last == "EVOLVE_GRIMMSNARL"
            and provisional_role == "EVOLVE_GRIMMSNARL"
            and l.ready_attackers == 1 and l.active_energy == 1
            and "RETREAT" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.REPOSITION, 995, 99,
                "one attacker was completed; rotate the one-Energy Active before developing another backup",
            ))

        # A newly placed Munkidori is an unfinished damage-engine resource.
        if (
            last == "PLAY_MUNKIDORI"
            and provisional_role == "PLAY_NIGHT_STRETCHER"
            and "ATTACH_MUNKIDORI" in legal and "RETREAT" in legal
            and not l.energy_attached and l.dark_in_hand >= 1
        ):
            out.append(PlannedObjective(
                TurnObjective.POWER_DAMAGE_ENGINE, 994, 99,
                "finish the newly placed Munkidori before opening the recovery queue",
            ))

        # With no attack body, multi-Basic expansion precedes the one-card
        # stadium activation that was opened by PLAY_SPIKEMUTH.
        if (
            last == "PLAY_SPIKEMUTH"
            and provisional_role == "ABILITY_SPIKEMUTH"
            and l.attack_lines == 0 and "PLAY_POFFIN" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.EXPAND_BOARD, 993, 99,
                "zero attack lines: resolve multi-Basic expansion before the single-card search",
            ))

        # After Adrena-Brain, exact stadium search finishes the missing attacker
        # objective before broad Poke Pad search.
        if (
            last == "ABILITY_ADRENA"
            and provisional_role == "PLAY_POKE_PAD"
            and l.ready_attackers == 0 and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.BUILD_ATTACKER, 992, 99,
                "damage movement finished but no attacker is ready; consume exact stadium search first",
            ))

        # After the first Impidimp is placed, continue attack-line acquisition
        # with broad search before another multi-Basic expansion.
        if (
            last == "PLAY_IMPIDIMP"
            and provisional_role == "PLAY_POFFIN"
            and l.active_id == 646 and "PLAY_POKE_PAD" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.BROAD_SEARCH, 991, 99,
                "first attack body placed; continue the attack-line queue with broad Pokemon search",
            ))

        # The live stadium search is part of the current engine queue and is
        # completed before opening Unfair Stamp's disruption queue.
        if (
            last == "ABILITY_ADRENA"
            and provisional_role == "PLAY_UNFAIR_STAMP"
            and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.BUILD_ATTACKER, 990, 100,
                "finish the live stadium-search step before the supporter-disruption step",
            ))

        # A mature attack board makes the newly placed Munkidori the sole
        # unfinished engine resource.
        if (
            last == "PLAY_MUNKIDORI"
            and provisional_role == "PLAY_POKE_PAD"
            and l.attack_lines >= 3 and "ATTACH_MUNKIDORI" in legal
            and not l.energy_attached and l.dark_in_hand >= 1
        ):
            out.append(PlannedObjective(
                TurnObjective.POWER_DAMAGE_ENGINE, 989, 99,
                "mature attack board: power the newly placed damage engine before broad search",
            ))

        # Once one damage body is added, complete the exposed Active Morgrem
        # instead of adding another Munkidori.
        if (
            last == "PLAY_MUNKIDORI"
            and provisional_role == "PLAY_MUNKIDORI"
            and l.active_id == 647 and l.ready_attackers == 0
            and "EVOLVE_GRIMMSNARL" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.COMPLETE_ATTACKER, 988, 99,
                "damage body placed; complete the Active Morgrem before another engine body",
            ))

        # Redirect a generic bench attachment to the Munkidori just placed.
        if (
            last == "PLAY_MUNKIDORI"
            and provisional_role == "ATTACH_IMPIDIMP"
            and "ATTACH_MUNKIDORI" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.POWER_DAMAGE_ENGINE, 987, 100,
                "newly placed Munkidori is the intended destination of the unused Dark attachment",
            ))

        # If Snorunt is Active, finish that active engine resource before
        # powering the newly placed bench Impidimp.
        if (
            last == "PLAY_IMPIDIMP"
            and provisional_role == "ATTACH_IMPIDIMP"
            and l.active_id == 860 and "ATTACH_SNORUNT" in legal
            and not l.energy_attached
        ):
            out.append(PlannedObjective(
                TurnObjective.POWER_ACTIVE_BASIC, 984, 98,
                "continue the active-engine queue by powering Snorunt before the new bench attacker",
            ))

        # Recovery has supplied resources; activate the exposed Munkidori before
        # reopening broad search.
        if (
            last == "PLAY_NIGHT_STRETCHER"
            and provisional_role == "PLAY_POKE_PAD"
            and l.munkidori_unpowered >= 1 and "ATTACH_MUNKIDORI" in legal
            and not l.energy_attached and l.dark_in_hand >= 1
        ):
            out.append(PlannedObjective(
                TurnObjective.POWER_DAMAGE_ENGINE, 983, 97,
                "recovery completed; activate the exposed Munkidori before broad search",
            ))

        # Once the stadium search is complete, keep damage tempo instead of
        # retreating a ready Active Grimmsnarl.
        if (
            last == "ABILITY_SPIKEMUTH"
            and provisional_role == "RETREAT" and l.dark_in_hand >= 1
            and "ATTACK_SHADOW_BULLET" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.KEEP_PRESSURE, 982, 97,
                "stadium search completed; preserve attack tempo instead of rotating the ready Active",
            ))

        # An Active Morgrem after damage movement still lacks its exact attacker
        # resource, so stadium search precedes recovery.
        if (
            last == "ABILITY_ADRENA"
            and provisional_role == "PLAY_NIGHT_STRETCHER"
            and l.active_id == 647 and "ABILITY_SPIKEMUTH" in legal
        ):
            out.append(PlannedObjective(
                TurnObjective.BUILD_ATTACKER, 981, 97,
                "damage movement completed with Active Morgrem; exact attacker search precedes recovery",
            ))

        return out
