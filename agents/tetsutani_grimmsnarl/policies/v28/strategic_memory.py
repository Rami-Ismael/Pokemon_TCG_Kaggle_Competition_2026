from __future__ import annotations

from dataclasses import dataclass, field

from .domain_cards import profile
from .resource_ledger import FROSLASS, SNORUNT, ResourceLedger
from .turn_objective import ObjectiveAssessment, TurnObjective
from .turn_phase import PhaseAssessment


@dataclass(frozen=True)
class StrategicMemoryView:
    primary: TurnObjective
    objective_age: int
    phase_age: int
    last_turn_job: str
    current_turn_last_job: str
    passive_pressure_campaign: bool
    passive_campaign_age: int
    reserved_snorunt_alive: bool
    reason: str


@dataclass
class StrategicMemory:
    """Public, cross-turn strategic continuity.

    The memory is reset at the start of every game.  It stores only information
    previously visible in public observations or actions selected by this policy.
    It never stores replay identifiers, teacher actions, hidden cards, or fitted
    parameters.
    """

    last_turn: int = -1
    objective: TurnObjective | None = None
    objective_since_turn: int = 0
    phase_name: str = ""
    phase_since_turn: int = 0
    last_turn_job: str = ""
    current_turn_jobs: list[str] = field(default_factory=list)
    passive_pressure_since_turn: int | None = None
    reserved_snorunt_turn: int | None = None
    serial_first_seen_turn: dict[int, int] = field(default_factory=dict)
    serial_first_energy_turn: dict[int, int] = field(default_factory=dict)
    serial_first_evolved_turn: dict[int, int] = field(default_factory=dict)
    serial_first_board_order: dict[int, int] = field(default_factory=dict)
    _board_counter: int = 0

    def reset(self) -> None:
        self.last_turn = -1
        self.objective = None
        self.objective_since_turn = 0
        self.phase_name = ""
        self.phase_since_turn = 0
        self.last_turn_job = ""
        self.current_turn_jobs.clear()
        self.passive_pressure_since_turn = None
        self.reserved_snorunt_turn = None
        self.serial_first_seen_turn.clear()
        self.serial_first_energy_turn.clear()
        self.serial_first_evolved_turn.clear()
        self.serial_first_board_order.clear()
        self._board_counter = 0

    @staticmethod
    def _job(option: dict) -> str:
        t = int(option.get("type", -1) if option.get("type") is not None else -1)
        source = int(option.get("source_id", 0) or 0)
        target = int(option.get("target_id", 0) or 0)
        if t == 13:
            return "attack"
        if t == 12:
            return "pivot"
        if t == 10:
            return "gym"
        if t == 9:
            return "evolve_froslass" if source == FROSLASS else "evolve_attacker"
        if t == 8:
            return "attach_control" if target == 112 else "attach_energy"
        if t == 7:
            return {
                1086: "poffin",
                1219: "petrel",
                1152: "pad",
                1122: "gear",
                1097: "night",
                112: "bench_control",
                646: "bench_attacker",
                SNORUNT: "bench_snorunt",
            }.get(source, f"play_{source}")
        if t == 14:
            return "pass"
        return f"context_action_{t}"

    @staticmethod
    def _objective_complete(objective: TurnObjective, ledger: ResourceLedger) -> bool:
        if objective == TurnObjective.ESTABLISH_BOARD:
            return ledger.marnie_line_count >= 2 or ledger.bench_slots == 0
        if objective == TurnObjective.BUILD_BACKUP_ATTACKER:
            return ledger.marnie_line_count >= 2
        if objective == TurnObjective.COMPLETE_ATTACKER:
            return ledger.attack_ready_grimmsnarl > 0
        if objective == TurnObjective.ENABLE_CONTROL:
            return ledger.control_online
        if objective == TurnObjective.COMPLETE_FROSLASS:
            return ledger.froslass_online
        if objective == TurnObjective.RECOVER_RESOURCES:
            return ledger.recoverable_core <= 2 or ledger.supporter_played
        if objective == TurnObjective.PRESERVE_TEMPO:
            return ledger.active_id != FROSLASS
        return False

    def _update_identity(self, ledger: ResourceLedger) -> None:
        for card in ledger.own_cards():
            serial = int(card.get("serial", -1) if card.get("serial") is not None else -1)
            if serial < 0:
                continue
            if serial not in self.serial_first_seen_turn:
                self.serial_first_seen_turn[serial] = ledger.turn
                self.serial_first_board_order[serial] = self._board_counter
                self._board_counter += 1
            if int(card.get("en", 0) or 0) > 0:
                self.serial_first_energy_turn.setdefault(serial, ledger.turn)
            if card.get("preEvolution"):
                self.serial_first_evolved_turn.setdefault(serial, ledger.turn)

    def observe(
        self,
        ledger: ResourceLedger,
        inferred: ObjectiveAssessment,
        phase: PhaseAssessment,
    ) -> StrategicMemoryView:
        if self.last_turn >= 0 and ledger.turn < self.last_turn:
            self.reset()
        if ledger.turn != self.last_turn:
            if self.last_turn >= 0:
                self.last_turn_job = self.current_turn_jobs[-1] if self.current_turn_jobs else ""
            self.current_turn_jobs.clear()
            self.last_turn = ledger.turn

        self._update_identity(ledger)
        phase_name = phase.phase.value
        if phase_name != self.phase_name:
            self.phase_name = phase_name
            self.phase_since_turn = ledger.turn

        candidate = inferred.primary
        if self.objective is None:
            self.objective = candidate
            self.objective_since_turn = ledger.turn
        elif self._objective_complete(self.objective, ledger):
            self.objective = candidate
            self.objective_since_turn = ledger.turn
        elif inferred.committed and candidate != self.objective:
            # Hard public objectives may pre-empt a soft unfinished objective.
            self.objective = candidate
            self.objective_since_turn = ledger.turn
        elif ledger.turn - self.objective_since_turn >= 3 and candidate != self.objective:
            # A non-completing soft objective expires after three own turns.
            self.objective = candidate
            self.objective_since_turn = ledger.turn

        opponent = profile(ledger.opponent_active_id)
        long_horizon = (
            opponent.is_hand_size_punisher
            or opponent.is_repeatable_draw_engine
            or opponent.is_energy_engine
            or opponent.blocks_ex_attack_damage
        )
        passive_unfinished = ledger.board[SNORUNT] > 0 or ledger.hand[FROSLASS] > 0 or ledger.hand[SNORUNT] > 0
        if long_horizon and passive_unfinished:
            if self.passive_pressure_since_turn is None:
                self.passive_pressure_since_turn = ledger.turn
        elif ledger.froslass_online and ledger.board[SNORUNT] == 0 and ledger.hand[SNORUNT] == 0:
            self.passive_pressure_since_turn = None

        if ledger.board[SNORUNT] > 0 or ledger.hand[SNORUNT] > 0:
            self.reserved_snorunt_turn = self.reserved_snorunt_turn if self.reserved_snorunt_turn is not None else ledger.turn
        elif ledger.froslass_online:
            self.reserved_snorunt_turn = None

        primary = self.objective or candidate
        return StrategicMemoryView(
            primary=primary,
            objective_age=max(0, ledger.turn - self.objective_since_turn),
            phase_age=max(0, ledger.turn - self.phase_since_turn),
            last_turn_job=self.last_turn_job,
            current_turn_last_job=self.current_turn_jobs[-1] if self.current_turn_jobs else "",
            passive_pressure_campaign=self.passive_pressure_since_turn is not None,
            passive_campaign_age=0 if self.passive_pressure_since_turn is None else max(0, ledger.turn - self.passive_pressure_since_turn),
            reserved_snorunt_alive=self.reserved_snorunt_turn is not None,
            reason=f"persistent objective={primary.value}; phase={phase_name}; last_turn_job={self.last_turn_job or 'none'}",
        )

    def record(self, selected_options: list[dict]) -> None:
        for option in selected_options:
            self.current_turn_jobs.append(self._job(option))
            if self._job(option) == "bench_snorunt" and self.last_turn >= 0:
                self.reserved_snorunt_turn = self.last_turn
