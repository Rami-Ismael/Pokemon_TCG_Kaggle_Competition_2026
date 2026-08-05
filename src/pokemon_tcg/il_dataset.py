"""Streaming behavior-cloning dataset for the Pokemon TCG imitation-learning agent.

Two interchangeable episode sources produce the identical example stream:
``ILDataset`` reads raw ``{episode_id}.json`` files from the on-disk split
folders named in ``data/episodes/splits/splits.json``; ``ShardILDataset``
reads the same episodes from the zstd Parquet shards of ADR-001 (private
Hugging Face repo ``config.HF_EPISODES_REPO``, or a local shard directory),
so the corpus no longer has to fit on this laptop raw. Either way the split
discipline is the "held-out-DAY rule": train and eval are distinct calendar
days (never shuffle across them), enforced by folder naming locally and by
the ``{split}/day=YYYY-MM-DD/`` layout on the Hub.

Each decision (one step, one agent POV) becomes one fixed-size training
example:

* state tokens -- my active Pokemon, my bench (padded to ``MAX_BENCH``),
  the opponent's active + bench, and my hand (padded to ``MAX_HAND``); each
  slot carries a card-id embedding index plus (hp_frac, energy_norm,
  tool_norm, present) scalars. Bench/hand length is genuinely variable
  (0..N cards) so empty slots are explicitly zeroed and masked rather than
  assumed to exist.
* one token per legal move in ``select.option``, plus a synthetic DECLINE
  slot whenever ``select.minCount == 0`` (see ``_encode_options``), padded
  to ``MAX_OPTIONS``.
* a label = the index the recorded agent chose (``len(select.option)``
  means DECLINE).

Every decision is still trained as a single categorical choice over the
(masked) option set -- Pattern B, one shared scoring head, cross-entropy
against the recorded index. Multi-select decisions (``select.maxCount > 1``,
e.g. "discard 2 of these cards") are handled by unrolling them
autoregressively into one training example per pick, re-masking prior picks
via ``exclude`` (see ``iter_decisions``), rather than by a second model or a
combinatorial loss -- see notes/phase1_decisions.md §1.3 for why.

Privacy invariant (standing rule #2): the agent may only see what a real
player sees -- the board (both sides' in-play Pokemon), its OWN hand, and
public counts (deckCount, handCount, prize count). The per-agent
``observation.current`` is POV-filtered by the engine (opponent hand is
None, deck contents absent, facedown prizes None), so encoding from it is
safe. The full hidden state -- both players' decks, hands, and prizes per
frame -- DOES exist in every episode file at ``steps[0][0]["visualize"]``
and must never be read: features mined from it would be unavailable live,
which is invisible offline and fatal on the ladder. Enforced by
tests/test_privacy_no_leak.py; re-run it when touching this module.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import torch
from torch.utils.data import IterableDataset

from . import config

_CG_DIR = config.PROJECT_ROOT / "data" / "external" / "cg-lib"
if (_CG_DIR / "cg" / "api.py").exists() and str(_CG_DIR) not in sys.path:
    sys.path.insert(0, str(_CG_DIR))

from cg.api import AreaType, OptionType, SelectData, State  # noqa: E402
from cg.utils import to_dataclass  # noqa: E402

# --- Fixed-size shape constants (chosen from an empirical scan of the train
# split: observed max option count 42, max hand size 31 in a 60-episode
# sample -- headroom is added above the observed max rather than the exact
# max, since rarer long-tail states exist in the full 4,554-episode split).
MAX_BENCH = 8
MAX_HAND = 32
MAX_OPTIONS = 48
# slots: my_active, my_bench, opp_active, opp_bench, my_hand
N_STATE_SLOTS = 1 + MAX_BENCH + 1 + MAX_BENCH + MAX_HAND

CARD_VOCAB_SIZE = 1269  # card ids observed 1..1267; 0 = PAD/empty slot; last row = OOV bucket
ATTACK_VOCAB_SIZE = 1557  # attack ids observed 1..1556; 0 = PAD/none; last row = OOV bucket
SPECIAL_VOCAB_SIZE = 6  # SpecialConditionType 0..4, shifted +1; 0 = none
SELECT_TYPE_VOCAB_SIZE = 11  # SelectType 0..10
# SelectContext is 0..48 today; cg.api warns more ids may be appended, so leave headroom
SELECT_CONTEXT_VOCAB_SIZE = 96
# OptionType is 0..16; id 17 is a synthetic DECLINE slot (see below), not part of cg.api.
DECLINE_OPTION_TYPE = 17
OPTION_TYPE_VOCAB_SIZE = 18


def _clamp_id(x: int, vocab_size: int) -> int:
    """Map any id outside the trained vocab to the last row (shared OOV bucket).

    Without this, an unseen card/attack id (new cards may be appended during
    the competition per cg.api's own comment) indexes past the embedding
    table and raises IndexError -- an uncaught crash is an instant loss.
    """
    if x < 0 or x >= vocab_size:
        return vocab_size - 1
    return x

N_SLOT_SCALARS = 4  # hp_frac, energy_norm, tool_norm, present
N_GLOBAL_SCALARS = 13
N_OPT_SCALARS = 2  # number_norm, has_ref
N_REF_SCALARS = 3  # hp_frac, energy_norm, tool_norm (per referenced card)

# --- Deterministic-future features (notes/feature_ablation_candidates.md).
# Hand-computed board arithmetic a strong player does before deciding:
# damage-race turns, prize arithmetic, energy timelines. ALL groups are
# always computed into two extra tensors (`extra_global` [N_EXTRA_GLOBAL],
# `extra_opt` [MAX_OPTIONS, N_EXTRA_OPT]); which columns a model actually
# consumes is chosen per-checkpoint by PTCGILConfig.global_features /
# .opt_features (see il_model.py), so one encoder serves every ablation
# arm and legacy checkpoints (no feature fields) simply ignore the extras.
# Inputs are ONLY the POV-filtered observation plus the static public card
# database (cg.api.all_card_data/all_attack -- the rulebook every player
# has memorized); privacy invariant unchanged, see module docstring.
# Spec order defines column order -- append only, never reorder.
GLOBAL_FEATURE_SPECS: dict[str, int] = {
    "ko_race": 5,             # t_my_active_ko/10, t_opp_active_ko/10, race_win, my_dmg/300, opp_dmg/300
    "prize_race": 4,          # pv_opp_active/3, pv_my_active/3, ko_opp_wins_game, my_ko_loses_game
    "energy_deficit": 3,      # my_active_min_def/4, my_bench_min_def/4, opp_active_min_def/4
    "status_conditions": 10,  # poisoned..confused for me, then for opp
}
OPT_FEATURE_SPECS: dict[str, int] = {
    "attack_tactical": 3,     # dmg_vs_opp_active/300, kos_now, ko_wins_game
    "attach_enable": 2,       # post_attach_min_def/4, enables_attack
    "retreat_switch": 2,      # candidate_survival_turns/10, candidate_payable_dmg/300
}
N_EXTRA_GLOBAL = sum(GLOBAL_FEATURE_SPECS.values())
N_EXTRA_OPT = sum(OPT_FEATURE_SPECS.values())

_KO_TURNS_CAP = 10
_DEFICIT_CAP = 4
_DMG_NORM = 300.0


def feature_columns(names: Iterable[str], specs: dict[str, int]) -> list[int]:
    """Column indices (into extra_global / extra_opt) for the named groups.

    Raises on unknown names -- a checkpoint asking for a feature this
    encoder build doesn't produce must fail loudly, not silently misalign.
    """
    offsets: dict[str, int] = {}
    off = 0
    for name, dim in specs.items():
        offsets[name] = off
        off += dim
    cols: list[int] = []
    for name in names:
        if name not in offsets:
            raise KeyError(f"unknown feature group {name!r}; known: {list(specs)}")
        cols.extend(range(offsets[name], offsets[name] + specs[name]))
    return cols


_CARD_DB: dict | None = None
_ATTACK_DB: dict | None = None


def _static_db() -> tuple[dict, dict]:
    """Lazy one-time {cardId: CardData}, {attackId: Attack} from cg's public DB.

    Wrapped so a lib failure degrades to empty dicts (features all-zero)
    instead of crashing -- on the evaluator an uncaught exception is an
    instant loss, and a zeroed feature is strictly recoverable noise.
    """
    global _CARD_DB, _ATTACK_DB
    if _CARD_DB is None:
        try:
            from cg.api import all_attack, all_card_data

            _CARD_DB = {c.cardId: c for c in all_card_data()}
            _ATTACK_DB = {a.attackId: a for a in all_attack()}
        except Exception:
            _CARD_DB, _ATTACK_DB = {}, {}
    return _CARD_DB, _ATTACK_DB


# EnergyType constants used by the greedy cost matcher (values pinned by
# cg.api.EnergyType; imported as ints to keep this pure integer math).
_E_COLORLESS, _E_PSYCHIC, _E_DARKNESS, _E_RAINBOW, _E_TEAM_ROCKET = 0, 5, 7, 10, 11


def _energy_deficit(required: list[int], attached: list[int]) -> int:
    """Unmatched cost symbols under greedy matching (0 = payable now).

    Colored requirements consume a same-type attached energy, else a
    wildcard (RAINBOW matches anything; TEAM_ROCKET matches {PSYCHIC,
    DARKNESS}); COLORLESS requirements consume any leftover. Greedy is not
    a perfect bipartite matcher but is exact for every real cost list in
    this format (costs are same-type runs + colorless padding).
    """
    pool: dict[int, int] = {}
    for e in attached:
        pool[e] = pool.get(e, 0) + 1
    deficit = 0
    n_colorless = 0
    for r in required:
        if r == _E_COLORLESS:
            n_colorless += 1
            continue
        if pool.get(r, 0) > 0:
            pool[r] -= 1
        elif pool.get(_E_RAINBOW, 0) > 0:
            pool[_E_RAINBOW] -= 1
        elif r in (_E_PSYCHIC, _E_DARKNESS) and pool.get(_E_TEAM_ROCKET, 0) > 0:
            pool[_E_TEAM_ROCKET] -= 1
        else:
            deficit += 1
    remaining = sum(pool.values())
    deficit += max(0, n_colorless - remaining)
    return deficit


def _damage_vs(attack, attacker_cd, defender_cd) -> int:
    """Printed damage with weakness x2 / resistance -30 vs a known defender.

    Weakness x2 and the 1:1 base rule were verified from visible episode
    logs (modal ATTACK->HP_CHANGE ratios 2.0 / 1.0); resistance -30 is the
    standard-format rule, unconfirmed in our sparse log sample -- treated
    as an approximation. Variable-damage attacks contribute their printed
    base (often 0): a deliberate lower bound, the attack-id embedding
    still identifies them to the model.
    """
    dmg = attack.damage or 0
    if dmg <= 0:
        return 0
    if defender_cd is not None:
        if defender_cd.weakness is not None and defender_cd.weakness == attacker_cd.energyType:
            dmg *= 2
        if defender_cd.resistance is not None and defender_cd.resistance == attacker_cd.energyType:
            dmg = max(dmg - 30, 0)
    return dmg


def _attack_profile(pokemon, defender) -> tuple[int, int]:
    """(best payable damage vs defender, min energy deficit over attacks).

    `pokemon`/`defender` are cg Pokemon dataclasses (defender may be None
    -- facedown or empty active -- in which case damage skips modifiers).
    Returns (0, _DEFICIT_CAP) when the attacker is None/unknown/attackless.
    """
    card_db, attack_db = _static_db()
    if pokemon is None:
        return 0, _DEFICIT_CAP
    cd = card_db.get(getattr(pokemon, "id", None))
    if cd is None or not cd.attacks:
        return 0, _DEFICIT_CAP
    defender_cd = card_db.get(getattr(defender, "id", None)) if defender is not None else None
    attached = list(getattr(pokemon, "energies", None) or [])
    best_dmg = 0
    min_def = _DEFICIT_CAP
    for atk_id in cd.attacks:
        atk = attack_db.get(atk_id)
        if atk is None:
            continue
        deficit = _energy_deficit(list(atk.energies or []), attached)
        min_def = min(min_def, deficit)
        if deficit == 0:
            best_dmg = max(best_dmg, _damage_vs(atk, cd, defender_cd))
    return best_dmg, min_def


def _ko_turns(hp: int | None, dmg: int) -> int:
    """Turns of this incoming damage rate until KO, capped (cap = 'not soon')."""
    if not hp or hp <= 0:
        return 0
    if dmg <= 0:
        return _KO_TURNS_CAP
    return min(_KO_TURNS_CAP, -(-hp // dmg))


def _prize_value(card_id) -> int:
    """Prizes taken for KOing this Pokemon: 3 megaEx / 2 ex / 1 (0 unknown)."""
    card_db, _ = _static_db()
    cd = card_db.get(card_id)
    if cd is None:
        return 0
    return 3 if cd.megaEx else 2 if cd.ex else 1


def _extra_global_features(state: State, my_index: int) -> list[float]:
    """All GLOBAL_FEATURE_SPECS groups, in spec order. Pure visible-state math."""
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None

    my_dmg, my_def = _attack_profile(my_active, opp_active)
    opp_dmg, opp_def = _attack_profile(opp_active, my_active)
    t_my_ko = _ko_turns(getattr(my_active, "hp", 0), opp_dmg)
    t_opp_ko = _ko_turns(getattr(opp_active, "hp", 0), my_dmg)
    # It is my decision, so I move first: I win the race on ties.
    race_win = 1.0 if (my_dmg > 0 and opp_active is not None and t_opp_ko <= t_my_ko) else 0.0

    pv_opp = _prize_value(getattr(opp_active, "id", None)) if opp_active is not None else 0
    pv_mine = _prize_value(getattr(my_active, "id", None)) if my_active is not None else 0
    ko_opp_wins = 1.0 if pv_opp > 0 and pv_opp >= len(me.prize) else 0.0
    my_ko_loses = 1.0 if pv_mine > 0 and pv_mine >= len(opp.prize) else 0.0

    bench_def = min(
        (_attack_profile(p, opp_active)[1] for p in (me.bench or [])),
        default=_DEFICIT_CAP,
    )

    return [
        # ko_race
        t_my_ko / _KO_TURNS_CAP,
        t_opp_ko / _KO_TURNS_CAP,
        race_win,
        min(my_dmg / _DMG_NORM, 2.0),
        min(opp_dmg / _DMG_NORM, 2.0),
        # prize_race
        pv_opp / 3.0,
        pv_mine / 3.0,
        ko_opp_wins,
        my_ko_loses,
        # energy_deficit
        my_def / _DEFICIT_CAP,
        bench_def / _DEFICIT_CAP,
        opp_def / _DEFICIT_CAP,
        # status_conditions
        1.0 if me.poisoned else 0.0,
        1.0 if me.burned else 0.0,
        1.0 if me.asleep else 0.0,
        1.0 if me.paralyzed else 0.0,
        1.0 if me.confused else 0.0,
        1.0 if opp.poisoned else 0.0,
        1.0 if opp.burned else 0.0,
        1.0 if opp.asleep else 0.0,
        1.0 if opp.paralyzed else 0.0,
        1.0 if opp.confused else 0.0,
    ]  # length must equal N_EXTRA_GLOBAL; pinned by tests/test_il_features.py


class DecisionMeta(NamedTuple):
    """Per-row provenance for outcome weighting / filtering (Stage 2, REWEIGHT).

    ``outcome`` is the ACTING seat's terminal result remapped to
    0.0 = loss, 0.5 = draw, 1.0 = win, or -1.0 when the episode carries no
    usable ``rewards`` pair (crashed/truncated dumps -- callers that weight
    by outcome must treat -1.0 as "exclude", never as a weight). The raw
    cabt -1/0/1 value deliberately never leaves this module: negative
    terminal rewards are a known pathology magnet (bc_pipeline_v2 §8.4).
    """

    episode_id: int  # int(filename stem); -1 if the stem is not numeric
    seat: int  # agent_idx: 0 or 1
    outcome: float  # 0.0 loss / 0.5 draw / 1.0 win / -1.0 unknown
    turn: int  # obs["current"]["turn"] at this decision; -1 if absent


def _seat_outcome(rewards: object, seat: int) -> float:
    """Remap the episode-level rewards pair to this seat's {0, 0.5, 1} outcome.

    A ``None`` reward means this seat errored out -- a loss, matching
    ``winning_agent``'s treatment of the same case (None = lowest value).
    """
    if not isinstance(rewards, list) or len(rewards) != 2:
        return -1.0
    r = rewards[seat]
    if r is None:
        return 0.0
    if not isinstance(r, (int, float)):
        return -1.0
    if r > 0:
        return 1.0
    if r < 0:
        return 0.0
    return 0.5


def load_manifest_scores(manifest_path: Path | None = None) -> dict[int, float]:
    """episode_id -> avg_score (the player-rating field) from manifest.csv.

    Loaded once and held as a plain dict -- never re-read per row. Episodes
    absent from the manifest simply have no entry; ILDataset substitutes a
    -1.0 sentinel (avg_score is always positive in the real manifest).
    """
    path = manifest_path or (config.EPISODES_DIR / "manifest.csv")
    scores: dict[int, float] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                scores[int(row["episode_id"])] = float(row["avg_score"])
            except (KeyError, ValueError):
                continue
    return scores


def resolve_split_dir(split: str) -> Path:
    """Resolve 'train' / 'eval' to its folder via data/episodes/splits/splits.json."""
    splits_json = config.EPISODES_DIR / "splits" / "splits.json"
    meta = json.loads(splits_json.read_text())[split]
    return config.EPISODES_DIR / meta["folder"]


def split_meta(split: str) -> tuple[str, list[str], int]:
    """splits.json key -> (hub split kind, calendar days, expected episode count).

    Kind is 'train' or 'eval' -- the ``{split}/day=.../`` prefix on the Hub
    and the prefix of the local folder name. Multi-day unions (e.g.
    ``train_combined``) list their days in a ``dates`` field; single-day
    splits derive the day from the folder name (``train-2026-07-26`` ->
    ``2026-07-26``), mirroring pack_episodes.py. This is how a splits.json
    key selects Hub days for ShardILDataset without hand-listing them, so
    hub-sourced training follows the SAME split definition as local raw --
    not simply "every day currently uploaded under train/" (which would
    silently absorb new ingest days into the corpus).
    """
    splits_json = config.EPISODES_SPLITS_DIR / "splits.json"
    meta = json.loads(splits_json.read_text())[split]
    folder_name = Path(meta["folder"]).name
    kind = folder_name.split("-", 1)[0]
    days = meta.get("dates") or [folder_name.split("-", 1)[1]]
    return kind, days, int(meta["episodes"])


def _card_features(card: object | None) -> tuple[int, float, float, float]:
    """(card_id, hp_frac, energy_norm, tool_norm) for a Card or Pokemon, 0s if None."""
    if card is None:
        return 0, 0.0, 0.0, 0.0
    card_id = _clamp_id(getattr(card, "id", 0) or 0, CARD_VOCAB_SIZE)
    hp = getattr(card, "hp", None)
    max_hp = getattr(card, "maxHp", None)
    hp_frac = (hp / max_hp) if hp is not None and max_hp else 0.0
    energies = getattr(card, "energies", None)
    energy_norm = (len(energies) / 6.0) if energies is not None else 0.0
    tools = getattr(card, "tools", None)
    tool_norm = (len(tools) / 3.0) if tools is not None else 0.0
    return card_id, hp_frac, energy_norm, tool_norm


def _get_card(
    state: State,
    select: SelectData,
    area: int | None,
    index: int | None,
    player_index: int,
):
    """Best-effort lookup of the Card/Pokemon an option field points at.

    Used only to derive soft scoring features, never for legality -- the
    game engine has already guaranteed every option is legal, so a wrong
    guess here (e.g. assuming a DISCARD option targets your own side when it
    targets the opponent's) only costs a slightly noisier feature, not a
    correctness bug.
    """
    if area is None or index is None:
        return None
    try:
        if area == AreaType.HAND:
            hand = state.players[player_index].hand
            return hand[index] if hand else None
        if area == AreaType.ACTIVE:
            active = state.players[player_index].active
            return active[index] if index < len(active) else None
        if area == AreaType.BENCH:
            bench = state.players[player_index].bench
            return bench[index] if index < len(bench) else None
        if area == AreaType.DISCARD:
            return state.players[player_index].discard[index]
        if area == AreaType.PRIZE:
            return state.players[player_index].prize[index]
        if area == AreaType.STADIUM:
            return state.stadium[index]
        if area == AreaType.DECK:
            return select.deck[index] if select.deck else None
        if area == AreaType.LOOKING:
            return state.looking[index] if state.looking else None
    except (IndexError, TypeError):
        return None
    return None


def _resolve_option_refs(option, my_index: int):
    """Return up to two (area, index, player_index) refs an option carries.

    ref1 is "the card this option is fundamentally about" (the card played /
    attached / evolved / discarded / whose ability fires). ref2 is only
    populated for ATTACH/EVOLVE, where the option also names an in-play
    Pokemon target distinct from the source card.
    """
    ot = option.type
    ref1 = ref2 = None
    if ot == OptionType.PLAY:
        ref1 = (AreaType.HAND, option.index, my_index)
    elif ot in (
        OptionType.CARD,
        OptionType.TOOL_CARD,
        OptionType.ENERGY_CARD,
        OptionType.ENERGY,
    ):
        pi = option.playerIndex if option.playerIndex is not None else my_index
        ref1 = (option.area, option.index, pi)
    elif ot in (OptionType.ATTACH, OptionType.EVOLVE):
        ref1 = (option.area, option.index, my_index)
        ref2 = (option.inPlayArea, option.inPlayIndex, my_index)
    elif ot in (OptionType.ABILITY, OptionType.DISCARD):
        ref1 = (option.area, option.index, my_index)
    return ref1, ref2


def _min_deficit(pokemon, extra_energies: list[int]) -> int:
    """Min energy deficit over a Pokemon's attacks, with hypothetical extra energy."""
    card_db, attack_db = _static_db()
    if pokemon is None:
        return _DEFICIT_CAP
    cd = card_db.get(getattr(pokemon, "id", None))
    if cd is None or not cd.attacks:
        return _DEFICIT_CAP
    attached = list(getattr(pokemon, "energies", None) or []) + extra_energies
    best = _DEFICIT_CAP
    for atk_id in cd.attacks:
        atk = attack_db.get(atk_id)
        if atk is None:
            continue
        best = min(best, _energy_deficit(list(atk.energies or []), attached))
    return best


# int values pinned by cg.api enums; kept as ints so this stays pure arithmetic
_OT_CARD, _OT_ATTACH, _OT_ATTACK = 3, 8, 13
_CTX_SWITCH, _CTX_TO_ACTIVE = 3, 4
_AREA_ACTIVE, _AREA_BENCH = 4, 5
_CT_BASIC_ENERGY, _CT_SPECIAL_ENERGY = 5, 6


def _extra_opt_row(state: State, select: SelectData, option, my_index: int) -> list[float]:
    """All OPT_FEATURE_SPECS groups for one option, in spec order.

    Rows are zero except where the group applies (ATTACK options for
    attack_tactical, ATTACH for attach_enable, own-Pokemon CARD options in
    switch/promote contexts for retreat_switch). Uses only visible state +
    the static card DB -- see _extra_global_features.
    """
    row = [0.0] * N_EXTRA_OPT
    card_db, attack_db = _static_db()
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    ot = int(option.type)

    if ot == _OT_ATTACK and option.attackId is not None:
        atk = attack_db.get(option.attackId)
        my_cd = card_db.get(getattr(my_active, "id", None)) if my_active is not None else None
        opp_cd = card_db.get(getattr(opp_active, "id", None)) if opp_active is not None else None
        if atk is not None and my_cd is not None:
            dmg = _damage_vs(atk, my_cd, opp_cd)
            opp_hp = getattr(opp_active, "hp", 0) or 0
            kos_now = dmg > 0 and opp_hp > 0 and dmg >= opp_hp
            wins = kos_now and _prize_value(opp_active.id) >= len(me.prize)
            row[0] = min(dmg / _DMG_NORM, 2.0)
            row[1] = 1.0 if kos_now else 0.0
            row[2] = 1.0 if wins else 0.0

    elif ot == _OT_ATTACH:
        attached_card = _get_card(state, select, option.area, option.index, my_index)
        target = _get_card(state, select, option.inPlayArea, option.inPlayIndex, my_index)
        cd = card_db.get(getattr(attached_card, "id", None)) if attached_card is not None else None
        if cd is not None and target is not None:
            if cd.cardType == _CT_BASIC_ENERGY:
                new_energy = [int(cd.energyType)]
            elif cd.cardType == _CT_SPECIAL_ENERGY:
                new_energy = [_E_RAINBOW]  # approximation: special energy as wildcard
            else:
                new_energy = None  # tool etc.: no energy timeline change
            if new_energy is not None:
                pre = _min_deficit(target, [])
                post = _min_deficit(target, new_energy)
                row[3] = post / _DEFICIT_CAP
                row[4] = 1.0 if (pre > 0 and post == 0) else 0.0

    elif (
        ot == _OT_CARD
        and int(select.context) in (_CTX_SWITCH, _CTX_TO_ACTIVE)
        and (option.playerIndex is None or option.playerIndex == my_index)
        and option.area in (_AREA_ACTIVE, _AREA_BENCH)
    ):
        candidate = _get_card(state, select, option.area, option.index, my_index)
        if candidate is not None and getattr(candidate, "hp", None) is not None:
            incoming, _ = _attack_profile(opp_active, candidate)
            outgoing, _ = _attack_profile(candidate, opp_active)
            row[5] = _ko_turns(candidate.hp, incoming) / _KO_TURNS_CAP
            row[6] = min(outgoing / _DMG_NORM, 2.0)

    return row


def _build_state_slots(
    state: State, my_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    slot_card_id = [0] * N_STATE_SLOTS
    slot_scalar = [[0.0] * N_SLOT_SCALARS for _ in range(N_STATE_SLOTS)]
    pos = 0

    def fill(card) -> None:
        nonlocal pos
        cid, hp_f, e_n, t_n = _card_features(card)
        slot_card_id[pos] = cid
        slot_scalar[pos] = [hp_f, e_n, t_n, 1.0 if card is not None else 0.0]
        pos += 1

    me = state.players[my_index]
    opp = state.players[1 - my_index]

    fill(me.active[0] if me.active else None)
    for i in range(MAX_BENCH):
        fill(me.bench[i] if i < len(me.bench) else None)
    fill(opp.active[0] if opp.active else None)
    for i in range(MAX_BENCH):
        fill(opp.bench[i] if i < len(opp.bench) else None)
    hand = me.hand or []
    for i in range(MAX_HAND):
        fill(hand[i] if i < len(hand) else None)

    assert pos == N_STATE_SLOTS
    return (
        torch.tensor(slot_card_id, dtype=torch.long),
        torch.tensor(slot_scalar, dtype=torch.float32),
    )


def _global_scalars(state: State, my_index: int, n_options: int) -> torch.Tensor:
    me = state.players[my_index]
    opp = state.players[1 - my_index]
    vals = [
        state.turn / 50.0,
        state.turnActionCount / 10.0,
        len(me.prize) / 6.0,
        len(opp.prize) / 6.0,
        me.deckCount / 60.0,
        opp.deckCount / 60.0,
        me.handCount / 10.0,
        opp.handCount / 10.0,
        1.0 if state.supporterPlayed else 0.0,
        1.0 if state.stadiumPlayed else 0.0,
        1.0 if state.energyAttached else 0.0,
        1.0 if state.retreated else 0.0,
        min(n_options, MAX_OPTIONS) / MAX_OPTIONS,
    ]
    assert len(vals) == N_GLOBAL_SCALARS
    return torch.tensor(vals, dtype=torch.float32)


def _encode_options(
    state: State, select: SelectData, my_index: int, add_decline: bool, exclude: frozenset
):
    """Encode option slots, optionally appending a synthetic DECLINE slot.

    `add_decline` reserves one slot (at index n, right after the real
    options) representing "submit an empty selection" -- legal whenever
    `select.minCount == 0`. It carries no card/attack identity, just its own
    learned DECLINE_OPTION_TYPE embedding, and is scored by the exact same
    head as every real option (Pattern B, unchanged).

    `exclude` marks option indices to hide from this decision even though
    they're real and legal -- used by the multi-select autoregressive
    unroll (1.3) to re-mask options already picked in a prior step of the
    same underlying decision.
    """
    budget = MAX_OPTIONS - (1 if add_decline else 0)
    n = min(len(select.option), budget)
    opt_type = [0] * MAX_OPTIONS
    opt_attack = [0] * MAX_OPTIONS
    opt_special = [0] * MAX_OPTIONS
    opt_scalar = [[0.0] * N_OPT_SCALARS for _ in range(MAX_OPTIONS)]
    opt_ref_card_id = [[0, 0] for _ in range(MAX_OPTIONS)]
    opt_ref_scalar = [[0.0] * (2 * N_REF_SCALARS) for _ in range(MAX_OPTIONS)]
    opt_extra = [[0.0] * N_EXTRA_OPT for _ in range(MAX_OPTIONS)]
    opt_mask = [False] * MAX_OPTIONS

    for i in range(n):
        o = select.option[i]
        opt_extra[i] = _extra_opt_row(state, select, o, my_index)
        opt_type[i] = int(o.type)
        opt_attack[i] = _clamp_id(o.attackId or 0, ATTACK_VOCAB_SIZE)
        opt_special[i] = (
            (o.specialConditionType + 1) if o.specialConditionType is not None else 0
        )
        opt_scalar[i][0] = (o.number / 10.0) if o.number is not None else 0.0

        ref1, ref2 = _resolve_option_refs(o, my_index)
        opt_scalar[i][1] = 1.0 if ref1 is not None else 0.0
        if ref1 is not None:
            card = _get_card(state, select, *ref1)
            cid, hp_f, e_n, t_n = _card_features(card)
            opt_ref_card_id[i][0] = cid
            opt_ref_scalar[i][0:3] = [hp_f, e_n, t_n]
        if ref2 is not None:
            card = _get_card(state, select, *ref2)
            cid, hp_f, e_n, t_n = _card_features(card)
            opt_ref_card_id[i][1] = cid
            opt_ref_scalar[i][3:6] = [hp_f, e_n, t_n]
        opt_mask[i] = i not in exclude

    if add_decline:
        opt_type[n] = DECLINE_OPTION_TYPE
        opt_mask[n] = True

    return (
        torch.tensor(opt_type, dtype=torch.long),
        torch.tensor(opt_attack, dtype=torch.long),
        torch.tensor(opt_special, dtype=torch.long),
        torch.tensor(opt_scalar, dtype=torch.float32),
        torch.tensor(opt_ref_card_id, dtype=torch.long),
        torch.tensor(opt_ref_scalar, dtype=torch.float32),
        torch.tensor(opt_extra, dtype=torch.float32),
        torch.tensor(opt_mask, dtype=torch.bool),
        n,
    )


def encode_observation(
    obs_dict: dict, exclude: frozenset = frozenset()
) -> dict[str, torch.Tensor] | None:
    """Encode one raw obs_dict (``current`` + ``select``) into fixed-size tensors.

    Returns None if there is no decision to make here (``select`` is None --
    e.g. the initial deck-submission step, which is out of scope for this
    move-selection policy) or ``select.option`` is empty.

    A synthetic DECLINE slot is appended whenever ``select.minCount == 0``
    (see ``_encode_options``); its index is always ``n_real_options``.
    ``exclude`` re-masks already-picked option indices for the multi-select
    autoregressive unroll (1.3) -- see ``iter_decisions``.
    """
    select_dict = obs_dict.get("select")
    current_dict = obs_dict.get("current")
    if not select_dict or not current_dict:
        return None
    state: State = to_dataclass(current_dict, State)
    select: SelectData = to_dataclass(select_dict, SelectData)
    if not select.option:
        return None
    my_index = state.yourIndex
    # Decline is legal at this step iff the remaining minimum is already
    # satisfied by picks made earlier in this same multi-select unroll
    # (`exclude`, see iter_decisions) -- NOT simply `minCount == 0` in
    # isolation, which only covers the maxCount==1 case (exclude is always
    # empty there, so this reduces to the same check).
    add_decline = (select.minCount or 0) <= len(exclude)

    slot_card_id, slot_scalar = _build_state_slots(state, my_index)
    (
        opt_type,
        opt_attack,
        opt_special,
        opt_scalar,
        opt_ref_card_id,
        opt_ref_scalar,
        opt_extra,
        opt_mask,
        n_real,
    ) = _encode_options(state, select, my_index, add_decline, exclude)
    global_scalar = _global_scalars(state, my_index, len(select.option))
    extra_global = torch.tensor(
        _extra_global_features(state, my_index), dtype=torch.float32
    )

    select_context = min(int(select.context), SELECT_CONTEXT_VOCAB_SIZE - 1)
    return {
        "slot_card_id": slot_card_id,
        "slot_scalar": slot_scalar,
        "global_scalar": global_scalar,
        "extra_global": extra_global,
        "select_type": torch.tensor(int(select.type), dtype=torch.long),
        "select_context": torch.tensor(select_context, dtype=torch.long),
        "opt_type": opt_type,
        "opt_attack": opt_attack,
        "opt_special": opt_special,
        "opt_scalar": opt_scalar,
        "opt_ref_card_id": opt_ref_card_id,
        "opt_ref_scalar": opt_ref_scalar,
        "extra_opt": opt_extra,
        "opt_mask": opt_mask,
        "n_real_options": n_real,
    }


def _decision_signature(obs: dict) -> tuple:
    """A cheap fingerprint of "is this the same decision as the previous tick".

    kaggle-environments' cabt interpreter only refreshes the OBSERVATION of
    whichever player is currently active (see
    kaggle_environments/envs/cabt/cabt.py's `interpreter()`); the inactive
    player's `select` is simply never cleared, so it re-appears unchanged at
    every subsequent raw tick until that player is active again. Naively
    treating every tick with `select is not None` as a fresh decision means
    ~47% of "decisions" are stale echoes of an already-answered select, not
    new ones (confirmed empirically: a 150-episode signature-repeat vs.
    empty-paired-action cross-tab showed 0 false negatives/positives -- see
    notes/phase0_discovery_report.md). This used to be handled only as an
    emergent side effect of requiring `len(action) == 1` downstream, which
    silently breaks the moment declines (`len(action) == 0`) are allowed
    back in as a real label. Filtering on signature-repeat directly makes
    the dedup explicit and independent of that downstream length check.
    """
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    return (
        cur.get("turn"),
        cur.get("turnActionCount"),
        sel.get("context"),
        sel.get("minCount"),
        sel.get("maxCount"),
        len(sel.get("option", [])),
    )


def winning_agent(episode: dict) -> int | None:
    """Return the agent index that won `episode`, or None if there is no winner.

    Winner = the unique-largest entry in `episode["rewards"]`. A `None` reward
    (opponent errored out) is treated as the lowest possible value, so the other
    seat still counts as the winner. Draws and any tie for the max (e.g. the
    single `(0, 0)` game in 2026-07-01) return None -- no seat is clonable.
    """
    rewards = episode.get("rewards") if isinstance(episode, dict) else None
    if not rewards:
        return None
    scored = [float("-inf") if r is None else r for r in rewards]
    top = max(scored)
    winners = [i for i, v in enumerate(scored) if v == top]
    return winners[0] if len(winners) == 1 else None


def iter_decisions(
    split_dir: Path, max_episodes: int | None = None, winner_only: bool = False
) -> Iterator[tuple[dict, int, frozenset, DecisionMeta]]:
    """Yield (obs_dict, chosen_index, exclude, meta) for every real decision in a split.

    `meta` is a `DecisionMeta` carrying the acting seat, episode id, turn, and
    the seat's terminal `outcome` from the episode-level `rewards` pair -- the
    Stage-2 (REWEIGHT) fields this iterator previously dropped: both seats of
    every match used to be cloned identically with zero reference to who won.
    `winner_only=True` is the hard-filter special case (S2-E1); `meta.outcome`
    is what the soft-weighting arms (S2-E2/E3) consume.

    `chosen_index` indexes into `select.option`, EXCEPT it equals
    `len(select.option)` (the synthetic DECLINE slot, see `encode_observation`)
    when the recorded response was a legal empty selection
    (`minCount == 0`, `action == []`).

    `exclude` is the set of option indices to mask out because they were
    already picked earlier in the same multi-select decision (see below);
    empty for ordinary single-choice decisions.

    Pairing quirk (verified against the actual environment source, see
    `_decision_signature`): the response to `decisions[i]`'s select is
    logged one tick later, at `decisions[i + 1]`'s action field. Stale
    echoes of an already-answered decision (the inactive player's carried-
    forward observation) are dropped explicitly via signature-repeat
    detection, not left to an incidental length check.

    Multi-select (`maxCount > 1`) decisions are unrolled autoregressively:
    the recorded action list is treated as a pick order (the engine itself
    only records the final atomic set, so this is the best available proxy
    for "which was picked first"), each pick becomes one training example
    with all earlier picks in `exclude`, and -- if the player stopped
    before `maxCount` and `minCount` permitted stopping -- one trailing
    DECLINE-labeled example is emitted representing "no more picks."
    """
    files = sorted(split_dir.glob("*.json"))
    if max_episodes is not None:
        files = files[:max_episodes]
    for path in files:
        try:
            episode = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        episode_id = int(path.stem) if path.stem.isdigit() else -1
        yield from iter_episode_decisions(
            episode, episode_id=episode_id, winner_only=winner_only
        )


def iter_episode_decisions(
    episode: dict | list, episode_id: int = -1, winner_only: bool = False
) -> Iterator[tuple[dict, int, frozenset, DecisionMeta]]:
    """Yield (obs_dict, chosen_index, exclude, meta) for every real decision in
    ONE parsed episode -- the shared core behind ``iter_decisions`` (raw JSON
    folders) and ``ShardILDataset`` (packed Parquet shards, ADR-001). Label,
    meta, and unrolling semantics are documented on ``iter_decisions``.
    """
    steps = episode.get("steps") if isinstance(episode, dict) else episode
    if steps:
        rewards = episode.get("rewards") if isinstance(episode, dict) else None
        if winner_only:
            w = winning_agent(episode)
            if w is None:
                return  # draw / no unique winner -- nothing to clone
            seats: tuple[int, ...] = (w,)
        else:
            seats = (0, 1)
        for agent_idx in seats:
            outcome = _seat_outcome(rewards, agent_idx)
            decisions: list[tuple[dict, list[int] | None]] = []
            for step in steps:
                if agent_idx >= len(step):
                    continue
                agent_step = step[agent_idx]
                obs = agent_step.get("observation")
                if not obs or not obs.get("select"):
                    continue
                decisions.append((obs, agent_step.get("action")))

            prev_sig = None
            for i in range(len(decisions) - 1):
                obs, _ = decisions[i]
                sig = _decision_signature(obs)
                if sig == prev_sig:
                    prev_sig = sig
                    continue  # stale echo, not a fresh decision
                prev_sig = sig

                # the response to `obs`'s select is logged one tick later (see above)
                _, action = decisions[i + 1]
                select = obs["select"]
                n_opts = len(select.get("option", []))
                min_count = select.get("minCount") or 0
                max_count = select.get("maxCount")
                if action is None:
                    continue

                turn = (obs.get("current") or {}).get("turn")
                meta = DecisionMeta(
                    episode_id=episode_id,
                    seat=agent_idx,
                    outcome=outcome,
                    turn=turn if isinstance(turn, int) else -1,
                )

                if max_count == 1:
                    if len(action) == 1:
                        label = action[0]
                        if label >= n_opts or label >= MAX_OPTIONS:
                            continue
                        yield obs, label, frozenset(), meta
                    elif len(action) == 0 and min_count == 0:
                        if n_opts >= MAX_OPTIONS:
                            continue  # no room for the synthetic DECLINE slot
                        yield obs, n_opts, frozenset(), meta
                    # else: anomalous length for a maxCount==1 response -- drop
                    continue

                if max_count is not None and max_count > 1:
                    if any(a >= n_opts or a >= MAX_OPTIONS for a in action):
                        continue
                    excluded: set[int] = set()
                    for picked in action:
                        yield obs, picked, frozenset(excluded), meta
                        excluded.add(picked)
                    if len(action) < max_count and min_count <= len(action) and n_opts < MAX_OPTIONS:
                        yield obs, n_opts, frozenset(excluded), meta


def _decisions_to_features(
    decisions: Iterable[tuple[dict, int, frozenset, DecisionMeta]],
    with_meta: bool = False,
    scores: dict[int, float] | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Encode a decision stream into training examples, dropping non-decisions.

    ``with_meta=True`` attaches the ``ILDataset.META_KEYS`` weighting fields
    (see that docstring); ``scores`` is the ``load_manifest_scores()`` dict
    backing the ``avg_score`` field.
    """
    for obs, label, exclude, meta in decisions:
        feats = encode_observation(obs, exclude=exclude)
        if feats is None:
            continue
        feats.pop("n_real_options", None)
        feats["label"] = torch.tensor(label, dtype=torch.long)
        if with_meta:
            feats["outcome"] = torch.tensor(meta.outcome, dtype=torch.float32)
            feats["seat"] = torch.tensor(meta.seat, dtype=torch.long)
            feats["episode_id"] = torch.tensor(meta.episode_id, dtype=torch.long)
            feats["turn"] = torch.tensor(meta.turn, dtype=torch.long)
            feats["avg_score"] = torch.tensor(
                (scores or {}).get(meta.episode_id, -1.0), dtype=torch.float32
            )
        yield feats


def _shuffled(examples: Iterator, buffer_size: int, rng: random.Random) -> Iterator:
    """Pass-through when buffer_size <= 1; otherwise pool-and-drain shuffling.

    Same scheme ILDataset has always used: fill a pool of ``buffer_size``
    examples, shuffle it, emit it all, repeat -- decorrelates neighbors
    (same episode / same match) without holding the whole split in memory.
    """
    if buffer_size <= 1:
        yield from examples
        return
    buf: list = []
    for ex in examples:
        buf.append(ex)
        if len(buf) >= buffer_size:
            rng.shuffle(buf)
            yield from buf
            buf = []
    if buf:
        rng.shuffle(buf)
        yield from buf


class ILDataset(IterableDataset):
    """Streams (features, label) examples from a split folder.

    A small in-memory shuffle buffer decorrelates consecutive examples
    (which otherwise come from the same episode / same match) without
    loading the whole split into memory -- 4,554 + 4,430 episodes is large.
    Use ``shuffle_buffer=1`` for the eval split (deterministic order).

    ``with_meta=True`` (Stage 2, REWEIGHT) additionally attaches per-row
    weighting fields: ``outcome`` (0/0.5/1, -1 unknown), ``seat``,
    ``episode_id``, ``turn``, and ``avg_score`` (the manifest player-rating
    field; -1.0 when the episode is not in the manifest). ⚠️ The existing
    train loop calls ``model(**batch)`` with the whole batch dict, so a
    weighted trainer must pop these five keys before the forward pass --
    which is why the flag defaults to False instead of always-on.
    """

    META_KEYS = ("outcome", "seat", "episode_id", "turn", "avg_score")

    def __init__(
        self,
        split_dir: Path,
        max_episodes: int | None = None,
        shuffle_buffer: int = 2000,
        seed: int = 0,
        winner_only: bool = False,
        with_meta: bool = False,
    ) -> None:
        self.split_dir = split_dir
        self.max_episodes = max_episodes
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.winner_only = winner_only
        self.with_meta = with_meta

    def __iter__(self):
        rng = random.Random(self.seed)
        scores = load_manifest_scores() if self.with_meta else None
        decisions = iter_decisions(
            self.split_dir, self.max_episodes, winner_only=self.winner_only
        )
        feats = _decisions_to_features(decisions, with_meta=self.with_meta, scores=scores)
        yield from _shuffled(feats, self.shuffle_buffer, rng)


def _resolve_hub_shards(repo_id: str, split: str, days: list[str] | None) -> list[str]:
    """Sorted repo-relative shard paths for a split on the Hub.

    The held-out-DAY discipline lives in the repo layout
    (``{split}/day=YYYY-MM-DD/shard-NNN.parquet``), so restricting training
    to specific days is a pure path filter -- days never mix across splits.
    """
    from huggingface_hub import HfApi  # lazy: only the Hub path needs it

    pat = re.compile(
        rf"^{re.escape(split)}/day=(\d{{4}}-\d{{2}}-\d{{2}})/shard-\d+\.parquet$"
    )
    return sorted(
        f
        for f in HfApi().list_repo_files(repo_id, repo_type="dataset")
        if (m := pat.match(f)) and (days is None or m.group(1) in days)
    )


class ShardILDataset(IterableDataset):
    """Same (features, label) stream as ILDataset, but read from the zstd
    Parquet episode shards of ADR-001 instead of raw JSON folders.

    Shards come from the private Hub repo (``config.HF_EPISODES_REPO``) or a
    local directory with the same layout (``{split}/day=YYYY-MM-DD/
    shard-NNN.parquet``). Hub shards are fetched one file at a time with
    ``hf_hub_download`` into the standard huggingface_hub cache: the first
    pass over the corpus pays the network cost, every later pass reads the
    cached file at SSD speed. Rows are decompressed in memory, encoded, and
    discarded -- the raw corpus never lands on disk.

    With ``DataLoader(num_workers=N)`` the shard list is split evenly across
    workers (worker w reads shards w, w+N, ...), so full parallelism needs
    at least as many shards as workers (see ``n_shards``). ``max_episodes``
    caps episodes PER WORKER in that case. Call ``set_epoch(e)`` before each
    re-iteration so the shard order and shuffle rng differ per pass; with
    ``shuffle_buffer <= 1`` iteration is deterministic (sorted shards, pack
    order) for eval.

    ``winner_only`` and ``with_meta`` mirror ILDataset's REWEIGHT plumbing
    (rl_pipeline_v2 §2.B0): meta rows carry outcome/seat/episode_id/turn plus
    ``avg_score`` joined from a freshly loaded manifest via the shard's
    ``episode_id`` column — NOT from the shard's baked score columns, which
    are null forever for any day packed before its manifest rows were
    backfilled (2026-07-26).
    """

    def __init__(
        self,
        split: str,
        days: list[str] | None = None,
        repo_id: str | None = None,
        local_root: Path | None = None,
        max_episodes: int | None = None,
        shuffle_buffer: int = 2000,
        seed: int = 0,
        winner_only: bool = False,
        with_meta: bool = False,
        episode_ids: set[int] | None = None,
    ) -> None:
        self.split = split
        self.repo_id = repo_id or config.HF_EPISODES_REPO
        self.local_root = local_root
        self.max_episodes = max_episodes
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.winner_only = winner_only
        self.with_meta = with_meta
        # Optional allowlist (skill-filtered demonstrations): only episodes
        # whose shard `episode_id` is in this set are decoded/streamed. None
        # = no filter. Filtering happens BEFORE max_episodes counting.
        self.episode_ids = episode_ids
        self._epoch = 0
        if local_root is not None:
            self.files = sorted(
                str(p.relative_to(local_root))
                for p in (local_root / split).glob("day=*/shard-*.parquet")
                if days is None or p.parent.name.removeprefix("day=") in days
            )
        else:
            self.files = _resolve_hub_shards(self.repo_id, split, days)
        if not self.files:
            raise FileNotFoundError(
                f"no parquet shards for split {split!r}"
                + (f", days {days}" if days else "")
                + (f" under {local_root}" if local_root else f" in {self.repo_id}")
            )

    @property
    def n_shards(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _local_path(self, rel: str) -> str:
        if self.local_root is not None:
            return str(self.local_root / rel)
        from huggingface_hub import hf_hub_download  # lazy: only the Hub path needs it

        return hf_hub_download(self.repo_id, rel, repo_type="dataset")

    def n_episodes(self) -> int:
        """Total episode rows, read from shard footers only (no full downloads)."""
        import pyarrow.parquet as pq  # lazy: keep the evaluator bundle pyarrow-free

        if self.local_root is not None:
            return sum(
                pq.ParquetFile(self.local_root / f).metadata.num_rows for f in self.files
            )
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        total = 0
        for f in self.files:
            with fs.open(f"datasets/{self.repo_id}/{f}", "rb") as fh:
                total += pq.ParquetFile(fh).metadata.num_rows
        return total

    def __iter__(self):
        import pyarrow.parquet as pq  # lazy: keep the evaluator bundle pyarrow-free

        worker = torch.utils.data.get_worker_info()
        wid, n_workers = (worker.id, worker.num_workers) if worker else (0, 1)

        files = list(self.files)
        if self.shuffle_buffer > 1:
            # Integer arithmetic, NOT hash(tuple): spawned worker processes get
            # different PYTHONHASHSEEDs, and every worker must derive the SAME
            # permutation for the strided split below to stay disjoint.
            random.Random(self.seed * 1_000_003 + self._epoch).shuffle(files)
        files = files[wid::n_workers]
        rng = random.Random((self.seed * 1_000_003 + self._epoch) * 64 + wid)
        # avg_score joins from a FRESHLY loaded manifest at iteration time, by
        # the shard's own episode_id column -- the score columns baked into the
        # shards are stale for any day packed before its manifest rows existed
        # (2026-07-26 was packed with zero coverage; see rl_pipeline_v2 §2.B0).
        scores = load_manifest_scores() if self.with_meta else None

        def episodes() -> Iterator[tuple[int, dict]]:
            n = 0
            for rel in files:
                pf = pq.ParquetFile(self._local_path(rel))
                # row groups are small by construction (pack_episodes.py uses
                # ~32 rows), so this holds a few MB decompressed at a time
                for rb in pf.iter_batches(
                    batch_size=8, columns=["episode_id", "episode_json"]
                ):
                    for eid, raw in zip(
                        rb.column(0).to_pylist(), rb.column(1).to_pylist(), strict=True
                    ):
                        if self.episode_ids is not None and eid not in self.episode_ids:
                            continue
                        if self.max_episodes is not None and n >= self.max_episodes:
                            return
                        n += 1
                        yield int(eid), json.loads(raw)

        def features() -> Iterator[dict[str, torch.Tensor]]:
            for eid, episode in episodes():
                yield from _decisions_to_features(
                    iter_episode_decisions(
                        episode, episode_id=eid, winner_only=self.winner_only
                    ),
                    with_meta=self.with_meta,
                    scores=scores,
                )

        yield from _shuffled(features(), self.shuffle_buffer, rng)
