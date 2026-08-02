"""Streaming behavior-cloning dataset for the Pokemon TCG imitation-learning agent.

Reads episodes directly from the on-disk split folders named in
``data/episodes/splits/splits.json`` -- no downloading, no re-splitting.
Train uses ``train-2026-07-26/``, validation uses the held-out calendar day
``eval-2026-07-27/`` (see splits.json's "held-out-DAY rule": distinct days
avoid within-match leakage, so never shuffle across them).

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
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from . import config

# data/external/cg-lib is a gitignored local unpack, so a fresh clone (CI, a
# container, a remote session) has no cg at all. The in-repo beginner-guide
# copy ships the same cg/api.py with a Linux libcg.so -- correct on Linux,
# dlopen-crashes on macOS, hence the platform guard. cg-lib still wins when
# present, so mac behaviour is unchanged.
_CG_CANDS = [config.PROJECT_ROOT / "data" / "external" / "cg-lib"]
if sys.platform != "darwin":
    _CG_CANDS.append(config.PROJECT_ROOT / "notebooks" / "beginner-guide" / "sample-agent-output")
for _CG_DIR in _CG_CANDS:
    if (_CG_DIR / "cg" / "api.py").exists() and str(_CG_DIR) not in sys.path:
        sys.path.insert(0, str(_CG_DIR))
        break

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


def resolve_split_dir(split: str) -> Path:
    """Resolve 'train' / 'eval' to its folder via data/episodes/splits/splits.json."""
    splits_json = config.EPISODES_DIR / "splits" / "splits.json"
    meta = json.loads(splits_json.read_text())[split]
    return config.EPISODES_DIR / meta["folder"]


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
    opt_mask = [False] * MAX_OPTIONS

    for i in range(n):
        o = select.option[i]
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
        opt_mask,
        n_real,
    ) = _encode_options(state, select, my_index, add_decline, exclude)
    global_scalar = _global_scalars(state, my_index, len(select.option))

    select_context = min(int(select.context), SELECT_CONTEXT_VOCAB_SIZE - 1)
    return {
        "slot_card_id": slot_card_id,
        "slot_scalar": slot_scalar,
        "global_scalar": global_scalar,
        "select_type": torch.tensor(int(select.type), dtype=torch.long),
        "select_context": torch.tensor(select_context, dtype=torch.long),
        "opt_type": opt_type,
        "opt_attack": opt_attack,
        "opt_special": opt_special,
        "opt_scalar": opt_scalar,
        "opt_ref_card_id": opt_ref_card_id,
        "opt_ref_scalar": opt_ref_scalar,
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


def iter_decisions(
    split_dir: Path, max_episodes: int | None = None
) -> Iterator[tuple[dict, int, frozenset]]:
    """Yield (obs_dict, chosen_index, exclude) for every real decision in a split.

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
        steps = episode.get("steps") if isinstance(episode, dict) else episode
        if not steps:
            continue
        for agent_idx in (0, 1):
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

                if max_count == 1:
                    if len(action) == 1:
                        label = action[0]
                        if label >= n_opts or label >= MAX_OPTIONS:
                            continue
                        yield obs, label, frozenset()
                    elif len(action) == 0 and min_count == 0:
                        if n_opts >= MAX_OPTIONS:
                            continue  # no room for the synthetic DECLINE slot
                        yield obs, n_opts, frozenset()
                    # else: anomalous length for a maxCount==1 response -- drop
                    continue

                if max_count is not None and max_count > 1:
                    if any(a >= n_opts or a >= MAX_OPTIONS for a in action):
                        continue
                    excluded: set[int] = set()
                    for picked in action:
                        yield obs, picked, frozenset(excluded)
                        excluded.add(picked)
                    if len(action) < max_count and min_count <= len(action) and n_opts < MAX_OPTIONS:
                        yield obs, n_opts, frozenset(excluded)


class ILDataset(IterableDataset):
    """Streams (features, label) examples from a split folder.

    A small in-memory shuffle buffer decorrelates consecutive examples
    (which otherwise come from the same episode / same match) without
    loading the whole split into memory -- 4,554 + 4,430 episodes is large.
    Use ``shuffle_buffer=1`` for the eval split (deterministic order).
    """

    def __init__(
        self,
        split_dir: Path,
        max_episodes: int | None = None,
        shuffle_buffer: int = 2000,
        seed: int = 0,
    ) -> None:
        self.split_dir = split_dir
        self.max_episodes = max_episodes
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buf: list[dict] = []
        for obs, label, exclude in iter_decisions(self.split_dir, self.max_episodes):
            feats = encode_observation(obs, exclude=exclude)
            if feats is None:
                continue
            feats.pop("n_real_options", None)
            feats["label"] = torch.tensor(label, dtype=torch.long)
            if self.shuffle_buffer <= 1:
                yield feats
                continue
            buf.append(feats)
            if len(buf) >= self.shuffle_buffer:
                rng.shuffle(buf)
                yield from buf
                buf = []
        if buf:
            rng.shuffle(buf)
            yield from buf
