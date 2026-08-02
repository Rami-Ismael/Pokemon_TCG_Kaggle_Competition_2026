"""Mega Lucario ex RL+MCTS runtime (fresh build from official sample).

Built by scripts/bootstrap_lucario_mcts_runtime.py — do not hand-edit the
mechanical sample block; re-run bootstrap after sample updates, then re-apply
patches in PATCH markers below.

Fixes vs Kaggle sample:
  - Real opponent deck in search_begin (not Snorlax 1072)
  - d128 training defaults; draw terminal value 0
  - Champion-gated eval helpers for field training
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn
import torch.nn.functional
import torch.optim


def _bootstrap_cg_path() -> str:
    """Return directory added to sys.path (parent of the cg package)."""
    root = Path(__file__).resolve().parents[1]
    env = os.environ.get("CG_LIB", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        root / "cg-lib",
        root / "data" / "sim" / "sample_submission",
    ])
    for hit in glob.glob("/kaggle/input/**/cg-lib", recursive=True):
        candidates.append(Path(hit))
    for p in candidates:
        if (p / "cg" / "game.py").exists():
            sys.path.insert(0, str(p))
            return str(p)
        if p.name == "cg" and (p / "game.py").exists():
            sys.path.insert(0, str(p.parent))
            return str(p.parent)
    raise FileNotFoundError(
        "cg engine not found. Set CG_LIB or run scripts/fetch_sim_engine.py "
        "(Windows cg.dll) or place cg-lib/ with cg/ next to this repo."
    )


_BOOT = _bootstrap_cg_path()


# --- training profile (d128 basic; override via env LUC_*) --------------------
def _ei(name, default):
    return int(os.environ.get(name, default))


def _ef(name, default):
    return float(os.environ.get(name, default))


SEED = _ei("LUC_SEED", 42)
SEARCH_COUNT = _ei("LUC_SEARCH_COUNT", 12)
BATCH_SIZE = _ei("LUC_BATCH", 128)
LR = _ef("LUC_LR", 3e-4)
GRAD_CLIP = _ef("LUC_GRAD_CLIP", 1.0)
GATE_GAMES = _ei("LUC_GATE_GAMES", 20)
GATE_WINRATE = _ef("LUC_GATE_WINRATE", 0.55)
SELFPLAY_GAMES = _ei("LUC_SELFPLAY_GAMES", 40)
EVAL_GAMES = _ei("LUC_EVAL_GAMES", 20)
REPLAY_ITERS = _ei("LUC_REPLAY_ITERS", 2)
TEMP_PLIES = _ei("LUC_TEMP_PLIES", 8)
VALUE_LAMBDA = _ef("LUC_VALUE_LAMBDA", 0.9)
DIRICHLET_ALPHA = _ef("LUC_DIRICHLET_ALPHA", 0.03)
DIRICHLET_EPS = _ef("LUC_DIRICHLET_EPS", 0.25)
TIME_BUDGET = _ef("LUC_TIME_BUDGET_SEC", 6.0 * 3600)
# Kaggle: 10 min/player cumulative decision time; forfeit on overrun. Train with 9:59 buffer.
PLAYER_CLOCK_LIMIT_SEC = _ef("LUC_PLAYER_CLOCK_LIMIT_SEC", 599.0)
_PLAYER_CLOCK_ENABLED = bool(_ei("LUC_PLAYER_CLOCK", 1))

# #region agent log
_DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-093ff2.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        import json
        payload = {
            "sessionId": "093ff2",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _obs_debug_summary(obs) -> dict:
    sel = obs.select
    opts = []
    for i, o in enumerate(sel.option[:8]):
        opts.append({
            "i": i,
            "type": getattr(o, "type", None),
            "area": getattr(o, "area", None),
            "index": getattr(o, "index", None),
            "playerIndex": getattr(o, "playerIndex", None),
        })
    return {
        "context": str(getattr(sel, "context", "")),
        "n_opts": len(sel.option),
        "min": int(getattr(sel, "minCount", 0) or 0),
        "max": int(getattr(sel, "maxCount", 0) or 0),
        "turn": int(getattr(obs.current, "turn", -1)),
        "yourIndex": int(getattr(obs.current, "yourIndex", -1)),
        "result": int(getattr(obs.current, "result", -1)),
        "has_select_deck": getattr(sel, "deck", None) is not None,
        "opts": opts,
        "opp_name": _TRAIN_OPP_NAME,
    }
# #endregion


def set_field_training_flags(
    *,
    player_clock: bool = True,
    deck_scope: bool = True,
    clock_limit_sec: float | None = None,
) -> None:
    """Configure 9:59 forfeit cliff and deck-scoped lever soft masking for field train."""
    global PLAYER_CLOCK_ENABLED, _DECK_SCOPE_ENABLED, PLAYER_CLOCK_LIMIT_SEC
    PLAYER_CLOCK_ENABLED = player_clock
    _DECK_SCOPE_ENABLED = deck_scope
    if clock_limit_sec is not None:
        PLAYER_CLOCK_LIMIT_SEC = clock_limit_sec


# Back-compat name used by PlayerClock default
PLAYER_CLOCK_ENABLED = _PLAYER_CLOCK_ENABLED

D_MODEL = _ei("LUC_D_MODEL", 128)
NUM_HEADS = _ei("LUC_HEADS", 2)
D_FF = _ei("LUC_D_FF", 256)
ENC_LAYERS = _ei("LUC_ENC_LAYERS", 1)
DEC_LAYERS = _ei("LUC_DEC_LAYERS", 1)

WORK = os.environ.get("LUC_WORK", "rl_mcts_field/lucarioex_v1")

import glob
import math
import os
import random
import sys

import torch
import torch.nn
import torch.nn.functional
import torch.optim


from cg.api import (
    AreaType,
    Card,
    CardType,
    Observation,
    OptionType,
    PlayerState,
    Pokemon,
    SearchState,
    SelectContext,
    all_attack,
    all_card_data,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation_class,
)
from cg.game import battle_start, battle_finish, battle_select

# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
card_count = max(all_card, key=lambda c: c.cardId).cardId + 1 # Max Card ID + 1

attack_count = max(all_attack(), key=lambda a: a.attackId).attackId + 1 # Max Attack ID + 1

num_words_encoder = 24
encoder_size = 22000 # Encoder input size exceeding the vocabulary size

decoder_main_feature = 8 # Feature count of SelectContext.Main
decoder_attack_offset = 14 # First index of Attack feature
decoder_card_offset = decoder_attack_offset + attack_count # First index of Card Feature
decoder_size = decoder_card_offset + (1 + decoder_main_feature + SelectContext.RECOVER_SPECIAL_CONDITION) * card_count # Decoder input vocabulary size


# Decoder Layer of MyModel
class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int):
        super(DecoderLayer, self).__init__()

        self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = torch.nn.Linear(d_model, d_feedforward)
        self.fc2 = torch.nn.Linear(d_feedforward, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = self.fc1(res)
        y = torch.nn.functional.relu(y)
        y = self.fc2(y)
        return self.norm2(res + y)

# My Transformer Model
class MyModel(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_feedforward: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int
    ):
        super(MyModel, self).__init__()

        self.d_model = d_model

        self.encoder_bag = torch.nn.EmbeddingBag(encoder_size, d_model, mode="sum")
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers_encoder, enable_nested_tensor=False)
        self.encoder_fc = torch.nn.Linear(d_model, 1)
        self.decoder_bag = torch.nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = torch.nn.ModuleList()
        for _ in range(num_layers_decoder):
            self.decoder.append(DecoderLayer(d_model, num_heads, d_feedforward))
        self.decoder_fc = torch.nn.Linear(d_model, 1)

    def forward(self,
                index_encoder: torch.Tensor,
                value_encoder: torch.Tensor,
                offset_encoder: torch.Tensor,
                index_decoder: torch.Tensor,
                value_decoder: torch.Tensor,
                offset_decoder: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, num_words_encoder, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        v = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        p = torch.tanh(p)
        return (v, p)

# torch.nn.EmbeddingBag input
class SparseVector:
    index: list[int]
    value: list[float]
    offset: list[int]
    pos: int

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index: int, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos: int):
        self.pos += pos

    def add_single(self, value: float | int | bool):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))

# Add encoder card feature
def add_card(sv: SparseVector, card: Card | Pokemon | None):
    if card != None:
        sv.add(card.id, 1)
    sv.add_pos(card_count)

# Add encoder cards feature
def add_cards(sv: SparseVector, cards: list[Card] | None, value: float):
    if cards != None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(card_count)

# Add encoder Pokémon feature
def add_pokemon(sv: SparseVector, poke: Pokemon | None):
    if poke == None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * card_count)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        add_card(sv, poke)
        add_cards(sv, poke.tools, 1.0)
        add_cards(sv, poke.energyCards, 0.5)
        
# Add encoder player feature
def add_player(sv: SparseVector, ps: PlayerState):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)

    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)

    add_cards(sv, ps.discard, 0.25)

def get_encoder_input(obs: Observation, your_deck: list[int]) -> SparseVector:
    your_index = obs.current.yourIndex
    state = obs.current

    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8): # For bench
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                add_pokemon(sv, ps.bench[j])
            else:
                add_pokemon(sv, None)
            if j != 7:  # Not last
                sv.pos = pos  # Return to the previous position
    
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            add_pokemon(sv, ps.active[0])
        else:
            add_pokemon(sv, None)

    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        add_player(sv, ps)
        
    sv.word_start()
    add_cards(sv, state.players[your_index].hand, 0.25)
        
    sv.word_start()
    for id in your_deck:
        sv.add(id, 0.25)
    sv.add_pos(card_count)
        
    sv.word_start()
    add_cards(sv, state.stadium, 1.0)

    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None

# Add decoder Main Select feature
def decoder_main(sv: SparseVector, feature_index: int, card: Card | Pokemon | None):
    if card != None:
        sv.add(decoder_card_offset + feature_index * card_count + card.id, 1)
        
# Add decoder Card ID feature
def decoder_card_id(sv: SparseVector, context: SelectContext, card_id: int):
    sv.add(decoder_card_offset + (decoder_main_feature + context) * card_count + card_id, 1)

# Add decoder Card feature
def decoder_card(sv: SparseVector, context: SelectContext, card: Card | Pokemon | None):
    if card != None:
        decoder_card_id(sv, context, card.id)

def get_decoder_input(obs: Observation, actions: list[list[int]]) -> SparseVector:
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        
        if len(action) == 0:
            sv.add(0, 1)
            continue
        
        for i in action:
            o = obs.select.option[i]
            match o.type:
                case OptionType.END:
                    sv.add(1, 1)
                case OptionType.YES:
                    sv.add(2, 1)
                case OptionType.NO:
                    sv.add(3, 1)
                case OptionType.SPECIAL_CONDITION:
                    sv.add(4 + o.specialConditionType, 1)
                case OptionType.NUMBER:
                    sv.add(9 + min(o.number, 4), 1)
                case OptionType.ATTACK:
                    sv.add(decoder_attack_offset + o.attackId, 1)
                case OptionType.PLAY:
                    decoder_main(sv, 0, ps.hand[o.index])
                case OptionType.ATTACH:
                    decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 2, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.EVOLVE:
                    decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index))
                    decoder_main(sv, 4, get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
                case OptionType.ABILITY:
                    decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index))
                case OptionType.DISCARD:
                    decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index))
                case OptionType.RETREAT:
                    decoder_main(sv, 7, ps.active[0])
                case OptionType.CARD:
                    decoder_card(sv, context, get_card(obs, o.area, o.index, o.playerIndex))
                case OptionType.TOOL_CARD:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.tools[o.toolIndex])
                case OptionType.ENERGY_CARD | OptionType.ENERGY:
                    card = get_card(obs, o.area, o.index, o.playerIndex)
                    decoder_card(sv, context, card.energyCards[o.energyIndex])
                case OptionType.SKILL:
                    decoder_card_id(sv, context, o.cardId)

    return sv

# Evaluate with My Model
def eval_nn(sv_enc: SparseVector, sv_dec:SparseVector, model: MyModel) -> tuple[float, list[float]]:
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device))

    return (value.tolist()[0][0], policy.tolist()[0])

# Single Training Sample
class LearnSample:
    def __init__(self, value: float, policy: list[float], sv_enc: SparseVector, sv_dec:SparseVector):
        self.value = value # Encoder output
        self.policy = policy # Decoder output
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
   
# MCTS Node Child
class Child:
    node: 'Node | None'
    select: list[int] # Selected option indices
    prob: float # Probability

    def __init__(self, select: list[int], prob: float):
        self.node = None
        self.select = select
        self.prob = prob
        self.invalid = False

# MCTS Node
class Node:
    value: float # Self value
    total: float # Total value
    visit: int # Visit count
    parent: 'Node | None' # Parent node
    children: list[Child]
    state: SearchState # Search State of this node

    def __init__(self, parent: 'Node | None', state: SearchState):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    # Backpropagation value
    def backprop(self, value: float):
        self.total += value
        self.visit += 1
        if self.parent != None:
            self.parent.backprop(value)


def _enumerate_action_combos(obs) -> list[list[int]]:
    """Up to 64 combinatorial option-index tuples (may include illegal combos)."""
    actions: list[list[int]] = []
    indices = list(range(obs.select.maxCount))
    for _ in range(64):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < len(obs.select.option) - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


def _fallback_actions(obs) -> list[list[int]]:
    """Minimal fallback when combinatorial enumeration yields no legal combos."""
    n_opts = len(obs.select.option)
    mc = int(obs.select.maxCount)
    if mc <= 1:
        return [[i] for i in range(n_opts)]
    if mc <= n_opts:
        return [list(range(mc))]
    return [[0]]


def _filter_legal_actions(
    search_id: int,
    actions: list[list[int]],
    *,
    obs_summary: dict | None = None,
) -> list[list[int]]:
    """Keep only selections the search engine accepts from this search state."""
    legal: list[list[int]] = []
    failures: list[dict] = []
    for select in actions:
        try:
            child_state = search_step(search_id, select)
            search_release(child_state.searchId)
            legal.append(select)
        except (ValueError, RuntimeError) as exc:
            failures.append({
                "select": select,
                "err": type(exc).__name__,
                "msg": str(exc),
            })
            continue
    if not legal and failures:
        _debug_log(
            "H1",
            "lucario_mcts_runtime.py:_filter_legal_actions",
            "all search_step probes failed",
            {"failures": failures, "n_actions": len(actions), "obs": obs_summary or {}},
        )
    return legal


def _selections_equal(a: list[int], b: list[int]) -> bool:
    """Compare option picks; order-insensitive when multi-select."""
    return sorted(a) == sorted(b)


def _apply_dirichlet_noise(children: list[Child]) -> None:
    """Root exploration noise (AlphaZero-style prior mix)."""
    if not children:
        return
    noise = [random.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in children]
    total = sum(noise) or 1.0
    for child, n in zip(children, noise):
        child.prob = (1.0 - DIRICHLET_EPS) * child.prob + DIRICHLET_EPS * (n / total)


def _dead_expand_node(
    parent: Node,
    your_index: int,
) -> Node:
    """Leaf for illegal search_step — bad for the player to move."""
    dead = Node(parent, parent.state)
    yi = parent.state.observation.current.yourIndex
    dead.value = -1.0 if yi == your_index else 1.0
    dead.visit = 1
    dead.total = dead.value
    dead.backprop(dead.value)
    return dead


def create_node(parent: Node | None,
                search_state: SearchState,
                your_index: int,
                your_deck: list[int],
                model: MyModel
    ) -> tuple[Node, LearnSample | None]:
    node = Node(parent, search_state)

    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        # Battle finished
        if state.result == 2:
            node.value = 0
        elif state.result == your_index:
            node.value = 1
        else:
            node.value = -1
        node.backprop(node.value)
        sample = None
    else:
        obs_summary = _obs_debug_summary(obs)
        combos = _enumerate_action_combos(obs)
        actions = _filter_legal_actions(
            search_state.searchId, combos, obs_summary=obs_summary,
        )
        if not actions:
            actions = _filter_legal_actions(
                search_state.searchId, _fallback_actions(obs), obs_summary=obs_summary,
            )
        if not actions:
            raw = _fallback_actions(obs)
            if raw:
                _debug_log(
                    "H2",
                    "lucario_mcts_runtime.py:create_node",
                    "unverified fallback actions (search_step probes failed)",
                    {
                        "obs": obs_summary,
                        "parent_is_root": parent is None,
                        "combos": combos,
                        "fallback": raw,
                    },
                )
                actions = raw
        if not actions:
            _debug_log(
                "H2",
                "lucario_mcts_runtime.py:create_node",
                "no legal MCTS actions",
                {
                    "obs": obs_summary,
                    "parent_is_root": parent is None,
                    "combos": combos,
                    "fallback": _fallback_actions(obs),
                },
            )
            raise RuntimeError(
                f"no legal MCTS actions (options={len(obs.select.option)} "
                f"min={obs.select.minCount} max={obs.select.maxCount})"
            )

        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, model)
        v = value
        if state.yourIndex != your_index:
            v = -v
        node.value = v
        node.backprop(v)

        sum = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(Child(actions[i], p))
            sum += p
        for c in node.children:
            c.prob /= sum
        sample = LearnSample(value, policy, sv_enc, sv_dec)

    return (node, sample)



# Helper class to construct batch inputs for the neural network.
class LearnInput:
    index: list[int]
    value: list[float]
    offset: list[int]

    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv: SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)

# Opponent for evaluation.
def random_agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    return random.sample(list(range(len(obs.select.option))), obs.select.maxCount) # Select at random.

# For displaying progress.
def progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // count
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if(current >= count):
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1

# === PATCH: field-aware MCTS (Fix #3) ========================================

# Matchup levers (LucarioScorer) bias root action when training vs real opponents.
_LUCARIO_SCORER = None
_LEVER_BLEND = 0.0
_TRAIN_OPP_NAME = ""
_TRAIN_OPP_DECK: list[int] | None = None
_DECK_SCOPE_ENABLED = True


def set_lucario_lever_teaching(deck_path: str, blend: float = 0.35) -> None:
    """Wire agent/matchup_levers.py into MCTS root selection (our side only)."""
    global _LUCARIO_SCORER, _LEVER_BLEND
    from agent.lucario_policy import LucarioScorer

    _LUCARIO_SCORER = LucarioScorer(deck_path=deck_path)
    _LEVER_BLEND = max(0.0, min(1.0, float(blend)))


def set_training_opponent_context(
    opp_name: str,
    opp_deck_ids: list[int] | None,
    *,
    deck_scope_enabled: bool = True,
) -> None:
    """Per-matchup context for deck-scoped soft masking during field training."""
    global _TRAIN_OPP_NAME, _TRAIN_OPP_DECK, _DECK_SCOPE_ENABLED
    _TRAIN_OPP_NAME = opp_name or ""
    _TRAIN_OPP_DECK = list(opp_deck_ids) if opp_deck_ids else None
    _DECK_SCOPE_ENABLED = deck_scope_enabled


def clear_training_opponent_context() -> None:
    set_training_opponent_context("", None, deck_scope_enabled=True)


def _scorer_root_pick(obs_dict: dict) -> list[int] | None:
    if _LUCARIO_SCORER is None or _LEVER_BLEND <= 0:
        return None
    try:
        sel = obs_dict.get("select") or {}
        options = sel.get("option") or []
        if not options:
            return None
        return _LUCARIO_SCORER.choose(
            obs_dict, sel, obs_dict.get("current"), options,
        )
    except Exception:
        return None


def _pick_root_child(
    visited: list,
    default_child,
    obs_dict: dict,
    *,
    temperature: float,
) -> object:
    """MCTS visits + optional LucarioScorer/lever preference at root."""
    if not visited:
        return default_child
    scorer_pick = _scorer_root_pick(obs_dict)
    if scorer_pick is None:
        return default_child
    effective_blend = _LEVER_BLEND
    if _DECK_SCOPE_ENABLED and _TRAIN_OPP_DECK is not None:
        from agent.deck_scope import soft_lever_blend, visible_board_ids

        board = visible_board_ids(obs_dict)
        effective_blend = soft_lever_blend(
            _LEVER_BLEND,
            opp_name=_TRAIN_OPP_NAME,
            opp_deck_ids=_TRAIN_OPP_DECK,
            board_ids=board,
        )
    if effective_blend <= 0:
        return default_child
    lever_bonus = effective_blend * 1000.0
    best_child, best_score = default_child, -1.0
    for child, visits in visited:
        score = float(visits)
        if _selections_equal(child.select, scorer_pick):
            score += lever_bonus
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def _sample_hidden(deck: list[int], n: int) -> list[int]:
    if n <= 0:
        return []
    if n <= len(deck):
        return random.sample(deck, n)
    return (deck * (n // len(deck) + 1))[:n]


def _deck_basic_ids(deck: list[int]) -> list[int]:
    basics: list[int] = []
    for cid in deck:
        data = card_table.get(cid)
        if (
            data is not None
            and data.cardType == CardType.POKEMON
            and getattr(data, "basic", False)
        ):
            basics.append(cid)
    return basics


def _sample_hidden_deck_for_search(deck: list[int], n: int) -> list[int]:
    """search_begin deck belief: cg requires >=1 Basic during setup."""
    if n <= 0:
        return []
    basics = _deck_basic_ids(deck)
    if not basics:
        return _sample_hidden(deck, n)
    if n == 1:
        return [random.choice(basics)]
    sample = _sample_hidden(deck, n)
    if any(cid in basics for cid in sample):
        return sample
    sample[random.randrange(n)] = random.choice(basics)
    return sample


def _stub_pokemon_id(deck: list[int]) -> int:
    for cid in _deck_basic_ids(deck):
        return cid
    for cid in deck:
        data = card_table.get(cid)
        if data is not None and data.cardType == CardType.POKEMON:
            return cid
    return deck[0] if deck else 677


def _stub_energy_id(deck: list[int]) -> int:
    for cid in deck:
        data = card_table.get(cid)
        if data is not None and data.cardType in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            return cid
    return 6


def mcts_agent(
    obs_dict,
    your_deck: list[int],
    model,
    *,
    opp_deck: list[int] | None = None,
    add_noise: bool = False,
    temperature: float = 0.0,
):
    """MCTS with simulator-valid children; opponent belief from real deck list."""
    opp_deck = opp_deck or your_deck
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    state = obs.current
    opp_ps = state.players[1 - your_index]
    active = opp_ps.active

    search_state = search_begin(
        obs,
        your_deck=_sample_hidden_deck_for_search(
            your_deck, state.players[your_index].deckCount,
        ),
        your_prize=_sample_hidden(your_deck, len(state.players[your_index].prize)),
        opponent_deck=_sample_hidden_deck_for_search(opp_deck, opp_ps.deckCount),
        opponent_prize=_sample_hidden(opp_deck, len(opp_ps.prize)),
        opponent_hand=[_stub_energy_id(opp_deck)] * opp_ps.handCount,
        opponent_active=[_stub_pokemon_id(opp_deck)] if len(active) > 0 and active[0] is None else [],
    )
    root, sample = create_node(None, search_state, your_index, your_deck, model)
    if add_noise and root.children:
        _apply_dirichlet_noise(root.children)

    for _ in range(SEARCH_COUNT):
        current = root
        nxt = None
        while True:
            value = -1e9
            c = 0.4 * math.sqrt(current.visit)
            for child in current.children:
                if child.invalid:
                    continue
                visit = 0
                if child.node is None:
                    v = current.total / current.visit
                else:
                    v = child.node.total / child.node.visit
                    visit = child.node.visit
                if current.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if value < v:
                    value = v
                    nxt = child
            if nxt is None:
                break
            if nxt.node is None:
                try:
                    ss = search_step(current.state.searchId, nxt.select)
                    nxt.node, _ = create_node(current, ss, your_index, your_deck, model)
                except (ValueError, RuntimeError):
                    nxt.invalid = True
                    nxt.node = _dead_expand_node(current, your_index)
                break
            current = nxt.node
            if current.state.observation.current.result >= 0:
                current.backprop(current.value)
                break

    visited = [
        (c, c.node.visit) for c in root.children
        if c.node is not None and not c.invalid
    ]
    lever_override = False
    if visited:
        if temperature > 0.0 and len(visited) > 1:
            weights = [v ** (1.0 / temperature) for _, v in visited]
            tot = sum(weights) or 1.0
            r = random.random() * tot
            acc, default_child = 0.0, visited[-1][0]
            for (child, _), w in zip(visited, weights):
                acc += w
                if r <= acc:
                    default_child = child
                    break
        else:
            default_child = max(visited, key=lambda t: t[1])[0]
        max_child = default_child
        if obs.current.yourIndex == your_index and _LEVER_BLEND > 0:
            lever_child = _pick_root_child(
                visited, default_child, obs_dict, temperature=temperature,
            )
            lever_override = not _selections_equal(lever_child.select, default_child.select)
            max_child = lever_child
    else:
        max_child = root.children[0] if root.children else Child(
            random.sample(list(range(len(obs.select.option))), obs.select.maxCount), 1.0
        )

    min_value = 10.0
    for child in root.children:
        if child.node is not None and not child.invalid:
            v = child.node.total / child.node.visit
            if min_value > v:
                min_value = v
    if sample is not None:
        sample.value = root.total / max(1, root.visit)
        if lever_override:
            for i, child in enumerate(root.children):
                sample.policy[i] = (
                    1.0 if _selections_equal(child.select, max_child.select) else -1.0
                )
        else:
            for i, child in enumerate(root.children):
                base = sample.value
                if child.node is None or child.invalid:
                    v = min_value - base - 0.03
                else:
                    v = child.node.total / child.node.visit - base
                sample.policy[i] = max(-1.0, min(1.0, v))

    search_end()
    return max_child.select, sample


def load_lucario_deck(path: str | Path | None = None) -> list[int]:
    p = Path(path) if path else Path(__file__).resolve().parents[1] / "agent_decks" / "real_mega_lucario_ex.csv"
    ids = [int(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(ids) != 60:
        raise ValueError(f"{p} has {len(ids)} cards, expected 60")
    return ids


LUCARIO_DECK = load_lucario_deck()


class PlayerClock:
    """Per-player cumulative think-time (wall clock). Past limit => forfeit loss."""

    __slots__ = ("limit_sec", "used", "enabled")

    def __init__(self, limit_sec: float | None = None, *, enabled: bool | None = None):
        self.limit_sec = PLAYER_CLOCK_LIMIT_SEC if limit_sec is None else limit_sec
        self.enabled = PLAYER_CLOCK_ENABLED if enabled is None else enabled
        self.used = [0.0, 0.0]

    def charge(self, player: int, seconds: float) -> int | None:
        """Add think time for player; return winner index if they forfeit, else None."""
        if not self.enabled or self.limit_sec <= 0:
            return None
        self.used[player] += max(0.0, seconds)
        if self.used[player] > self.limit_sec:
            return 1 - player
        return None


def timed_act(
    act_fn,
    obs: dict,
    clock: PlayerClock,
    player: int,
):
    """Run act_fn(obs), charge player clock, return (selection, forfeit_winner|None)."""
    t0 = time.perf_counter()
    selection = act_fn(obs)
    forfeit = clock.charge(player, time.perf_counter() - t0)
    return selection, forfeit


def timed_mcts(
    act_fn,
    obs: dict,
    clock: PlayerClock,
    player: int,
    *args,
    **kwargs,
):
    """Like timed_act but for (selected, sample) MCTS returns."""
    t0 = time.perf_counter()
    selected, sample = act_fn(obs, *args, **kwargs)
    forfeit = clock.charge(player, time.perf_counter() - t0)
    return selected, sample, forfeit


def label_samples(samples, terminal_result: int, your_index: int, sink: list) -> None:
    """terminal_result: winner index, or 2 for draw (simulator simultaneous KO)."""
    if terminal_result == 2:
        value = 0.0
    elif terminal_result == your_index:
        value = 1.0
    else:
        value = -1.0
    for sample in reversed(samples):
        if sample is None:
            continue
        sample.value = (value + sample.value) * 0.5
        value = value * VALUE_LAMBDA + sample.value * (1.0 - VALUE_LAMBDA)
        sink.append(sample)


def train_on_samples(model, optimizer, scheduler, scaler, device, loss_fn_enc, loss_fn_dec, sample_list):
    if not sample_list:
        return 0.0
    model.train()
    random.shuffle(sample_list)
    n = len(sample_list)
    total_loss = 0.0
    steps = 0
    use_amp = device.type == "cuda"
    offset = 0
    while offset < n:
        batch_size = min(BATCH_SIZE, n - offset)
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask, label_enc, label_dec = [], [], []
        for j in range(offset, offset + batch_size):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(64 - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        mask_t = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        le_t = torch.tensor(label_enc, dtype=torch.float32, device=device).view(batch_size, -1)
        ld_t = torch.tensor(label_dec, dtype=torch.float32, device=device).view(batch_size, -1)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out_enc, out_dec = model(
                torch.tensor(input_enc.index, dtype=torch.int32, device=device),
                torch.tensor(input_enc.value, dtype=torch.float32, device=device),
                torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
                torch.tensor(input_dec.index, dtype=torch.int32, device=device),
                torch.tensor(input_dec.value, dtype=torch.float32, device=device),
                torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
            )
            loss_enc = loss_fn_enc(out_enc, le_t)
            loss_dec = (loss_fn_dec(out_dec, ld_t) * mask_t).sum() / float(batch_size)
            loss = loss_enc + loss_dec

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach())
        steps += 1
        offset += batch_size
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(1, steps)


def selfplay_game(model, deck: list[int], clock: PlayerClock | None = None) -> list:
    out: list = []
    clock = clock or PlayerClock()
    obs, start = battle_start(deck, deck)
    if start.errorPlayer >= 0:
        raise ValueError(f"deck error type={start.errorType}")
    samples: list[list] = [[], []]
    ply = 0
    forfeit_result = None
    while obs["current"]["result"] < 0 and forfeit_result is None:
        yi = obs["current"]["yourIndex"]
        temp = 1.0 if ply < TEMP_PLIES else 0.0
        selected, sample, forfeit_result = timed_mcts(
            mcts_agent,
            obs,
            clock,
            yi,
            deck,
            model,
            opp_deck=deck,
            add_noise=True,
            temperature=temp,
        )
        samples[yi].append(sample)
        if forfeit_result is None:
            obs = battle_select(selected)
        ply += 1
    battle_finish()
    result = forfeit_result if forfeit_result is not None else obs["current"]["result"]
    for i in range(2):
        label_samples(samples[i], result, i, out)
    return out


def eval_vs_random(model, deck: list[int], games: int) -> float:
    wins = 0
    model.eval()
    with torch.inference_mode():
        for i in range(games):
            your_index = i % 2
            decks = (deck, deck) if your_index == 0 else (deck, deck)
            obs, start = battle_start(*decks)
            if start.errorPlayer >= 0:
                raise ValueError(f"deck error type={start.errorType}")
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, deck, model, opp_deck=deck)
                else:
                    selected = random_agent(obs)
                obs = battle_select(selected)
            battle_finish()
            if obs["current"]["result"] == your_index:
                wins += 1
    denom = max(1, games)
    return wins / denom


def eval_vs_model(candidate, champion, deck: list[int], games: int) -> float:
    wins = 0
    candidate.eval()
    champion.eval()
    with torch.inference_mode():
        for i in range(games):
            your_index = i % 2
            obs, start = battle_start(deck, deck)
            if start.errorPlayer >= 0:
                raise ValueError(f"deck error type={start.errorType}")
            while obs["current"]["result"] < 0:
                if obs["current"]["yourIndex"] == your_index:
                    selected, _ = mcts_agent(obs, deck, candidate, opp_deck=deck)
                else:
                    selected, _ = mcts_agent(obs, deck, champion, opp_deck=deck)
                obs = battle_select(selected)
            battle_finish()
            if obs["current"]["result"] == your_index:
                wins += 1
    return wins / max(1, games)


if __name__ == "__main__":
    print("lucario_mcts_runtime loaded; use scripts/train_lucario_field_mcts.py to train.")
