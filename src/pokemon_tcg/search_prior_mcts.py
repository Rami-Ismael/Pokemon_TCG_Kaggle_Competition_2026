"""IL-prior MCTS: the engine's official Search API guided by OUR trained
imitation policy (Phase 3 of the kiyotah-pipeline port).

piKL-shaped (Jacob et al., arXiv:2112.07544): search anchored to a strong
human-cloned prior. The scaffold (determinized `search_begin`, PUCT-like
selection with c = 0.4*sqrt(N), most-visited final move, <=64 combination
children) is kept from the notebook port in `search_api.py` for
comparability; what changes is the evaluator:

- child priors: softmax over PTCGImitationPolicy option logits (mean of
  member logits for multi-select combos; the DECLINE logit for the legal
  empty selection, a child the notebook scheme lacks entirely);
- leaf values: the S2-E4-style offline critic (PTCGImitationPolicy trunk +
  scalar head, outcome scale [0,1]) mapped to [-1,1] from the searching
  player's perspective.

Determinization legality: identical to search_api.mcts_agent — own deck
sampled from our known 60, opponent unknowns filled with placeholders,
only public counts read (Phase-2c report).

Per-node cost is two 3.3M-param CPU forwards (~6 ms single-thread), so
SEARCH_COUNT=30 is ~0.2 s/decision — see reports/search_api_phase2.md for
the budget arithmetic. Inference here must stay CPU-single-thread safe.
"""

from __future__ import annotations

import dataclasses
import math
import random

import torch

from .il_dataset import encode_observation
from .search_api import (
    MAX_ACTION_COMBOS,  # noqa: F401  (re-exported context for callers)
    PLACEHOLDER_CARD,
    PLACEHOLDER_POKEMON,
    Node,
    Child,
    enumerate_actions,
    search_begin,
    search_end,
    search_step,
    to_observation_class,
)


def _obs_to_dict(obs) -> dict:
    """SearchState.observation (cg dataclass) -> the raw-dict form
    encode_observation expects. Root observations are already dicts."""
    if isinstance(obs, dict):
        return obs
    return dataclasses.asdict(obs)


class ILPriorEvaluator:
    """(value, child_priors) for a search node, from the IL policy + critic.

    `policy` is a PTCGImitationPolicy (option scoring, Pattern B).
    `critic` is an OfflineValueModel (same trunk family, scalar outcome head)
    or None — with None, non-terminal leaves evaluate to 0.0 (prior-only
    search; used as an ablation arm, not the main configuration).
    """

    def __init__(self, policy, critic, device: str = "cpu"):
        self.policy = policy
        self.critic = critic
        self.device = device

    @torch.inference_mode()
    def evaluate(self, obs_dict: dict, actions: list[list[int]]) -> tuple[float, list[float]]:
        """Return (to-act-player value in [-1,1], prior prob per action).

        `actions` are option-index combinations (the node's children).
        Falls back to (0, uniform) if the observation cannot be encoded —
        never raises into the search.
        """
        n_actions = len(actions)
        uniform = [1.0 / n_actions] * n_actions if n_actions else []
        feats = encode_observation(obs_dict)
        if feats is None:
            return 0.0, uniform
        n_real = feats.pop("n_real_options")
        batch = {k: v.unsqueeze(0).to(self.device) for k, v in feats.items()}

        logits = self.policy(**batch)["logits"][0]
        if bool(torch.isnan(logits[: n_real + 1]).any()):
            return 0.0, uniform
        # DECLINE slot exists at index n_real whenever minCount==0 AND the
        # option list didn't fill MAX_OPTIONS (>=48 options leave no room for
        # it -- reading logits[n_real] there is out of bounds, the 12.76%
        # IndexError-fallback bug of the first critic benchmark). When the
        # slot is unavailable, give the empty selection a NEUTRAL prior (mean
        # of real-option logits), not an extreme one.
        real = logits[:n_real]
        if n_real < logits.shape[0]:
            decline_logit = float(logits[n_real])
        else:
            decline_logit = float(real.mean()) if n_real else 0.0

        child_logits = []
        for action in actions:
            if len(action) == 0:
                child_logits.append(decline_logit)
            else:
                valid = [float(logits[i]) for i in action if i < n_real]
                child_logits.append(sum(valid) / len(valid) if valid else -30.0)
        t = torch.tensor(child_logits, dtype=torch.float32)
        priors = torch.softmax(t, dim=0).tolist()

        value = 0.0
        if self.critic is not None:
            v01 = float(self.critic(**batch).view(-1)[0])  # outcome scale [0,1]
            value = max(-1.0, min(1.0, 2.0 * v01 - 1.0))
        return value, priors


def _node_children(obs, evaluator: ILPriorEvaluator, your_index: int) -> tuple[float, list[Child]]:
    """Build children + priors for a non-terminal search observation."""
    actions = enumerate_actions(obs)
    # The legal empty selection (minCount==0, maxCount>=1) is a real move the
    # notebook scheme never enumerates (14.5% of corpus decisions allow
    # sub-max selections; the empty one is the case our DECLINE head scores).
    if (obs.select.minCount or 0) == 0 and (obs.select.maxCount or 0) >= 1:
        actions = actions + [[]]
    value, priors = evaluator.evaluate(_obs_to_dict(obs), actions)
    state = obs.current
    v = value if state.yourIndex == your_index else -value
    children = [Child(a, p) for a, p in zip(actions, priors)]
    return v, children


def _create_node(parent, search_state, your_index, evaluator) -> Node:
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current
    if state.result >= 0:
        if state.result == 2:
            node.value = 0.0
        elif state.result == your_index:
            node.value = 1.0
        else:
            node.value = -1.0
    else:
        node.value, node.children = _node_children(obs, evaluator, your_index)
    node.backprop(node.value)
    return node


def mcts_choose(
    obs_dict: dict,
    my_deck: list[int],
    evaluator: ILPriorEvaluator,
    search_count: int = 30,
    rng: random.Random | None = None,
) -> list[int]:
    """Pick an option-index list for a live env observation via IL-prior MCTS.

    Same tree scaffold as search_api.mcts_agent (selection constant, most-
    visited final choice); no training-sample generation.
    """
    rng = rng or random
    obs = to_observation_class(obs_dict)
    # Forced move: exactly one legal selection and no legal decline — search
    # cannot change the outcome, skip it (a large share of real decisions).
    sel = obs.select
    n_opts = len(sel.option)
    max_count = sel.maxCount if sel.maxCount is not None else 1
    min_count = sel.minCount or 0
    if min_count >= 1 and n_opts == max_count:
        return list(range(n_opts))
    your_index = obs.current.yourIndex
    state = obs.current
    active = state.players[1 - your_index].active
    search_state = search_begin(
        obs,
        your_deck=rng.sample(my_deck, state.players[your_index].deckCount),
        your_prize=rng.sample(my_deck, len(state.players[your_index].prize)),
        opponent_deck=[PLACEHOLDER_POKEMON] * state.players[1 - your_index].deckCount,
        opponent_prize=[PLACEHOLDER_CARD] * len(state.players[1 - your_index].prize),
        opponent_hand=[PLACEHOLDER_CARD] * state.players[1 - your_index].handCount,
        opponent_active=[PLACEHOLDER_POKEMON]
        if len(active) > 0 and active[0] == None  # noqa: E711
        else [],
    )
    try:
        root = _create_node(None, search_state, your_index, evaluator)

        for _ in range(search_count):
            current = root
            while True:
                best_v = -1e9
                best_child = None
                c = 0.4 * math.sqrt(current.visit)
                for child in current.children:
                    visit = 0
                    if child.node is None:
                        v = current.total / current.visit
                    else:
                        v = child.node.total / child.node.visit
                        visit = child.node.visit
                    if current.state.observation.current.yourIndex != your_index:
                        v = -v
                    v += c * child.prob / (1 + visit)
                    if best_v < v:
                        best_v = v
                        best_child = child

                if best_child is None:
                    break  # no children (should not happen on non-terminal nodes)
                if best_child.node is None:
                    next_state = search_step(current.state.searchId, best_child.select)
                    best_child.node = _create_node(current, next_state, your_index, evaluator)
                    break
                current = best_child.node
                if current.state.observation.current.result >= 0:
                    current.backprop(current.value)
                    break

        best = None
        best_visit = -1
        for child in root.children:
            if child.node is not None and child.node.visit > best_visit:
                best = child
                best_visit = child.node.visit
        if best is None:  # search_count == 0 or nothing expanded: prior argmax
            best = max(root.children, key=lambda ch: ch.prob)
        return best.select
    finally:
        search_end()
