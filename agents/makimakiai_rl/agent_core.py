# makimakiai tiny-RL baseline: learned-linear-weights option-scoring policy.
"""makimakiai/ptcg-tiny-rl-to-submission-baseline-guide — shippable agent.

Method (genuinely different from the rest of the pool): a *linear policy over
cabt option types*. Every legal option is turned into a scalar `option_score`
by a weight vector (one weight per OptionType plus a few card-type / board-state
terms); the agent selects the top-k options by that score. No search, no rules
tree, no torch — just a learned-weights ranker. This is the notebook's own
`MAIN_TEMPLATE`, verbatim except for the two adaptations noted below.

WEIGHTS PROVENANCE — read this before "improving" the numbers.
`LEARNED_WEIGHTS` below are the notebook's hand-set `DEFAULT_WEIGHTS`, i.e. the
INITIAL policy the evolutionary search starts from. They are NOT the RL-evolved
`best_weights`, and are deliberately labelled untrained. Why we did not bake the
"trained" weights (verified locally 2026-08-02, see notebooks/reference/INDEX.md):

  * The bake is CHEAP (~3.2 s for the full 4-generation search) but
    NONDETERMINISTIC. The search's fitness is the win/loss of games played by
    the compiled cg engine (cg/sim.py -> libcg via ctypes), whose RNG is not
    seedable from Python. `random.seed(SEED)` in the notebook fixes only the
    mutation proposals and the tiny score tiebreak, NOT the game outcomes.
    Two runs with the identical seed (20260618) produced materially different
    best_weights (attack 3.81 vs 3.48, attach 2.82 vs 1.22). The notebook never
    commits its own best_weights output either, so there is no canonical vector
    to restore.
  * The fitness signal is 4 games/candidate — noise-dominated. The "evolved"
    weights are a single random draw around these defaults, not a stable optimum.

Shipping a single nondeterministic draw dressed up as "the learned weights"
would be fabrication. The method diversity this agent adds lives in the
`option_score` architecture, which is identical whether the weights are the
hand-set defaults or a noisy perturbation of them — so we ship the honest,
reproducible defaults and label them as untrained.

Adaptations vs the raw MAIN_TEMPLATE (same reasoning as agents/plamen06_steel):
  1. `read_deck_csv()` returned a relative "deck.csv" path read at import time.
     Under this repo's benchmark harness (cwd = repo root, which has its own
     deck.csv for a different agent's frozen deck) that would silently load the
     wrong deck. Replaced with `return my_deck` — the module-level deck injected
     by scripts/benchmark_agents.py's loader (or a bundle main.py) from the
     sibling deck.csv, matching every other bare-module agent here.
  2. Deck is therefore resolved at call time (inside agent()), not at import,
     so the loader's post-import `my_deck` injection is in scope.
The scoring functions (get_card/damaged_amount/card_type_score/option_score/
agent) are the template's inference versions verbatim — note option_score has
no random tiebreak at inference, so play is deterministic given the board.
"""

from cg.api import AreaType, CardType, OptionType, to_observation_class, all_card_data

# Notebook DEFAULT_WEIGHTS (its untrained initial policy). See provenance above.
LEARNED_WEIGHTS = {
    "attack": 3.0,
    "attach": 2.0,
    "evolve": 1.7,
    "play": 1.2,
    "ability": 1.0,
    "retreat": -0.2,
    "yes": 0.1,
    "no": 0.0,
    "card_basic": 1.1,
    "card_pokemon": 0.6,
    "card_energy": 0.45,
    "card_trainer": 0.35,
    "damage_target": 1.5,
    "own_damaged": 0.75,
    "active_bonus": 0.4,
    "bench_penalty": -0.1,
}

CARD_TABLE = {card.cardId: card for card in all_card_data()}


# See adaptation (1) above: deck comes from the injected module-level `my_deck`,
# never from a relative path read at import time.
def read_deck_csv():
    return my_deck


def get_card(obs, area, index, player_index):
    try:
        player = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return player.hand[index]
        if area == AreaType.DISCARD:
            return player.discard[index]
        if area == AreaType.ACTIVE:
            return player.active[index]
        if area == AreaType.BENCH:
            return player.bench[index]
        if area == AreaType.PRIZE:
            return player.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def damaged_amount(card):
    try:
        return max(0, int(card.maxHp) - int(card.hp))
    except Exception:
        return 0


def card_type_score(card, weights):
    if card is None:
        return 0.0
    data = CARD_TABLE.get(getattr(card, "id", -1))
    if data is None:
        return 0.0
    if data.cardType == CardType.POKEMON:
        return weights["card_basic"] if data.basic else weights["card_pokemon"]
    if data.cardType == CardType.ENERGY:
        return weights["card_energy"]
    return weights["card_trainer"]


def option_score(obs, option, weights):
    score = 0.0
    my_index = obs.current.yourIndex
    if option.type == OptionType.ATTACK:
        score += weights["attack"]
    elif option.type == OptionType.ATTACH:
        score += weights["attach"]
        target = get_card(obs, option.inPlayArea, option.inPlayIndex, my_index)
        if option.inPlayArea == AreaType.ACTIVE:
            score += weights["active_bonus"]
        if option.inPlayArea == AreaType.BENCH:
            score += weights["bench_penalty"]
        score += 0.03 * damaged_amount(target)
    elif option.type == OptionType.EVOLVE:
        score += weights["evolve"]
    elif option.type == OptionType.PLAY:
        score += weights["play"]
        card = get_card(obs, AreaType.HAND, option.index, my_index)
        score += card_type_score(card, weights)
    elif option.type == OptionType.ABILITY:
        score += weights["ability"]
    elif option.type == OptionType.RETREAT:
        score += weights["retreat"]
    elif option.type == OptionType.YES:
        score += weights["yes"]
    elif option.type == OptionType.NO:
        score += weights["no"]
    elif option.type == OptionType.CARD:
        card = get_card(obs, option.area, option.index, option.playerIndex)
        score += card_type_score(card, weights)
        if option.playerIndex != my_index:
            score += weights["damage_target"]
        else:
            score += weights["own_damaged"] * min(1.0, damaged_amount(card) / 100.0)
    elif option.type == OptionType.NUMBER:
        score += float(getattr(option, "number", 0))
    return score


def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        if obs_dict.get("select") is None:
            return read_deck_csv()
        return [0]
    if obs.select is None:
        return read_deck_csv()
    try:
        n = len(obs.select.option)
        if n <= 0:
            return []
        order = sorted(
            range(n),
            key=lambda i: option_score(obs, obs.select.option[i], LEARNED_WEIGHTS),
            reverse=True,
        )
        min_count = max(0, int(obs.select.minCount))
        max_count = min(n, int(obs.select.maxCount))
        k = max(1, min(max_count, max(min_count, 1)))
        return order[:k]
    except Exception:
        n = len(obs.select.option)
        k = max(1, min(n, int(obs.select.maxCount)))
        return list(range(k))
