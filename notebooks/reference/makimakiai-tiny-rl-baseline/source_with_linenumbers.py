import glob
import json
import math
import os
import random
import shutil
import tarfile
import textwrap
from pathlib import Path

SEED = 20260618
GENERATIONS = 4
POPULATION = 8
GAMES_PER_CANDIDATE = 4
FINAL_EVAL_GAMES = 20
MUTATION_SCALE = 0.35
MAX_STEPS_PER_GAME = 700

random.seed(SEED)

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
OUT_DIR = WORK_DIR / "rl_baseline_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"work_dir={WORK_DIR.resolve()}")
print(f"out_dir={OUT_DIR.resolve()}")
import sys

cg_candidates = []
cg_candidates += glob.glob("/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission")
cg_candidates += glob.glob("/kaggle/input/**/cg-lib", recursive=True)
cg_candidates += [str(Path("competitions/pokemon-tcg-ai-battle/input/sample_submission").resolve())]

for candidate in cg_candidates:
    if (Path(candidate) / "cg").exists():
        sys.path.append(candidate)
        CG_SOURCE_DIR = Path(candidate) / "cg"
        break
    if (Path(candidate) / "cg").exists() or (Path(candidate) / "api.py").exists():
        sys.path.append(str(Path(candidate).parent))
        CG_SOURCE_DIR = Path(candidate)
        break
else:
    raise FileNotFoundError("Could not find cg runtime. Add the competition input to this notebook.")

from cg.api import AreaType, CardType, OptionType, Pokemon, SelectContext, all_card_data, to_observation_class
from cg.game import battle_finish, battle_select, battle_start

CARD_TABLE = {card.cardId: card for card in all_card_data()}
print(f"loaded {len(CARD_TABLE)} cards")
print(f"cg source: {CG_SOURCE_DIR}")
def load_sample_deck() -> list[int]:
    candidates = [
        Path("/kaggle/input/competitions/pokemon-tcg-ai-battle/sample_submission/deck.csv"),
        Path("competitions/pokemon-tcg-ai-battle/input/sample_submission/deck.csv"),
    ]
    for path in candidates:
        if path.exists():
            cards = [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]
            if len(cards) == 60:
                return cards
    raise FileNotFoundError("sample_submission/deck.csv was not found")


DECK = load_sample_deck()
print(f"deck length={len(DECK)}")
print(DECK[:12], "...")
DEFAULT_WEIGHTS = {
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
    "random_noise": 0.02,
}


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


def damaged_amount(card) -> int:
    try:
        return max(0, int(card.maxHp) - int(card.hp))
    except Exception:
        return 0


def card_type_score(card, weights: dict[str, float]) -> float:
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


def option_score(obs, option, weights: dict[str, float]) -> float:
    """Convert one legal option into a numeric score.

    Higher scores are selected first. This is the policy that RL will tune.
    """
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

    score += random.random() * weights["random_noise"]
    return score


def choose_options(obs_dict, weights: dict[str, float]) -> list[int]:
    """Return option indexes sorted by the learned policy score.

    Kaggle requires indexes into obs.select.option, not card IDs or action names.
    """
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return DECK
    options = obs.select.option
    if not options:
        return []
    order = sorted(range(len(options)), key=lambda i: option_score(obs, options[i], weights), reverse=True)
    min_count = max(0, int(obs.select.minCount))
    max_count = min(len(options), int(obs.select.maxCount))
    k = max(1, min(max_count, max(min_count, 1)))
    return order[:k]


def make_agent(weights):
    def agent(obs_dict):
        if obs_dict.get("select") is None:
            return DECK
        try:
            return choose_options(obs_dict, weights)
        except Exception:
            obs = to_observation_class(obs_dict)
            n = len(obs.select.option)
            k = max(1, min(n, int(obs.select.maxCount)))
            return list(range(k))

    return agent
def play_game(agent_a, agent_b, deck_a, deck_b, max_steps=MAX_STEPS_PER_GAME) -> int:
    """Play one simulated game.

    Return +1 if agent_a wins, -1 if agent_a loses, and 0 for a draw or max-step cutoff.
    """
    obs, start_data = battle_start(deck_a, deck_b)
    if getattr(start_data, "errorPlayer", -1) >= 0:
        raise ValueError(f"deck error: player={start_data.errorPlayer}, type={start_data.errorType}")
    steps = 0
    try:
        while obs["current"]["result"] < 0 and steps < max_steps:
            your_index = obs["current"]["yourIndex"]
            selected = agent_a(obs) if your_index == 0 else agent_b(obs)
            obs = battle_select(selected)
            steps += 1
        result = obs["current"]["result"]
    finally:
        battle_finish()
    if result == 0:
        return 1
    if result == 1:
        return -1
    return 0


def evaluate_weights(weights, games=GAMES_PER_CANDIDATE) -> dict:
    """Evaluate one policy by playing it against the default baseline policy."""
    learned_agent = make_agent(weights)
    baseline_agent = make_agent(DEFAULT_WEIGHTS)
    rewards = []
    for game_idx in range(games):
        if game_idx % 2 == 0:
            reward = play_game(learned_agent, baseline_agent, DECK, DECK)
        else:
            reward = -play_game(baseline_agent, learned_agent, DECK, DECK)
        rewards.append(reward)
    wins = sum(1 for reward in rewards if reward > 0)
    losses = sum(1 for reward in rewards if reward < 0)
    draws = sum(1 for reward in rewards if reward == 0)
    return {"reward": sum(rewards), "wins": wins, "losses": losses, "draws": draws}


def mutate_weights(parent, scale=MUTATION_SCALE):
    """Create a nearby policy by adding random noise to each weight."""
    child = dict(parent)
    for key, value in child.items():
        if key == "random_noise":
            child[key] = max(0.0, min(0.2, value + random.gauss(0, scale * 0.05)))
        else:
            child[key] = value + random.gauss(0, scale)
    return child


best_weights = dict(DEFAULT_WEIGHTS)
history = []

for generation in range(GENERATIONS):
    candidates = [best_weights] + [mutate_weights(best_weights) for _ in range(POPULATION - 1)]
    rows = []
    for idx, weights in enumerate(candidates):
        metrics = evaluate_weights(weights)
        row = {"generation": generation, "candidate": idx, **metrics, "weights": weights}
        rows.append(row)
        print(
            f"gen={generation:02d} cand={idx:02d} "
            f"reward={metrics['reward']:>3} W/L/D={metrics['wins']}/{metrics['losses']}/{metrics['draws']}"
        )
    rows.sort(key=lambda row: (row["reward"], row["wins"], -row["losses"]), reverse=True)
    best_weights = rows[0]["weights"]
    history.extend(rows)
    print(f"best generation {generation}: reward={rows[0]['reward']} weights={json.dumps(best_weights)}")

weights_path = OUT_DIR / "learned_policy_weights.json"
weights_path.write_text(json.dumps(best_weights, indent=2), encoding="utf-8")
history_path = OUT_DIR / "training_history.json"
history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

best_weights
generation_ids = sorted({row["generation"] for row in history})
best_rewards = []
mean_rewards = []

for generation in generation_ids:
    rewards = [row["reward"] for row in history if row["generation"] == generation]
    best_rewards.append(max(rewards))
    mean_rewards.append(sum(rewards) / len(rewards))


def save_svg_learning_curve(path: Path):
    width, height = 900, 500
    margin_left, margin_right, margin_top, margin_bottom = 70, 30, 55, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_values = best_rewards + mean_rewards + [0]
    y_min = min(all_values) - 1
    y_max = max(all_values) + 1
    x_min = min(generation_ids)
    x_max = max(generation_ids) if max(generation_ids) != x_min else x_min + 1

    def x_scale(x):
        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def y_scale(y):
        return margin_top + (y_max - y) / (y_max - y_min) * plot_h

    def points(values):
        return " ".join(f"{x_scale(x):.1f},{y_scale(y):.1f}" for x, y in zip(generation_ids, values))

    zero_y = y_scale(0)
    grid_lines = []
    for y in range(math.floor(y_min), math.ceil(y_max) + 1):
        py = y_scale(y)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{py:.1f}" x2="{width-margin_right}" y2="{py:.1f}" '
            f'stroke="#e6e8eb" stroke-width="1"/>'
            f'<text x="{margin_left-12}" y="{py+4:.1f}" text-anchor="end" font-size="12" fill="#5f6368">{y}</text>'
        )
    x_labels = []
    for x in generation_ids:
        px = x_scale(x)
        x_labels.append(
            f'<text x="{px:.1f}" y="{height-margin_bottom+28}" text-anchor="middle" '
            f'font-size="12" fill="#5f6368">{x}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width/2}" y="30" text-anchor="middle" font-size="24" font-weight="700" fill="#202124">Tiny RL policy search reward by generation</text>
  <line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width-margin_right}" y2="{zero_y:.1f}" stroke="#777" stroke-width="1.5" stroke-dasharray="5 5"/>
  {''.join(grid_lines)}
  <polyline fill="none" stroke="#1a73e8" stroke-width="4" points="{points(best_rewards)}"/>
  <polyline fill="none" stroke="#f9ab00" stroke-width="4" points="{points(mean_rewards)}"/>
  {''.join(f'<circle cx="{x_scale(x):.1f}" cy="{y_scale(y):.1f}" r="5" fill="#1a73e8"/>' for x, y in zip(generation_ids, best_rewards))}
  {''.join(f'<rect x="{x_scale(x)-5:.1f}" y="{y_scale(y)-5:.1f}" width="10" height="10" fill="#f9ab00"/>' for x, y in zip(generation_ids, mean_rewards))}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#3c4043" stroke-width="1.5"/>
  <line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#3c4043" stroke-width="1.5"/>
  {''.join(x_labels)}
  <text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14" fill="#3c4043">Generation</text>
  <text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle" font-size="14" fill="#3c4043">Reward per candidate</text>
  <rect x="{width-255}" y="62" width="205" height="62" rx="8" fill="#ffffff" stroke="#dadce0"/>
  <line x1="{width-238}" y1="84" x2="{width-202}" y2="84" stroke="#1a73e8" stroke-width="4"/>
  <text x="{width-192}" y="89" font-size="13" fill="#3c4043">best reward</text>
  <line x1="{width-238}" y1="108" x2="{width-202}" y2="108" stroke="#f9ab00" stroke-width="4"/>
  <text x="{width-192}" y="113" font-size="13" fill="#3c4043">mean reward</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


svg_path = WORK_DIR / "training_curve.svg"
artifact_svg_path = OUT_DIR / "training_curve.svg"
save_svg_learning_curve(svg_path)
save_svg_learning_curve(artifact_svg_path)

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(generation_ids, best_rewards, marker="o", linewidth=2.5, label="best reward")
    plt.plot(generation_ids, mean_rewards, marker="s", linewidth=2.0, label="mean reward")
    plt.axhline(0, color="#777777", linewidth=1, linestyle="--")
    plt.title("Tiny RL policy search reward by generation")
    plt.xlabel("Generation")
    plt.ylabel("Reward per candidate")
    plt.xticks(generation_ids)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    training_curve_path = WORK_DIR / "training_curve.png"
    artifact_curve_path = OUT_DIR / "training_curve.png"
    plt.savefig(training_curve_path, dpi=160)
    plt.savefig(artifact_curve_path, dpi=160)
    plt.show()
    print(f"saved learning curve PNG: {training_curve_path}")
except ModuleNotFoundError:
    try:
        from IPython.display import SVG, display

        display(SVG(filename=str(svg_path)))
    except Exception:
        pass
    print(f"matplotlib is not available; saved learning curve SVG: {svg_path}")
final_eval = evaluate_weights(best_weights, games=FINAL_EVAL_GAMES)
final_eval["games"] = FINAL_EVAL_GAMES
final_eval["opponent"] = "default_weight_baseline"
final_eval["note"] = "Small post-training battle check. Increase FINAL_EVAL_GAMES for a stronger estimate."

final_eval_path = OUT_DIR / "final_battle_evaluation.json"
final_eval_path.write_text(json.dumps(final_eval, indent=2), encoding="utf-8")
print(json.dumps(final_eval, indent=2))
MAIN_TEMPLATE = r'''
import os
import random

from cg.api import AreaType, CardType, OptionType, to_observation_class, all_card_data

LEARNED_WEIGHTS = __WEIGHTS_JSON__

CARD_TABLE = {card.cardId: card for card in all_card_data()}


def read_deck_csv():
    file_path = "deck.csv"
    if not os.path.exists(file_path):
        file_path = "/kaggle_simulations/agent/" + file_path
    with open(file_path, "r") as file:
        return [int(line.strip()) for line in file.read().splitlines() if line.strip()][:60]


MY_DECK = read_deck_csv()


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
            return MY_DECK
        return [0]
    if obs.select is None:
        return MY_DECK
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
'''

main_py = MAIN_TEMPLATE.replace("__WEIGHTS_JSON__", json.dumps(best_weights, sort_keys=True))
(WORK_DIR / "main.py").write_text(main_py.strip() + "\n", encoding="utf-8")
(OUT_DIR / "main.py").write_text(main_py.strip() + "\n", encoding="utf-8")
print((WORK_DIR / "main.py").read_text(encoding="utf-8")[:800])
deck_text = "\n".join(str(card_id) for card_id in DECK) + "\n"
(WORK_DIR / "deck.csv").write_text(deck_text, encoding="utf-8")
(OUT_DIR / "deck.csv").write_text(deck_text, encoding="utf-8")
print(f"deck.csv lines={len(DECK)}")
submission_path = WORK_DIR / "submission.tar.gz"
if submission_path.exists():
    submission_path.unlink()

with tarfile.open(submission_path, "w:gz") as tar:
    tar.add(WORK_DIR / "main.py", arcname="main.py")
    tar.add(WORK_DIR / "deck.csv", arcname="deck.csv")

    def clean_cg_filter(tarinfo):
        name = Path(tarinfo.name).name
        if name.startswith("._") or name == "__pycache__" or tarinfo.name.endswith(".pyc"):
            return None
        return tarinfo

    tar.add(CG_SOURCE_DIR, arcname="cg", filter=clean_cg_filter)

with tarfile.open(submission_path, "r:gz") as tar:
    names = tar.getnames()

print(f"created: {submission_path}")
print("top-level entries:", sorted({name.split('/')[0] for name in names})[:10])
print("contains main.py:", "main.py" in names)
print("contains deck.csv:", "deck.csv" in names)
print("contains cg:", any(name.startswith("cg/") for name in names))
