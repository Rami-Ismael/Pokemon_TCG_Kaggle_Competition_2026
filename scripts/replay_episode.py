"""Step through one recorded PTCG episode, turn by turn.

Usage:
    uv run python scripts/replay_episode.py                          # first eval episode
    uv run python scripts/replay_episode.py <episode_id>             # e.g. 88344168
    uv run python scripts/replay_episode.py --split train <id>       # from train split
    uv run python scripts/replay_episode.py --local                  # local_battle.json
    uv run python scripts/replay_episode.py <id> --agent 1           # agent 1's POV
    uv run python scripts/replay_episode.py <id> --full              # dump full obs each step
    uv run python scripts/replay_episode.py <id> --max-steps 20

Episodes are kaggle-environments records: {"steps": [[agent0, agent1], ...]}
Each agent entry: {action, observation, reward, status, info}.
observation.select  = the legal options the engine offered this step
observation.current = the board state the agent could see (its POV)
action              = list of option indices the agent picked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pokemon_tcg.config import EPISODES_DIR  # noqa: E402

SPLIT_DIRS = {
    "train": EPISODES_DIR / "splits" / "train-2026-07-26",
    "eval": EPISODES_DIR / "splits" / "eval-2026-07-27",
}


def resolve_episode(args) -> Path:
    if args.local:
        return EPISODES_DIR / "local" / "local_battle.json"
    folder = SPLIT_DIRS[args.split]
    if args.episode_id:
        p = folder / f"{args.episode_id}.json"
        if not p.exists():
            sys.exit(f"episode not found: {p}")
        return p
    files = sorted(folder.glob("*.json"))
    if not files:
        sys.exit(f"no episodes in {folder}")
    return files[0]


def brief_obs(obs: dict) -> str:
    """One-line summary of what the agent was offered/saw."""
    sel = obs.get("select") or {}
    cur = obs.get("current") or {}
    opts = sel.get("option") or []
    parts = [
        f"select.type={sel.get('type')} ctx={sel.get('context')}",
        f"n_options={len(opts)}",
    ]
    if cur:
        you = (cur.get("players") or [{}])[cur.get("yourIndex", 0)]
        opp_i = 1 - cur.get("yourIndex", 0)
        opp = (cur.get("players") or [{}, {}])[opp_i]
        parts.append(
            "turn={turn} you(hand={h},deck={d},active={a},bench={b}) "
            "opp(hand={oh},deck={od})".format(
                turn=cur.get("turn"),
                h=you.get("handCount"), d=you.get("deckCount"),
                a=len(you.get("active") or []), b=len(you.get("bench") or []),
                oh=opp.get("handCount"), od=opp.get("deckCount"),
            )
        )
    return " | ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_id", nargs="?", help="episode id (default: first file in split)")
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--local", action="store_true", help="use data/episodes/local/local_battle.json")
    ap.add_argument("--agent", type=int, default=0, choices=[0, 1], help="whose POV (default 0)")
    ap.add_argument("--full", action="store_true", help="print full obs_dict JSON per step")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    path = resolve_episode(args)
    data = json.loads(path.read_text())

    # kaggle-environments record: {"steps": [[agent0, agent1], ...], "rewards": [...]}
    steps = data["steps"] if isinstance(data, dict) else data
    print(f"episode: {path.name}  steps={len(steps)}  agent={args.agent}")
    if isinstance(data, dict) and "rewards" in data:
        print(f"final rewards: {data['rewards']}  statuses: {data.get('statuses')}")
    print("=" * 70)

    n = 0
    for i, step in enumerate(steps):
        if args.max_steps and n >= args.max_steps:
            print(f"... stopped at --max-steps {args.max_steps}")
            break
        # local files use 'obs'; kaggle-environments use 'observation'
        agent = step[args.agent] if isinstance(step, list) else step
        obs = agent.get("observation") or agent.get("obs") or {}
        action = agent.get("action")
        reward = agent.get("reward")
        status = agent.get("status")
        n += 1

        print(f"\n[step {i}] reward={reward} status={status}")
        if args.full:
            print("obs_dict:", json.dumps(obs, indent=2)[:4000])
        else:
            print("obs:", brief_obs(obs))
        print("action:", action)


if __name__ == "__main__":
    main()
