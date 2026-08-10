"""Timing measurements for the search/battle API port (search_api.py).

a) search_step cost (ms) and per-decision MCTS cost vs SEARCH_COUNT, on the
   current device config. Run once normally and once as the evaluator
   rehearsal (PTCG_DEVICE=cpu forces single-thread here). From this, the max
   affordable SEARCH_COUNT within the ladder's ~1 s/turn budget.
b) Direct battle-API self-play throughput (games/s, decisions/s) with
   random agents (pure engine, no NN), vs the kaggle_environments env.run
   path measured on the same machine in the same run.

Usage:
    uv run python scripts/measure_search_api.py                # laptop config
    PTCG_DEVICE=cpu uv run python scripts/measure_search_api.py  # rehearsal

Writes reports/search_api_measurements_<device>.json.
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.device import resolve_device  # noqa: E402


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def measure_mcts_cost(sa, model, deck, search_counts, games_per=4, seed=0):
    """Per-decision wall time + engine search_step time at each SEARCH_COUNT."""
    out = {}
    for sc in search_counts:
        random.seed(seed)
        step_times: list[float] = []
        decision_times: list[float] = []
        n_games = 0
        with torch.inference_mode():
            for g in range(games_per):
                obs, _ = sa.battle_start(deck, deck)
                your_index = g % 2
                while obs["current"]["result"] < 0:
                    if obs["current"]["yourIndex"] == your_index:
                        t0 = time.perf_counter()
                        sel, _ = sa.mcts_agent(obs, deck, model, sc, step_timings=step_times)
                        decision_times.append(time.perf_counter() - t0)
                    else:
                        sel = sa.random_agent(obs)
                    obs = sa.battle_select(sel)
                sa.battle_finish()
                n_games += 1
        out[sc] = {
            "games": n_games,
            "decisions": len(decision_times),
            "decision_ms_mean": st.mean(decision_times) * 1000,
            "decision_ms_p50": pct(decision_times, 50) * 1000,
            "decision_ms_p95": pct(decision_times, 95) * 1000,
            "decision_ms_max": max(decision_times) * 1000,
            "search_steps": len(step_times),
            "search_step_ms_mean": st.mean(step_times) * 1000 if step_times else None,
            "search_step_ms_p95": pct(step_times, 95) * 1000 if step_times else None,
        }
        print(
            f"SEARCH_COUNT={sc}: decision mean {out[sc]['decision_ms_mean']:.1f} ms "
            f"p95 {out[sc]['decision_ms_p95']:.1f} ms max {out[sc]['decision_ms_max']:.1f} ms; "
            f"engine search_step mean {out[sc]['search_step_ms_mean']:.2f} ms "
            f"({out[sc]['decisions']} decisions)",
            flush=True,
        )
    return out


def measure_battle_api_throughput(sa, games=60, seed=0):
    """Random-vs-random through the direct battle API: pure engine speed."""
    random.seed(seed)
    t0 = time.perf_counter()
    decisions = 0
    for _ in range(games):
        obs, _ = sa.battle_start(sa.sample_deck, sa.sample_deck)
        while obs["current"]["result"] < 0:
            obs = sa.battle_select(sa.random_agent(obs))
            decisions += 1
        sa.battle_finish()
    dt = time.perf_counter() - t0
    res = {
        "games": games,
        "decisions": decisions,
        "secs": dt,
        "games_per_sec": games / dt,
        "decisions_per_sec": decisions / dt,
    }
    print(
        f"battle API random-vs-random: {games} games, {decisions} decisions in {dt:.1f}s "
        f"-> {res['games_per_sec']:.1f} games/s, {res['decisions_per_sec']:.0f} decisions/s",
        flush=True,
    )
    return res


def measure_env_run_throughput(deck, games=20, seed=0):
    """Control: same machine, kaggle_environments env.run with random agents."""
    from kaggle_environments import make

    def rand_agent(obs):
        sel = obs.get("select")
        if sel is None:
            return deck  # initial deck-submission step
        n = len(sel["option"])
        k = sel["maxCount"] if sel.get("maxCount") is not None else 1
        k = min(k, n)
        return random.sample(range(n), k) if k > 0 else []

    t0 = time.perf_counter()
    steps = 0
    for _ in range(games):
        env = make("cabt")
        env.run([rand_agent, rand_agent])
        steps += len(env.steps)
    dt = time.perf_counter() - t0
    res = {
        "games": games,
        "env_steps": steps,
        "secs": dt,
        "games_per_sec": games / dt,
        "steps_per_sec": steps / dt,
    }
    print(
        f"env.run random-vs-random: {games} games, {steps} env steps in {dt:.1f}s "
        f"-> {res['games_per_sec']:.2f} games/s, {res['steps_per_sec']:.0f} steps/s",
        flush=True,
    )
    return res


def main() -> None:
    device_str = resolve_device()
    if device_str == "cpu":
        torch.set_num_threads(1)  # evaluator envelope rehearsal

    from pokemon_tcg import search_api as sa

    torch.manual_seed(config.RANDOM_SEED)
    model = sa.MyModel(128, 2, 256, 1, 1).to(device_str)
    model.eval()

    print(f"device={device_str} torch_threads={torch.get_num_threads()}", flush=True)
    report = {
        "device": device_str,
        "torch_threads": torch.get_num_threads(),
        "mcts_cost_by_search_count": measure_mcts_cost(
            sa, model, sa.sample_deck, search_counts=[5, 10, 20, 40]
        ),
        "battle_api_throughput": measure_battle_api_throughput(sa),
        "env_run_throughput": measure_env_run_throughput(sa.sample_deck),
    }

    out = config.REPORTS_DIR / f"search_api_measurements_{device_str}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
