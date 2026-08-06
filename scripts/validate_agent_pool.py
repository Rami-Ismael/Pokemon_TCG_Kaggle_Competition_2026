"""Which wired agents are actually usable as benchmark opponents?

An unusable opponent is worse than a missing one: a cg-ILLEGAL deck loses every
game instantly, and every agent that faces it banks free wins, which inflates the
whole table. A crashing agent does the same. Neither raises in the harness.

For each agent this plays a short match against `random_legal` and records:
  loads          the module imported and exposes a callable `agent`
  submits_deck   agent({}) returned a plausible 60-card list
  plays          games completed without an exception
  s/game         wall clock -- a sub-0.1s game means the deck was rejected at
                 battle_start, not that the agent is fast
  wins           record vs random_legal (the floor)

Verdict:
  OK        plays, and beats the random floor at least sometimes
  WEAK      plays legally but loses to random_legal -- usable, but it is a floor
            agent, not a competitor; too many of these flatten the pool
  SUSPECT   completes games but suspiciously fast, or submits a malformed deck
  BROKEN    fails to load, crashes, or loses every game at battle_start speed

Usage:
  uv run python scripts/validate_agent_pool.py --out reports/agent_pool_validation.json
  uv run python scripts/validate_agent_pool.py --agents tb_iono,wmh_mewtwo --pairs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import benchmark_agents as B  # noqa: E402

# Our own experiment arms -- not opponents, excluded from pool validation.
ARM_PREFIXES = ("s2_", "ppo_", "grid_", "il_bc", "il_medium", "il_small",
                "il_hfstream", "il_alldays")


def validate(name: str, pairs: int) -> dict:
    rec: dict = {"agent": name, "loads": False, "submits_deck": False,
                 "plays": False, "games": 0, "wins": 0, "sec_per_game": None,
                 "deck_cards": None, "verdict": "BROKEN", "error": None}
    try:
        fn = B.load_agent(name)
        rec["loads"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"load: {exc!r}"[:300]
        return rec

    try:
        deck = list(fn({}) or [])
        rec["deck_cards"] = len(deck)
        rec["submits_deck"] = len(deck) == 60 and all(isinstance(c, int) for c in deck)
    except Exception as exc:  # noqa: BLE001
        # Many agents build an Observation from the dict before checking for
        # `select`, so a bare agent({}) probe throws for them. That says nothing
        # about whether they PLAY -- it is a limitation of this probe, not a
        # defect in the agent. Record it and let the play test decide.
        rec["deck_probe_unsupported"] = True
        rec["error"] = f"deck-probe (non-blocking): {exc!r}"[:200]

    try:
        floor = B.load_agent("random_legal")
        env = lambda: __import__("kaggle_environments").make("cabt")  # noqa: E731
        t0 = time.time()
        w, l, d, _ = B.play_match(fn, floor, env, pairs=pairs,
                                  name_a=name, name_b="random_legal")
        elapsed = time.time() - t0
        rec["games"] = pairs * 2
        rec["wins"] = w
        rec["sec_per_game"] = round(elapsed / max(1, pairs * 2), 3)
        rec["plays"] = True
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"play: {exc!r}"[:300]
        return rec

    spg = rec["sec_per_game"] or 0
    if rec["wins"] == 0 and spg < 0.10:
        rec["verdict"] = "BROKEN"          # loses instantly => deck rejected
        rec["error"] = rec["error"] or "loses every game at battle_start speed (likely cg-ILLEGAL deck)"
    elif not rec["submits_deck"] and not rec.get("deck_probe_unsupported"):
        rec["verdict"] = "SUSPECT"
    elif rec["wins"] == 0:
        rec["verdict"] = "WEAK"
    else:
        rec["verdict"] = "OK"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default=None, help="comma-separated; default = all non-arm agents")
    ap.add_argument("--pairs", type=int, default=2, help="mirrored pairs vs random_legal")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "agent_pool_validation.json")
    args = ap.parse_args()

    if args.agents:
        names = [a for a in args.agents.split(",") if a]
    else:
        names = [k for k, v in B.AGENT_FILES.items()
                 if v.exists() and not k.startswith(ARM_PREFIXES) and k != "random_legal"]

    rows = []
    for i, n in enumerate(sorted(names), 1):
        r = validate(n, args.pairs)
        rows.append(r)
        print(f"[{i}/{len(names)}] {r['verdict']:8s} {n:32s} "
              f"{r['wins']}/{r['games']} vs floor  {r['sec_per_game']}s/game"
              + (f"  {r['error']}" if r["error"] else ""), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))

    by = {}
    for r in rows:
        by.setdefault(r["verdict"], []).append(r["agent"])
    print("\n=== summary ===")
    for v in ("OK", "WEAK", "SUSPECT", "BROKEN"):
        got = by.get(v, [])
        print(f"{v:8s} {len(got):3d}  {', '.join(sorted(got)) if got else '—'}")
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
