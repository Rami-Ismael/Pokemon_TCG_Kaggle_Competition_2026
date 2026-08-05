"""Legality gate for deck pools — blocking prerequisite of the deck-diversity
exploitability sweep (notes/experiments/2026-08-04-deck-diversity-exploitability-sweep.md).

Public decks scraped from other competitors' agents can be cg-ILLEGAL (the
known case: more than one ACE SPEC, which makes the engine return INVALID every
game). An illegal deck inside a pool converts that share of episodes into free
wins for whichever side is still standing, inflating the exploiter's win rate —
i.e. it would manufacture the very "no decay with diversity" result the sweep is
built to test. Every deck must pass this gate before entering a pool.

Two checks per deck:
  1. SUBMIT — hand the deck to both seats and step once. cabt validates deck
     construction here: an illegal list comes back INVALID immediately.
  2. PLAY-OUT (--play-out, default on) — finish one game with safe-choice
     agents on both seats, catching decks that are legal at construction but
     fault or softlock mid-game.

Writes a JSON report and a pass-list manifest that feeds straight back in as a
pool spec:

    .venv-ppo/bin/python scripts/probe_deck_legality.py
    ... --deck-pool '@configs/deck_pools/legal_agents.txt'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.deck_pool import _expand_spec, parse_deck  # noqa: E402
from pokemon_tcg.selfplay import _safe_choice  # noqa: E402  (also sets cg path)

BAD_STATUSES = {"INVALID", "ERROR"}


def probe_deck(kaggle_make, deck: list[int], play_out: bool,
               max_steps: int = 400) -> dict:
    """Return {'legal': bool, 'stage': str, 'statuses': [...], 'steps': int}."""
    env = kaggle_make("cabt")
    env.reset(2)
    env.step([deck, deck])
    statuses = [env.state[i]["status"] for i in range(2)]
    if any(s in BAD_STATUSES for s in statuses):
        return {"legal": False, "stage": "submit", "statuses": statuses,
                "steps": 1}
    if not play_out:
        return {"legal": True, "stage": "submit", "statuses": statuses,
                "steps": 1}

    steps = 1
    while not env.done and steps < max_steps:
        actions = []
        for i in range(2):
            if env.state[i]["status"] != "ACTIVE":
                actions.append([])
                continue
            sel = env.state[i]["observation"].get("select")
            # sel None mid-game would be another deck request; feed the deck.
            actions.append(_safe_choice(sel) if sel else deck)
        env.step(actions)
        steps += 1
        statuses = [env.state[i]["status"] for i in range(2)]
        if any(s in BAD_STATUSES for s in statuses):
            return {"legal": False, "stage": "play", "statuses": statuses,
                    "steps": steps}
    if steps >= max_steps and not env.done:
        # Not an illegality, but unusable in a pool: episodes would never end.
        return {"legal": False, "stage": "timeout", "statuses": statuses,
                "steps": steps}
    return {"legal": True, "stage": "play", "statuses": statuses,
            "steps": steps}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck-pool", default="all:agents",
                    help="pool spec to probe (default all:agents, "
                         "content-deduped); also accepts all:decklists, a "
                         "comma-separated ref list, or @manifest.txt")
    ap.add_argument("--play-out", dest="play_out", action="store_true",
                    default=True, help="also finish one safe-choice game (default)")
    ap.add_argument("--no-play-out", dest="play_out", action="store_false",
                    help="deck-submission check only (faster, misses mid-game faults)")
    ap.add_argument("--max-steps", type=int, default=400,
                    help="play-out step cap; hitting it fails the deck as unusable")
    ap.add_argument("--out", type=Path,
                    default=config.REPORTS_DIR / "deck_legality.json")
    ap.add_argument("--manifest", type=Path,
                    default=config.PROJECT_ROOT / "configs" / "deck_pools" / "legal_agents.txt",
                    help="pass-list written here, usable as '@<path>' pool spec")
    args = ap.parse_args()

    from kaggle_environments import make as kaggle_make

    pairs = _expand_spec(args.deck_pool)
    print(f"probing {len(pairs)} deck(s) from spec {args.deck_pool!r} "
          f"(play_out={args.play_out})")

    results, t0 = [], time.time()
    for name, path in pairs:
        deck = parse_deck(path)
        if len(deck) != 60:
            res = {"legal": False, "stage": "parse", "statuses": [],
                   "steps": 0, "note": f"{len(deck)} cards, expected 60"}
        else:
            res = probe_deck(kaggle_make, deck, args.play_out, args.max_steps)
        res.update({"name": name, "path": str(path)})
        results.append(res)
        print(f"  {'PASS' if res['legal'] else 'FAIL'}  {name:<36} "
              f"{res['stage']:<8} {res['statuses']}")

    legal = [r for r in results if r["legal"]]
    failed = [r for r in results if not r["legal"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "spec": args.deck_pool, "play_out": args.play_out,
        "probed": len(results), "legal": len(legal), "failed": len(failed),
        "seconds": round(time.time() - t0, 1),
        "results": results,
    }, indent=2))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        f"# legality-verified decks from spec {args.deck_pool!r}\n"
        f"# generated by scripts/probe_deck_legality.py "
        f"(play_out={args.play_out}); K={len(legal)}\n"
        + "".join(f"{r['path']}\n" for r in legal))

    print(f"\n{len(legal)}/{len(results)} legal  [{time.time() - t0:.0f}s]")
    if failed:
        print("EXCLUDED: " + ", ".join(f"{r['name']}({r['stage']})" for r in failed))
    print(f"report:   {args.out}")
    print(f"manifest: {args.manifest}   → use as --deck-pool '@{args.manifest}'")


if __name__ == "__main__":
    main()
