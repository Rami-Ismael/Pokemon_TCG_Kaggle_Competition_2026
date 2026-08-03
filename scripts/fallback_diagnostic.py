"""Fallback diagnostic: is the agent playing its model, or its safety net?

A fallback that fires silently is worse than a crash: the agent "runs",
scores badly, and nothing says whether the model is weak or `_safe_choice`
is answering 40% of decisions. This driver plays a target agent against
chosen opponents with PTCG_FALLBACK_TRACK=1 and reports, by cause, how
often each never-crash fallback path actually fired -- as a RATE over the
decision count, because 500 fallbacks in 200k decisions is noise while 500
in 1,200 means the agent is effectively playing random.

Reasons are counted by the tracker in agents/il_agent/agent_core.py (and
re-exported by the s2_arms wrappers); the rule agents' coarser _DIAG
pattern is collected too when the target or opponent exposes it.

USAGE
-----
    uv run python scripts/fallback_diagnostic.py                          # il_agent vs rule_baseline, 2 games
    uv run python scripts/fallback_diagnostic.py --pairs 3                # 6 games
    uv run python scripts/fallback_diagnostic.py --opponents rung2        # vs the public benchmark pool
    uv run python scripts/fallback_diagnostic.py --agent s2_e0_s42 --opponents rule_baseline,grunt
    uv run python scripts/fallback_diagnostic.py --assert-clean           # CI gate: model_action_illegal must be 0

Output: human-readable report on stdout, machine summary on stderr, full
JSON (including first-occurrence state/tracebacks per reason) at --out,
TensorBoard scalars under --tb-dir unless --no-tb.

Exit codes: 0 ok; 2 when --assert-clean finds a masking bug
(model_action_illegal > 0) or an empty run (decisions == 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must be set before any agent module is exec'd: the tracker reads it at
# import time (agents/il_agent/agent_core.py::_TRACK).
os.environ["PTCG_FALLBACK_TRACK"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_agents as bench  # noqa: E402  (shares cg bootstrap + agent registry)

REPO = bench.REPO


def _print_first_occurrence(reason: str, first: dict) -> None:
    print(f"    first `{reason}`:")
    for k, v in first.items():
        if k == "obs":
            sel = (v or {}).get("select") or {}
            cur = (v or {}).get("current") or {}
            print(f"      obs: turn={cur.get('turn')} options={len(sel.get('option') or [])} "
                  f"minCount={sel.get('minCount')} maxCount={sel.get('maxCount')} "
                  f"(full state in the JSON report)")
        elif k == "traceback":
            tail = [ln for ln in str(v).strip().splitlines() if ln.strip()][-2:]
            for ln in tail:
                print(f"      {ln.strip()}")
        else:
            print(f"      {k}: {v}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default="il_agent",
                    help="target agent (AGENT_FILES key, @deck-tag ok; default il_agent)")
    ap.add_argument("--opponents", default="rule_baseline",
                    help="comma-separated opponents or group names (rung2, ours, floor)")
    ap.add_argument("--pairs", type=int, default=1,
                    help="mirrored game pairs per opponent (2 games each; default 1)")
    ap.add_argument("--out", type=Path, default=REPO / "reports" / "fallback_diagnostic.json")
    ap.add_argument("--tb-dir", type=Path, default=REPO / "runs" / "fallback_diag")
    ap.add_argument("--no-tb", action="store_true", help="skip TensorBoard scalars")
    ap.add_argument("--assert-clean", action="store_true",
                    help="exit 2 unless model_action_illegal == 0 and decisions > 0 "
                         "(masking-correctness gate, see tests/test_fallback_tracking.py)")
    args = ap.parse_args()

    from kaggle_environments import make

    opponents = [o for o in bench.resolve_agent_selection(
        [t.strip() for t in args.opponents.split(",") if t.strip()]
    ) if o != args.agent]
    if not opponents:
        sys.exit("no opponents resolved")

    bench._LOADED_MODULES.clear()
    print(f"target: {args.agent}   opponents: {', '.join(opponents)}   "
          f"pairs/opponent: {args.pairs} ({2 * args.pairs} games each)")
    agent_fn = bench.load_agent(args.agent)
    opp_fns = {o: bench.load_agent(o) for o in opponents}
    for mod in bench._LOADED_MODULES.values():
        if hasattr(mod, "diag_reset"):
            mod.diag_reset()

    record = {"wins": 0, "losses": 0, "draws": 0}
    per_opponent: dict[str, dict] = {}
    for o in opponents:
        aw, bw, dr, secs = bench.play_match(
            agent_fn, opp_fns[o], lambda: make("cabt"), args.pairs,
            name_a=args.agent, name_b=o,
        )
        per_opponent[o] = {"wins": aw, "losses": bw, "draws": dr, "seconds": round(secs, 1)}
        record["wins"] += aw
        record["losses"] += bw
        record["draws"] += dr
        print(f"  vs {o}: {aw}-{bw}{('-' + str(dr) + 'd') if dr else ''}  ({secs:.1f}s)")

    target_mod = bench._LOADED_MODULES[args.agent]
    if not hasattr(target_mod, "diag_snapshot"):
        sys.exit(f"{args.agent} exposes no diag_snapshot(); nothing to measure. "
                 f"Instrumented agents: il_agent, s2_* wrappers, rule_baseline-family.")
    snap = target_mod.diag_snapshot()
    first = getattr(target_mod, "diag_first", dict)() or {}

    decisions = snap.get("decisions", 0)
    fallbacks = snap.get("fallbacks", 0)
    games = 2 * args.pairs * len(opponents)
    print(f"\n=== Fallback report: {args.agent}, {games} games ===")
    print(f"decisions={decisions}  fallbacks={fallbacks}  "
          f"fallback_rate={snap.get('fallback_rate', 0.0):.2%}")
    reason_counts = {k: v for k, v in sorted(snap.items())
                     if k not in ("enabled", "decisions", "fallbacks", "fallback_rate")
                     and isinstance(v, int)}
    if reason_counts:
        print("by reason (fallbacks + silent-degradation signals):")
        for k, v in reason_counts.items():
            print(f"  {k:40s} {v:6d}  ({v / max(1, decisions):.2%} of decisions)")
    else:
        print("no fallback or degradation signal fired.")
    for reason, f in first.items():
        _print_first_occurrence(reason, f)

    # Opponents with the coarser rule-agent _DIAG pattern: collect, don't judge.
    opponent_diag = {
        o: bench._LOADED_MODULES[o].diag_snapshot()
        for o in opponents if hasattr(bench._LOADED_MODULES.get(o), "diag_snapshot")
    }

    result = {
        "agent": args.agent,
        "opponents": opponents,
        "pairs_per_opponent": args.pairs,
        "games": games,
        "record": record,
        "per_opponent": per_opponent,
        "fallback": snap,
        "fallback_first": first,
        "opponent_diag": opponent_diag,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved: {args.out}")

    print(f"[fallback-diag] {args.agent}: {fallbacks}/{decisions} = "
          f"{snap.get('fallback_rate', 0.0):.2%} over {games} games; "
          f"record {record['wins']}-{record['losses']}-{record['draws']}",
          file=sys.stderr)

    if not args.no_tb:
        bench._write_fallback_tb({args.agent: snap}, args.tb_dir, step=int(time.time()))

    if args.assert_clean:
        illegal = sum(v for k, v in snap.items()
                      if isinstance(v, int) and k.split(":", 1)[0] == "model_action_illegal")
        if decisions == 0:
            print("[fallback-diag] ASSERT-CLEAN FAIL: 0 decisions measured "
                  "(tracking off, or the agent never got a select)", file=sys.stderr)
            return 2
        if illegal > 0:
            print(f"[fallback-diag] ASSERT-CLEAN FAIL: model_action_illegal={illegal} "
                  f"-- masking bug, not a fallback", file=sys.stderr)
            return 2
        print("[fallback-diag] assert-clean OK: model_action_illegal == 0", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
