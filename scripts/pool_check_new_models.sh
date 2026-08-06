#!/usr/bin/env bash
# Pre-submit pool check (standing rule, 2026-08-04): benchmark candidates
# against the local pool BEFORE any Kaggle submission, at a usable sample size.
#
# Question here: did the bigger corpus make the model STRONGER against the
# pool, or only harder for our own exploiter to beat? Those came apart on the
# ladder (exploitability .955 -> .385, ladder still 450.6), so measure it.
#
# Models (all the same PTCGImitationPolicy arch, swapped via IL_MODEL_DIR so
# the inference code path is identical and only weights differ):
#   original     models/il_agent                  4,554 eps / 38,562 steps
#   equalsteps   il_alldays_equalsteps_0804      15,032 eps / 38,562 steps
#   alldays      il_alldays_0804                 15,032 eps / 127,748 steps
#
# Opponents: agent_core_improved (heuristic + UCB1 search, BC prior dormant
# unless USE_SEARCH=1), improved_prob_main (heuristic + flat root Monte-Carlo,
# NO ML at all), rule_baseline (plain heuristic floor).
set -uo pipefail
cd "$(dirname "$0")/.."

PAIRS="${PAIRS:-25}"   # 25 mirrored pairs = 50 games per matchup
WT=.claude/worktrees/rewrite-kaggle-pokemon-tcg-prompt-c69997
OUT=reports/pool_check_new_models.json

PTCG_DEVICE=cpu PTCG_FALLBACK_TRACK=1 nice -n 10 uv run python - "$PAIRS" "$WT" "$OUT" <<'PY'
import json, os, sys, time
from pathlib import Path

pairs, wt, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, "scripts")
import torch; torch.set_num_threads(1)
from kaggle_environments import make

REPO = Path.cwd()
MODELS = {
    "original":   REPO / "models" / "il_agent",
    "equalsteps": REPO / wt / "models" / "il_alldays_equalsteps_0804",
    "alldays":    REPO / wt / "models" / "il_alldays_0804",
}
OPPONENTS = ["agent_core_improved", "improved_prob_main", "rule_baseline"]


def wilson(w, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


results = {}
for mname, mdir in MODELS.items():
    if not mdir.is_dir():
        print(f"SKIP {mname}: {mdir} missing", flush=True)
        continue
    # Swap weights, then (re)import the agent fresh so it picks them up.
    os.environ["IL_MODEL_DIR"] = str(mdir)
    for mod in [m for m in list(sys.modules) if "agent_core" in m or "benchmark_agents" in m]:
        del sys.modules[mod]
    import benchmark_agents as B
    agent = B.load_agent("il_agent")

    for opp in OPPONENTS:
        t0 = time.time()
        w, l, d, _ = B.play_match(agent, B.load_agent(opp), lambda: make("cabt"), pairs)
        n = w + l + d
        lo, hi = wilson(w, n)
        results[f"{mname}|{opp}"] = {
            "model": mname, "opponent": opp, "wins": w, "losses": l, "draws": d,
            "games": n, "win_rate": w / n if n else None, "wilson95": [lo, hi],
        }
        print(f"{mname:11} vs {opp:20} {w:3}W-{l:3}L of {n:3}  "
              f"wr={w/n:.3f} [{lo:.3f},{hi:.3f}]  [{time.time()-t0:.0f}s]", flush=True)

Path(out_path).write_text(json.dumps(results, indent=2))
print(f"saved: {out_path}")
PY
