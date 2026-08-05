#!/usr/bin/env bash
# agent_core_improved against a WIDE, diverse local pool, in two configs.
#
#   heuristic_only   USE_SEARCH=0  -> prior is unreachable code (line 895 returns
#                                     before the prior lookup). This is the pure
#                                     AdvancedPolicy heuristic; the all-days
#                                     checkpoint is loaded and never queried.
#                                     Requested as "search off + all-days prior";
#                                     naming it honestly here.
#   search_alldays   USE_SEARCH=1  -> UCB1 re-ranker live, PUCT-biased by the
#                                     all-days BC policy. The config that
#                                     actually uses the neural net.
#
# Pool spans our agents, the random floor, and 5 external/public agents, so this
# is a heterogeneity check rather than a two-opponent slice.
#
# POWER WARNING: 100 games/cell => Wilson half-width ~10 pts near 50%. The
# 2026-08-04 ablation measured a 12-pt noise floor between two byte-identical
# configs. Treat gaps under ~15 pts as unresolved; this run is for finding LARGE
# differences across a diverse pool, not for fine ranking.
set -uo pipefail
cd "$(dirname "$0")/.."

PAIRS="${PAIRS:-50}"    # 50 mirrored pairs = 100 games per cell
OUT=reports/pool_check_search_configs.json

PTCG_DEVICE=cpu PTCG_FALLBACK_TRACK=1 nice -n 10 uv run python - "$PAIRS" "$OUT" <<'PY'
import json, os, subprocess, sys
from pathlib import Path

pairs, out_path = int(sys.argv[1]), sys.argv[2]
REPO = Path.cwd()
ALLDAYS = REPO / "models" / "il_alldays_0804"

OPPONENTS = [
    "random_legal", "rule_baseline", "improved_prob_main", "il_agent",
    "kiyotah_dragapult", "dedquoc_rule_engine", "ryotasueyoshi_alakazam",
    "mechi22_alakazam", "makthanithin_improved_prob",
]
CONFIGS = {
    "heuristic_only": {"USE_SEARCH": "0", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ALLDAYS)},
    "search_alldays": {"USE_SEARCH": "1", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ALLDAYS)},
}

# Fresh subprocess per cell: agent_core_improved reads the switches at import
# time and bc_prior caches the model at module level.
WORKER = r'''
import json, sys, time
sys.path.insert(0, "scripts")
import torch; torch.set_num_threads(1)
import benchmark_agents as B
from kaggle_environments import make
pairs, opp = int(sys.argv[1]), sys.argv[2]
t0 = time.time()
w, l, d, _ = B.play_match(B.load_agent("agent_core_improved"),
                          B.load_agent(opp), lambda: make("cabt"), pairs)
print("RESULT " + json.dumps({"w": w, "l": l, "d": d, "secs": round(time.time()-t0, 1)}))
'''

def wilson(w, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = w / n; dd = 1 + z*z/n
    c = (p + z*z/(2*n)) / dd
    h = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / dd
    return (c - h, c + h)

results = {}
for cfg, extra in CONFIGS.items():
    print(f"--- {cfg} ---", flush=True)
    for opp in OPPONENTS:
        env = dict(os.environ); env.update(extra)
        p = subprocess.run([sys.executable, "-c", WORKER, str(pairs), opp],
                           capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"  {opp:26} FAILED: {p.stderr.strip()[-300:]}", flush=True)
            continue
        r = json.loads(line[7:]); n = r["w"] + r["l"] + r["d"]
        lo, hi = wilson(r["w"], n)
        results[f"{cfg}|{opp}"] = {"config": cfg, "opponent": opp, **r, "games": n,
                                   "win_rate": r["w"]/n if n else None, "wilson95": [lo, hi]}
        print(f"  {opp:26} {r['w']:3}W-{r['l']:3}L of {n:3}  "
              f"wr={r['w']/n:.3f} [{lo:.3f},{hi:.3f}]  [{r['secs']:.0f}s]", flush=True)

Path(out_path).write_text(json.dumps(results, indent=2))
print(f"saved: {out_path}")
PY
