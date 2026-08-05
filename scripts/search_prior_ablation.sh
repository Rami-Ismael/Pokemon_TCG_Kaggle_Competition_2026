#!/usr/bin/env bash
# Does the BC prior actually earn its place inside search?
# Card: notes/experiments/2026-08-04-search-prior-ablation.md
#
# agent_core_improved has TWO switches that interact:
#   USE_SEARCH   default "0"  -> the UCB1 re-ranker (returns early if off)
#   USE_BC_PRIOR default "1"  -> PUCT-style prior, consulted ONLY inside search
# So by default the BC model loads and is never asked anything. This ablation
# turns search on and varies the prior, which has never been measured -- the
# prior has always run on the ORIGINAL checkpoint whenever it ran at all.
#
# Arms (agent_core_improved is the agent under test in all of them):
#   search_off_prior_original   reference = the shipped default
#   search_off_prior_alldays    NEGATIVE CONTROL: prior cannot matter when
#                               search is off, so this MUST match the reference.
#                               If it does not, the harness is lying.
#   search_on_no_prior          pure UCB1, no neural net at all
#   search_on_prior_original    the 804.0 configuration
#   search_on_prior_alldays     the new question: does a better prior help?
set -uo pipefail
cd "$(dirname "$0")/.."

PAIRS="${PAIRS:-25}"     # 25 mirrored pairs = 50 games per arm/opponent
OUT=reports/search_prior_ablation.json

PTCG_DEVICE=cpu PTCG_FALLBACK_TRACK=1 nice -n 10 uv run python - "$PAIRS" "$OUT" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path

pairs, out_path = int(sys.argv[1]), sys.argv[2]
REPO = Path.cwd()
ORIGINAL = REPO / "models" / "il_agent"
ALLDAYS  = REPO / "models" / "il_alldays_0804"
OPPONENTS = ["improved_prob_main", "rule_baseline"]

# arm -> env overrides. Each arm runs in a FRESH subprocess: agent_core_improved
# reads USE_SEARCH/USE_BC_PRIOR at import time and bc_prior caches the loaded
# model at module level, so toggling them in-process would silently reuse the
# first arm's configuration.
ARMS = {
    "search_off_prior_original": {"USE_SEARCH": "0", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ORIGINAL)},
    "search_off_prior_alldays":  {"USE_SEARCH": "0", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ALLDAYS)},
    "search_on_no_prior":        {"USE_SEARCH": "1", "USE_BC_PRIOR": "0", "IL_MODEL_DIR": str(ORIGINAL)},
    "search_on_prior_original":  {"USE_SEARCH": "1", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ORIGINAL)},
    "search_on_prior_alldays":   {"USE_SEARCH": "1", "USE_BC_PRIOR": "1", "IL_MODEL_DIR": str(ALLDAYS)},
}

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
for arm, extra in ARMS.items():
    for opp in OPPONENTS:
        env = dict(os.environ); env.update(extra)
        p = subprocess.run([sys.executable, "-c", WORKER, str(pairs), opp],
                           capture_output=True, text=True, env=env)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print(f"{arm:26} vs {opp:20} FAILED\n{p.stderr[-500:]}", flush=True)
            continue
        r = json.loads(line[7:])
        n = r["w"] + r["l"] + r["d"]
        lo, hi = wilson(r["w"], n)
        results[f"{arm}|{opp}"] = {"arm": arm, "opponent": opp, **r, "games": n,
                                   "win_rate": r["w"]/n if n else None,
                                   "wilson95": [lo, hi], **extra}
        print(f"{arm:26} vs {opp:20} {r['w']:3}W-{r['l']:3}L of {n:3}  "
              f"wr={r['w']/n:.3f} [{lo:.3f},{hi:.3f}]  [{r['secs']:.0f}s]", flush=True)

Path(out_path).write_text(json.dumps(results, indent=2))
print(f"saved: {out_path}")
PY
