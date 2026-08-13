"""Step-1 diagnosis probe: WHERE does mcts_il_agent's local decision-making
actually differ from its own prior?

Instruments the live search without editing agent code (monkeypatch only), then
plays real cabt games. Answers, per decision:

  forced_fastpath        - answered by the min_count>=1 & n_opts==max_count shortcut
  searched               - search actually ran
  changed_vs_prior       - most-visited child != highest-prior child (search MOVED the pick)
  terminal_seen          - >=1 node in the tree was a finished game (the report's
                           claimed mechanism: "terminal-node lookahead")
  uniform_prior          - evaluator silently returned a UNIFORM prior (encode
                           failure / NaN) -- invisible to diag_snapshot()
  submax_gap             - a legal selection size strictly between 1 and maxCount
                           existed that the search's action space cannot express
  combo_capped           - enumerate_actions hit the 64-combination cap

Usage: uv run python probe_mcts_mechanism.py --opponent il_agent --pairs 2
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

REPO = Path.cwd().resolve()
while not (REPO / "scripts" / "benchmark_agents.py").exists():
    REPO = REPO.parent
    if REPO == REPO.parent:
        raise SystemExit("repo root not found (run from the repo/worktree root)")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("PTCG_DEVICE", "cpu")
os.environ["PTCG_FALLBACK_TRACK"] = "1"

import benchmark_agents as BA  # noqa: E402

STATS: collections.Counter = collections.Counter()
NODES_PER_DECISION: list[int] = []
CHILDREN_PER_DECISION: list[int] = []
DECISION_MS: list[float] = []
MAXDEPTH: list[int] = []
LEAF_VALUES: list[float] = []


def install_probe():
    from pokemon_tcg import search_prior_mcts as SPM

    orig_create = SPM._create_node
    orig_eval = SPM.ILPriorEvaluator.evaluate
    orig_choose = SPM.mcts_choose

    ctx = {"root": None, "nodes": 0, "terminal": 0, "uniform": 0}

    def create_node(parent, search_state, your_index, evaluator):
        node = orig_create(parent, search_state, your_index, evaluator)
        ctx["nodes"] += 1
        d, p = 0, parent
        while p is not None:
            d += 1
            p = p.parent
        ctx["maxdepth"] = max(ctx.get("maxdepth", 0), d)
        if search_state.observation.current.result >= 0:
            ctx["terminal"] += 1
        if parent is None:
            ctx["root"] = node
        return node

    def evaluate(self, obs_dict, actions, want_value=True):
        # Directly separate the two silent-degradation causes from a
        # legitimately flat prior. encode_observation returning None and NaN
        # logits both yield (0.0, uniform) with no counter anywhere.
        from pokemon_tcg.il_dataset import encode_observation as _enc
        import torch as _t

        cause = None
        try:
            f = _enc(obs_dict)
            if f is None:
                cause = "encode_none"
            else:
                nr = f["n_real_options"]
                bt = {k: v.unsqueeze(0) for k, v in f.items() if k != "n_real_options"}
                lg = self.policy(**bt)["logits"][0]
                if bool(_t.isnan(lg[: nr + 1]).any()):
                    cause = "nan_logits"
        except Exception as e:  # the probe must not change agent behaviour
            cause = f"probe_enc_exc:{type(e).__name__}"
        if cause:
            STATS[f"degraded:{cause}"] += 1
            ctx["uniform"] += 1
        value, priors = orig_eval(self, obs_dict, actions, want_value=want_value)
        if len(actions) >= 2 and len(set(round(p, 9) for p in priors)) == 1:
            STATS["flat_prior_ge2_actions"] += 1
        STATS["eval_calls"] += 1
        # Stage-3 centering check, measured in vivo rather than on eval rows:
        # the mean leaf value over the states search ACTUALLY visits is the
        # bias that the turn-parity sign flip turns into a systematic
        # preference. It must be ~0.
        # 2026-08-12 evaluator fix: opponent-view nodes no longer consult the
        # critic (want_value=False -> value None, inherited value computed in
        # _create_node instead) -- count them separately so the critic-leaf
        # distribution stays comparable to the pre-fix probe.
        if value is None:
            STATS["value_skipped_opponent_view"] += 1
        else:
            LEAF_VALUES.append(float(value))
        return value, priors

    def choose(obs_dict, my_deck, evaluator, search_count=30, rng=None):
        STATS["decisions"] += 1
        sel = obs_dict.get("select") or {}
        opts = sel.get("option") or []
        n_opts = len(opts)
        max_count = sel.get("maxCount") if sel.get("maxCount") is not None else 1
        min_count = sel.get("minCount") or 0
        STATS[f"nopts_ge48:{n_opts >= 48}"] += 0  # keep key stable
        if n_opts >= 48:
            STATS["nopts_ge48"] += 1
        # legal sizes are min_count..max_count; search enumerates only
        # size==max_count (plus [] when min_count==0)
        lo = max(min_count, 1)
        if max_count >= 2 and max_count > lo:
            STATS["submax_gap"] += 1
        if min_count >= 1 and n_opts == max_count:
            STATS["forced_fastpath"] += 1
            return orig_choose(obs_dict, my_deck, evaluator, search_count, rng)

        ctx.update(root=None, nodes=0, terminal=0, uniform=0, maxdepth=0)
        t0 = time.perf_counter()
        out = orig_choose(obs_dict, my_deck, evaluator, search_count, rng)
        DECISION_MS.append((time.perf_counter() - t0) * 1000.0)
        STATS["searched"] += 1
        NODES_PER_DECISION.append(ctx["nodes"])
        if ctx["terminal"]:
            STATS["terminal_seen"] += 1
        STATS["terminal_nodes_total"] += ctx["terminal"]
        if ctx["uniform"]:
            STATS["uniform_prior"] += 1
        root = ctx["root"]
        if root is not None and root.children:
            CHILDREN_PER_DECISION.append(len(root.children))
            if len(root.children) >= 64:
                STATS["combo_capped"] += 1
            best_prior = max(root.children, key=lambda ch: ch.prob)
            if list(best_prior.select) != list(out):
                STATS["changed_vs_prior"] += 1
                # was the overridden prior a near-tie, or a confident pick?
                chosen = next((c for c in root.children if list(c.select) == list(out)), None)
                if chosen is not None and best_prior.prob - chosen.prob > 0.10:
                    STATS["changed_vs_confident_prior"] += 1
            expanded = sum(1 for ch in root.children if ch.node is not None)
            STATS["root_children_expanded_total"] += expanded
            visits = sorted((c.node.visit for c in root.children if c.node), reverse=True)
            if len(visits) >= 2 and visits[0] - visits[1] <= 1:
                STATS["top2_visit_margin_le1"] += 1
            MAXDEPTH.append(ctx["maxdepth"])
        return out

    SPM._create_node = create_node
    SPM.ILPriorEvaluator.evaluate = evaluate
    SPM.mcts_choose = choose
    return SPM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="il_agent")
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--agent", default="mcts_il_agent",
                    help="search agent under probe; '<agent>@<deck-tag>' supported")
    ap.add_argument("--search-count", type=int, default=None,
                    help="MCTS_IL_SEARCH_COUNT for this run (N=0 is the no-search control)")
    ap.add_argument("--critic-dir", default=None, help="MCTS_IL_CRITIC_DIR override")
    ap.add_argument("--center-leaf", default=None, choices=["0", "1"],
                    help="MCTS_IL_CENTER_LEAF; 0 pins center=0.5 (uncentered control arm)")
    args = ap.parse_args()

    # Env must be set BEFORE load_agent: agent_core reads these at import.
    if args.search_count is not None:
        os.environ["MCTS_IL_SEARCH_COUNT"] = str(args.search_count)
    if args.critic_dir is not None:
        os.environ["MCTS_IL_CRITIC_DIR"] = args.critic_dir
    if args.center_leaf is not None:
        os.environ["MCTS_IL_CENTER_LEAF"] = args.center_leaf

    a = BA.load_agent(args.agent)
    # load_agent execs the file under a synthetic module name, so importing
    # agents.mcts_il_agent.agent_core here would give a DIFFERENT object.
    AC = BA._LOADED_MODULES[args.agent]

    SPM = install_probe()
    AC.mcts_choose = SPM.mcts_choose  # agent() resolves this as a module global
    AC.diag_reset()
    b = BA.load_agent(args.opponent)

    from kaggle_environments import make

    t0 = time.time()
    res = BA.play_match(a, b, lambda: make("cabt"), args.pairs,
                        name_a=args.agent, name_b=args.opponent)
    wall = time.time() - t0

    d = dict(STATS)
    d["agent"] = args.agent
    d["opponent"] = args.opponent
    d["pairs"] = args.pairs
    d["search_count"] = int(os.environ.get("MCTS_IL_SEARCH_COUNT", "30"))
    d["critic_dir"] = os.environ.get("MCTS_IL_CRITIC_DIR")
    d["model_dir"] = os.environ.get("MCTS_IL_MODEL_DIR")  # policy the wrapper serves
    d["center_leaf"] = os.environ.get("MCTS_IL_CENTER_LEAF", "1")
    if LEAF_VALUES:
        n = len(LEAF_VALUES)
        mean = sum(LEAF_VALUES) / n
        var = sum((v - mean) ** 2 for v in LEAF_VALUES) / n
        d["leaf_value"] = {
            "n": n, "mean": round(mean, 5), "std": round(var ** 0.5, 5),
            "frac_nonzero": round(sum(1 for v in LEAF_VALUES if v != 0.0) / n, 4),
        }
    # THE Stage-3 gate (b) number: of the decisions where search actually ran,
    # how often did the tree move the pick off the prior's argmax?
    if STATS.get("searched"):
        d["changed_vs_prior_rate_of_searched"] = round(
            STATS.get("changed_vs_prior", 0) / STATS["searched"], 4)
    if STATS.get("decisions"):
        d["changed_vs_prior_rate_of_all_decisions"] = round(
            STATS.get("changed_vs_prior", 0) / STATS["decisions"], 4)
    d["wall_sec"] = round(wall, 1)
    d["match_result"] = res if not hasattr(res, "_asdict") else res._asdict()
    d["nodes_per_searched_decision_mean"] = (
        round(sum(NODES_PER_DECISION) / len(NODES_PER_DECISION), 2) if NODES_PER_DECISION else None
    )
    d["maxdepth_mean"] = round(sum(MAXDEPTH)/len(MAXDEPTH),2) if MAXDEPTH else None
    d["maxdepth_max"] = max(MAXDEPTH) if MAXDEPTH else None
    d["root_children_mean"] = (
        round(sum(CHILDREN_PER_DECISION) / len(CHILDREN_PER_DECISION), 2)
        if CHILDREN_PER_DECISION
        else None
    )
    if DECISION_MS:
        s = sorted(DECISION_MS)
        d["ms_mean"] = round(sum(s) / len(s), 1)
        d["ms_p95"] = round(s[int(0.95 * (len(s) - 1))], 1)
        d["ms_max"] = round(s[-1], 1)
    d["agent_diag"] = AC.diag_snapshot()
    print(json.dumps(d, indent=1, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=1, default=str))


if __name__ == "__main__":
    main()
