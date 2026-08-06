"""Smoke driver for the PTCG agent: launch real cabt battles + poke the model.

This is the run-skill harness (.claude/skills/run-ptcg-battle/SKILL.md).
It reuses the proven loading path (scripts/benchmark_agents.py::load_agent)
instead of reimplementing it -- a hand-rolled loader once cost a session an
hour of "every game ends after 2 steps" (silent deck-submission failure).

Subcommands
-----------
  doctor    verify the checkout can actually run battles; print exact fixes
  battle    play N real games between two named agents (default cpu)
  forward   direct invocation: encoder + checkpoint forward pass, no env

Run from the repo root (or any worktree root):

  uv run python .claude/skills/run-ptcg-battle/driver.py doctor
  uv run python .claude/skills/run-ptcg-battle/driver.py battle
  uv run python .claude/skills/run-ptcg-battle/driver.py forward
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "scripts"
MODEL_DIR = REPO / "models" / "il_agent"
EPISODES = REPO / "data" / "episodes"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "src"))


def _find_in_parents(rel: str) -> Path | None:
    """Find rel in the main checkout when this is a bare worktree (same walk
    benchmark_agents.py uses for cg-lib)."""
    for p in [REPO, *REPO.parents]:
        cand = p / rel
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------- doctor ----

def cmd_doctor(_args) -> int:
    ok = True

    def check(label: str, passed: bool, fix: str = "") -> None:
        nonlocal ok
        mark = "ok " if passed else "FAIL"
        print(f"[{mark}] {label}")
        if not passed:
            ok = ok and False
            if fix:
                print(f"       fix: {fix}")

    # cg engine (needed for ANY battle; git-tracked, so worktrees have it too)
    cg_dir = None
    for cand in [REPO / "data" / "external" / "cg-lib",
                 *[p / "data" / "external" / "cg-lib" for p in REPO.parents]]:
        if (cand / "cg" / "api.py").exists():
            cg_dir = cand
            break
    check(f"cg engine package ({cg_dir})", cg_dir is not None,
          "data/external/cg-lib is git-tracked; your checkout is broken -- "
          "git checkout data/external/cg-lib")
    if cg_dir:
        sys.path.insert(0, str(cg_dir))
        try:
            import cg.api  # noqa: F401
            check("import cg.api", True)
        except Exception as e:  # noqa: BLE001
            check(f"import cg.api ({type(e).__name__}: {e})", False,
                  "native lib mismatch? cg-lib ships libcg.dylib "
                  "(macOS arm64) + libcg-arm64.so + libcg.so")

    try:
        import kaggle_environments
        from kaggle_environments import make
        env = make("cabt")
        check(
            f"kaggle_environments {kaggle_environments.__version__} "
            "+ make('cabt')",
            env is not None,
        )
    except Exception as e:  # noqa: BLE001
        check(f"kaggle_environments make('cabt') ({type(e).__name__}: {e})",
              False,
              "uv sync  (kaggle-environments is a pyproject dependency)")

    try:
        from pokemon_tcg.device import resolve_device
        print(f"[ok ] torch device would resolve to: {resolve_device()} "
              f"(PTCG_DEVICE={os.environ.get('PTCG_DEVICE', '<unset>')})")
    except Exception as e:  # noqa: BLE001
        check(f"torch / pokemon_tcg import ({type(e).__name__}: {e})",
              False, "uv sync")

    # il_agent checkpoint: WITHOUT this the agent silently degrades to
    # _safe_choice on every decision -- battles still "work", results are junk.
    main_model = _find_in_parents("models/il_agent/model.safetensors")
    check(f"models/il_agent checkpoint ({MODEL_DIR})",
          (MODEL_DIR / "model.safetensors").exists(),
          f"ln -sfn {main_model.parent} {MODEL_DIR}" if main_model
          else "train one (scripts/train_il.py) or copy models/il_agent "
               "from the main checkout")

    # Every OTHER agent wrapper's checkpoint. Not battle-critical (so this
    # warns rather than failing the exit code), but it is what rots silently:
    # on 2026-08-05 ten registry agents pointed at checkpoint dirs that existed
    # in no checkout at all, doctor still printed all-green because it only
    # ever looked at models/il_agent, and the load-all-agents test was the
    # thing that finally noticed. A wrapper whose dir is missing either raises
    # on import (the loader refuses to fall back to the wrong model) or, where
    # the load is optional like mcts_il_agent's critic, degrades SILENTLY --
    # which is the worse of the two.
    wanted: dict[str, list[str]] = {}
    for wrapper in sorted((REPO / "agents").rglob("agent_core.py")):
        for m in re.finditer(r'_REPO\s*/\s*"models"((?:\s*/\s*"[^"]+")+)',
                             wrapper.read_text()):
            rel = "models/" + "/".join(re.findall(r'"([^"]+)"', m.group(1)))
            wanted.setdefault(rel, []).append(
                str(wrapper.parent.relative_to(REPO)))
    missing = {rel: users for rel, users in wanted.items()
               if not (REPO / rel).exists()}
    if missing:
        print(f"[warn] {len(missing)} of {len(wanted)} agent checkpoint dir(s) "
              f"missing -- battles with models/il_agent still work, but "
              f"benchmark_agents.py and the load-all-agents test will fail:")
        for rel, users in sorted(missing.items()):
            src = _find_in_parents(rel)
            print(f"       {rel}  (used by {', '.join(sorted(set(users)))})")
            print(f"       fix: " + (
                f"ln -sfn {src} {REPO / rel}" if src else
                "not in this checkout or its parents -- check sibling "
                "worktrees, or remove the wrapper from AGENT_FILES in "
                "scripts/benchmark_agents.py"))
    else:
        print(f"[ok ] all {len(wanted)} agent checkpoint dir(s) present")

    # episodes: needed by `forward`, replay_episode.py, and training --
    # NOT by `battle`
    main_ep = _find_in_parents("data/episodes/splits/splits.json")
    check(f"data/episodes (only needed for forward/replay/training) "
          f"({EPISODES})",
          (EPISODES / "splits" / "splits.json").exists(),
          f"ln -sfn {main_ep.parents[1]} {EPISODES}" if main_ep
          else "download via scripts/ingest_episodes.py or symlink from "
               "the main checkout")

    # advisory: MPS contention from concurrent training sessions
    try:
        out = subprocess.run(
            ["pgrep", "-fl", "train_il|train_ppo"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        if out:
            print("[note] training process(es) running -- keep smokes on "
                  "cpu (the default):")
            for line in out.splitlines()[:4]:
                print(f"       {line}")
    except OSError:
        pass

    print("\ndoctor:",
          "all battle-critical checks passed" if ok
          else "FAILING -- fix the lines above")
    return 0 if ok else 1


# ---------------------------------------------------------------- battle ----

def cmd_battle(args) -> int:
    # Set the device BEFORE any torch-importing module loads. cpu is the
    # default on purpose: it reproduces the CPU-only Kaggle evaluator and
    # stays off MPS, which concurrent training sessions may be using.
    if args.device != "auto":
        os.environ["PTCG_DEVICE"] = args.device

    from benchmark_agents import AGENT_FILES, load_agent  # noqa: E402
    from kaggle_environments import make  # noqa: E402

    for name in (args.a, args.b):
        base = name.split("@")[0]
        if base not in AGENT_FILES:
            print(f"unknown agent '{name}'. "
                  f"Known: {', '.join(sorted(AGENT_FILES))}")
            return 2

    t0 = time.monotonic()
    agent_a, agent_b = load_agent(args.a), load_agent(args.b)
    print(f"loaded {args.a} vs {args.b} in {time.monotonic() - t0:.1f}s "
          f"(PTCG_DEVICE={os.environ.get('PTCG_DEVICE', '<unset>')})")

    wins = losses = draws = errors = 0
    env = None
    for i in range(args.games):
        t = time.monotonic()
        env = make("cabt")
        trace = env.run([agent_a, agent_b])
        final = trace[-1]
        r0, r1 = final[0]["reward"], final[1]["reward"]
        status = final[0]["status"]
        print(f"game {i}: rewards=({r0},{r1}) status={status} "
              f"steps={len(trace)} time={time.monotonic() - t:.1f}s")
        if status != "DONE" or r0 is None or r1 is None:
            errors += 1
        elif r0 > r1:
            wins += 1
        elif r0 < r1:
            losses += 1
        else:
            draws += 1

    print(f"\n{args.a} vs {args.b}: {wins}W-{losses}L-{draws}D, "
          f"{errors} errored / {args.games} games")
    print("(smoke only -- for strength claims use "
          "scripts/benchmark_agents.py, then the leaderboard-check skill)")

    if args.save_json and env is not None:
        Path(args.save_json).write_text(json.dumps(env.toJSON()))
        print(f"last game record -> {args.save_json}")
    if args.save_html and env is not None:
        Path(args.save_html).write_text(env.render(mode="html"))
        print(f"last game replay player -> {args.save_html} "
              "(open in a browser)")

    return 1 if errors else 0


# --------------------------------------------------------------- forward ----

def cmd_forward(args) -> int:
    """Encoder + checkpoint forward pass on recorded decisions. No env, no
    agent wrapper -- this is the layer most model/encoder PRs actually touch.
    Runs on cpu (evaluator parity)."""
    os.environ.setdefault("PTCG_DEVICE", "cpu")
    import torch  # noqa: E402

    from pokemon_tcg.il_dataset import (  # noqa: E402
        encode_observation,
        iter_decisions,
        resolve_split_dir,
    )
    from pokemon_tcg.il_model import PTCGImitationPolicy  # noqa: E402

    model_dir = Path(os.environ.get("IL_MODEL_DIR") or MODEL_DIR)
    if not (model_dir / "model.safetensors").exists():
        print(f"no checkpoint at {model_dir} -- run doctor for the "
              "symlink fix")
        return 1
    # NOT data/episodes/local/*.json: those are single-POV captures (steps are
    # dicts with 'obs'), not the two-agent kaggle-environments format
    # iter_decisions parses -- feeding them in KeyErrors on step[0].
    try:
        split_dir = resolve_split_dir(args.split)
    except Exception as e:  # noqa: BLE001
        print(f"cannot resolve split '{args.split}' ({e}) -- run doctor "
              "for the symlink fix")
        return 1

    t0 = time.monotonic()
    model = PTCGImitationPolicy.from_pretrained(model_dir)
    model.eval()
    print(f"loaded checkpoint {model_dir} in {time.monotonic() - t0:.1f}s")

    n = match = 0
    with torch.no_grad():
        for obs_dict, chosen, exclude in iter_decisions(
            split_dir, max_episodes=1
        ):
            feats = encode_observation(obs_dict, exclude=exclude)
            if feats is None:
                continue
            n_real = feats.pop("n_real_options")
            batch = {k: v.unsqueeze(0) for k, v in feats.items()}
            logits = model(**batch)["logits"][0]
            sel = obs_dict.get("select") or {}
            add_decline = (sel.get("minCount") or 0) <= len(exclude)
            upto = n_real + (1 if add_decline else 0)
            pred = int(logits[:upto].argmax().item())
            match += int(pred == chosen)
            n += 1
            if n >= args.max_decisions:
                break
    if n == 0:
        print("no decisions decoded from the first episode -- "
              "encoder regression?")
        return 1
    print(f"forward pass OK: top-1 match {match}/{n} on first "
          f"{args.split}-split episode (plumbing smoke -- the real "
          "offline eval is scripts/eval_rung1.py)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor",
                   help="verify battle prerequisites, print exact fixes")

    b = sub.add_parser("battle",
                       help="play real cabt games between two named agents")
    b.add_argument("--a", default="il_agent",
                   help="player 0 (benchmark_agents name)")
    b.add_argument("--b", default="rule_baseline", help="player 1")
    b.add_argument("--games", type=int, default=1)
    b.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu",
                   help="cpu (default) = evaluator parity + safe during "
                        "training runs")
    b.add_argument("--save-json",
                   help="write last game's kaggle-environments record here")
    b.add_argument("--save-html",
                   help="write last game's replay player page here")

    f = sub.add_parser("forward",
                       help="encoder+checkpoint smoke on recorded decisions")
    f.add_argument("--max-decisions", type=int, default=20)
    f.add_argument("--split", default="eval",
                   help="episode split to read (default eval)")

    args = ap.parse_args()
    dispatch = {
        "doctor": cmd_doctor,
        "battle": cmd_battle,
        "forward": cmd_forward,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
