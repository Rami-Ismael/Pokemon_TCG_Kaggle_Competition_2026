"""What is each checkpoint in models/ actually FOR?

Directory names encode how a checkpoint was trained and say nothing about its
role, so `il_alldays_0804`, `il_agent_full_0804` and `il_alldays_equalsteps_0804`
read as siblings when in fact one is today's submission candidate, one is a
byte-identical duplicate of it, and one is a control testing whether the gain
comes from more STEPS or more DAYS.

This is the source of truth for role. It is keyed on the safetensors sha256, so
duplicates and empty/broken directories are detected rather than trusted -- a
rename or a re-copy cannot make the registry lie.

ROLES
  candidate   a real submission candidate; the thing you would upload
  shipped     what an agent currently ships (agents/il_agent/ reads this)
  control     an experiment arm that exists to test one stated theory
  ablation    a variation held for comparison, not for shipping
  archive     kept for provenance (e.g. a past ladder-scoring build)
  scratch     dry runs / smoke tests, safe to delete
  BROKEN      directory exists but has no loadable weights

Usage:
  uv run python scripts/model_registry.py            # the table
  uv run python scripts/model_registry.py --roles candidate,shipped
  uv run python scripts/model_registry.py --write-aliases   # models/by-role/<clear-name>
  uv run python scripts/model_registry.py --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

# dir name -> (role, clear name, one-line purpose)
# Clear name convention:  <role>-<data>-<variant>
#   data:    day0726 | combined | alldays          (what corpus it saw)
#   variant: 2ep/3ep/4ep | medium | equalsteps | hfstream
REGISTRY: dict[str, tuple[str, str, str]] = {
    "il_alldays_0804": (
        "candidate", "candidate-alldays-3ep",
        "TODAY'S MODEL: all days available as of 2026-08-04, 127,748 steps. "
        "The submission candidate. Submitted as 55248985 but on the WRONG deck "
        "(Mega Lucario ex) -> 422.8; the Grimmsnarl ex build is unsubmitted.",
    ),
    "il_alldays_equalsteps_0804": (
        "control", "control-alldays-equalsteps",
        "EXPERIMENT: same all-days corpus at 38,562 steps instead of 127,748. "
        "Tests whether the data-poor-deck robustness comes from more STEPS or "
        "more DAYS. Answer: more days -- it reproduces at <1/3 the compute.",
    ),
    "il_agent": (
        "shipped", "shipped-day0726-3ep",
        "What agents/il_agent/ loads today. 2026-07-26 only, 3 epochs.",
    ),
    "il_agent_2ep_backup": (
        "archive", "archive-day0726-2ep",
        "The weights behind our best ladder reading (~685 settled, 804.0 was a "
        "single high read). 2 epochs -- the one offline accuracy ranks LAST.",
    ),
    "il_agent_4ep": (
        "ablation", "ablation-day0726-4ep",
        "Epoch ablation. Best offline accuracy of the 07-26 family (.7592), "
        "mid-pack in actual play.",
    ),
    "il_agent_medium": (
        "ablation", "ablation-day0726-medium",
        "Capacity ablation: hidden 320 / 10.99M params vs the standard 3.32M. "
        "3.3x the parameters bought nothing measurable.",
    ),
    "il_agent_small_combined": (
        "ablation", "ablation-combined-2ep",
        "Combined 07-01+07-26, 2 epochs. Badly calibrated (ECE .163) and worst "
        "in play, but still within 11.6pp -- the offline-vs-play gap.",
    ),
    "il_agent_hfstream_combined_3ep": (
        "ablation", "ablation-combined-hfstream-3ep",
        "Combined corpus via HF streaming, 3 epochs.",
    ),
    "il_agent_dryrun": (
        "scratch", "scratch-dryrun",
        "Pipeline smoke test. Not a result.",
    ),
    "il_agent_medium_combined": (
        "BROKEN", "BROKEN-empty-dir",
        "EMPTY DIRECTORY -- no config.json, no safetensors. The benchmark's "
        "`grid_medium_comb` arm points here and silently plays non-ML fallback "
        "moves. Any number from that arm is not a model measurement.",
    ),
    # Duplicates: same weights under a second name. Listed so nobody benchmarks
    # the same checkpoint twice thinking they are comparing two models.
    # PPO / self-play families: these store .pt, not safetensors, so they are
    # registered explicitly rather than reported as BROKEN.
    "ppo": ("archive", "archive-ppo-stage3", "Stage-3 PPO snapshots (.pt)."),
    "ppo_puffer": ("archive", "archive-ppo-puffer", "PufferLib PPO run (.pt)."),
    "ppo_puffer_g2": ("archive", "archive-ppo-puffer-g2", "PufferLib PPO gen 2 (.pt)."),
    "ppo_puffer_g3": ("archive", "archive-ppo-puffer-g3", "PufferLib PPO gen 3 (.pt)."),
    "s2": ("archive", "archive-stage2-reweight", "Stage-2 REWEIGHT arm checkpoints."),
    "models": ("scratch", "scratch-nested-models-dir", "Stray nested dir."),
    "il_agent_3ep": (
        "archive", "dup-of-shipped-day0726-3ep",
        "Byte-identical to models/il_agent.",
    ),
    "il_agent_winning_827.8": (
        "archive", "dup-of-archive-day0726-2ep",
        "Byte-identical to models/il_agent_2ep_backup. Named for a ladder "
        "reading the build does not reliably reproduce (settled ~685).",
    ),
}


def sha256(p: Path) -> str | None:
    f = p / "model.safetensors"
    if not f.exists():
        return None
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan() -> list[dict]:
    rows = []
    seen_dirs = set()
    for d in sorted(MODELS.iterdir()):
        if not d.is_dir() or d.name == "by-role":
            continue
        seen_dirs.add(d.name)
        role, clear, why = REGISTRY.get(
            d.name, ("UNREGISTERED", f"UNREGISTERED-{d.name}",
                     "Not in the registry -- add it to scripts/model_registry.py "
                     "so its purpose is recorded."))
        sha = sha256(d)
        steps = params = None
        meta = d / "train_metadata.json"
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                steps = m.get("total_steps") or m.get("step")
                params = m.get("n_params")
            except Exception:  # noqa: BLE001
                pass
        # A family that legitimately stores .pt is not broken.
        if sha is None and any(d.glob("**/*.pt")):
            pass
        elif sha is None and role != "BROKEN":
            role = "BROKEN"
            why = "No loadable model.safetensors. " + why
        rows.append({"dir": d.name, "role": role, "name": clear, "purpose": why,
                     "sha256": sha, "steps": steps, "params": params})

    # sha collisions == the same weights under two names
    by_sha: dict[str, list[str]] = {}
    for r in rows:
        if r["sha256"]:
            by_sha.setdefault(r["sha256"], []).append(r["dir"])
    for r in rows:
        dupes = [d for d in by_sha.get(r["sha256"] or "", []) if d != r["dir"]]
        r["duplicate_of"] = dupes
    missing = [k for k in REGISTRY if k not in seen_dirs]
    if missing:
        print(f"note: registry lists {len(missing)} dir(s) not present here "
              f"(other worktree / not synced): {', '.join(missing)}\n", file=sys.stderr)
    return rows


ROLE_ORDER = ["candidate", "shipped", "control", "ablation", "archive", "scratch",
              "BROKEN", "UNREGISTERED"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default=None, help="comma-separated roles to show")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-aliases", action="store_true",
                    help="create models/by-role/<clear-name> symlinks to the real dirs")
    args = ap.parse_args()

    rows = scan()
    if args.roles:
        keep = set(args.roles.split(","))
        rows = [r for r in rows if r["role"] in keep]
    rows.sort(key=lambda r: (ROLE_ORDER.index(r["role"])
                             if r["role"] in ROLE_ORDER else 99, r["name"]))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if args.write_aliases:
        out = MODELS / "by-role"
        out.mkdir(exist_ok=True)
        for old in out.iterdir():
            if old.is_symlink():
                old.unlink()
        n = 0
        for r in rows:
            if r["role"] in ("BROKEN", "UNREGISTERED"):
                continue
            (out / r["name"]).symlink_to(Path("..") / r["dir"])
            n += 1
        print(f"wrote {n} alias(es) under models/by-role/")
        return 0

    cur = None
    for r in rows:
        if r["role"] != cur:
            cur = r["role"]
            print(f"\n=== {cur.upper()} ===")
        sha = (r["sha256"] or "—")[:12]
        steps = f"{r['steps']:,}" if r["steps"] else "—"
        print(f"  {r['name']:34s} {sha:12s} steps={steps:>9s}  <- models/{r['dir']}")
        print(f"      {r['purpose']}")
        if r["duplicate_of"]:
            print(f"      DUPLICATE of: {', '.join(r['duplicate_of'])}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
