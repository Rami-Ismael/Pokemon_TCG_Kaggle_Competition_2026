"""Split agents/ into a TRAIN league and a HELD-OUT evaluation field.

Why this exists
---------------
Every self-play run so far (run_selfplay_g{1,2,3}.sh) trained against a league
of four of OUR OWN checkpoints, drew external opponents 0% of the time
(--mix default `0.625,0.375,0`), and played one deck on both seats (no
--deck-pool -> Mega Lucario ex, 2.6% of the corpus meta). It was then scored on
a pool that overlapped that training distribution. There was no held-out
anything, so "RL made the agent worse" was never separable from "RL specialized
into a room the evaluation also lived in".

This script fixes the missing piece: a FROZEN, disjoint partition of the pool
that every RL arm shares, so a win on the field is a claim about generalization
rather than about memorizing its own sparring partners.

Two axes, split separately and labelled, because they answer different
questions and previous work conflated them:

  policy novelty  an opponent POLICY the learner never trained against, playing
                  an archetype it did train against. This is the ladder's own
                  shape -- you meet unseen code piloting the same meta decks --
                  so it is the PRIMARY metric.
  deck novelty    an ARCHETYPE absent from the training deck pool entirely.
                  Harder and rarer on the ladder; reported separately, never
                  averaged into the primary number.

THE PARTITION IS BY PROVENANCE GROUP, NOT BY AGENT
--------------------------------------------------
Splitting agent-by-agent silently leaks. notebooks/reference/tombombadyl/
INDEX.md records that all ten `tb_*` agents are one competitor's work sharing a
single vendored package (agents/tb_shared/agent/, 32 .py files), and that
tb_dragapult / tb_iono / tb_abomasnow / tb_lucario / tb_alakazam are ports of
the same upstream sources as our kiyotah_* and ryotasueyoshi_* agents. Training
against kiyotah_dragapult and then "testing" on tb_dragapult measures memory,
not generalization. So agents are grouped by shared authorship or documented
shared upstream, and whole GROUPS are dealt to a side.

Split rules (deterministic; no RNG, so the partition is reproducible and
reviewable rather than merely seeded):

  1. Only role=="opponent" agents are eligible (scripts/agent_registry.py).
     Our own line, arms, baselines and diagnostics are never field members --
     beating ourselves is not evidence.
  2. `wmh_*` are OUR heuristic pilots cloning a ladder deck: real DECK
     diversity, our own policy. They may serve as training opponents and as
     deck-novelty probes, but never count toward the primary policy-novelty
     metric.
  3. TOMBOMBADYL_GROUP goes to HELD-OUT whole. It is one real competitor's
     complete portfolio -- ten brain x deck submissions, most carrying his own
     published ladder scores -- which makes it the best available stand-in for
     the unseen opponents the ladder actually serves. Its documented upstream
     twins (kiyotah_*, ryotasueyoshi_*) follow it across, so no code spans the
     boundary.
  4. Every other provenance group starts in TRAIN.
  5. META REPAIR: for any archetype whose external pilots are then ALL in
     TRAIN, the smallest remaining group piloting it moves to HELD-OUT (ties
     broken alphabetically), provided TRAIN keeps at least one pilot of that
     archetype. Without this the field cannot measure the archetype that
     matters most -- Marnie's Grimmsnarl ex is 37.2% of the corpus meta, and
     rule 3 alone leaves it entirely on the training side.
  6. Archetypes whose only pilots are `wmh_*` clones carry no external policy,
     so reserving them costs the training league nothing: they become the
     deck-novelty holdout and are excluded from the training deck pool.

Usage:
    uv run python scripts/build_field_split.py            # print + write
    uv run python scripts/build_field_split.py --dry-run  # print only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pokemon_tcg import config  # noqa: E402

OUT_PATH = config.CONFIGS_DIR / "field_split.json"
GLICKO_PATH = config.REPORTS_DIR / "glicko_ratings.json"

CLONE_PREFIX = "wmh_"

# Provenance is a fact about where code came from, not something derivable from
# a directory name, so it is declared here with its source. Anything not listed
# is its own group, keyed by the author prefix before the first underscore.
#
# tombombadyl: agents/tb_* all import agents/tb_shared/agent/ -- literally one
# codebase. kiyotah_* and ryotasueyoshi_alakazam join the group because
# notebooks/reference/tombombadyl/INDEX.md names them as the upstream his
# tb_dragapult / tb_iono / tb_abomasnow / tb_lucario / tb_alakazam were ported
# from ("Overlaps upstream with our kiyotah_dragapult", "Same upstream as our
# ryotasueyoshi_alakazam"). Keeping them together is what makes "held-out"
# true.
PROVENANCE_GROUPS: dict[str, str] = {
    **{f"tb_{n}": "tombombadyl" for n in (
        "abomasnow", "alakazam", "archaludon", "dragapult", "heuristic",
        "iono", "lucario", "rulecore", "search", "starmie")},
    "kiyotah_abomasnow": "tombombadyl",
    "kiyotah_dragapult": "tombombadyl",
    "kiyotah_iono": "tombombadyl",
    "ryotasueyoshi_alakazam": "tombombadyl",
}
HELD_OUT_GROUP = "tombombadyl"  # rule 3


def group_of(agent_dir: str) -> str:
    if agent_dir in PROVENANCE_GROUPS:
        return PROVENANCE_GROUPS[agent_dir]
    return agent_dir.split("_", 1)[0]


def load_registry() -> list[dict]:
    """agent_registry.py --json, run as a subprocess.

    Importing it would be cheaper, but it mutates sys.path to vendor cg-lib and
    warns that benchmark_agents does the same; a subprocess keeps that
    contamination out of this process.
    """
    out = subprocess.run(
        ["uv", "run", "python", str(REPO / "scripts" / "agent_registry.py"), "--json"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout
    # uv prints build chatter to stdout before the payload on a cold cache.
    return json.loads(out[out.index("["):])


def load_glicko() -> dict[str, float]:
    if not GLICKO_PATH.exists():
        return {}
    raw = json.loads(GLICKO_PATH.read_text())
    raw = raw.get("ratings", raw)
    return {k: (v["rating"] if isinstance(v, dict) else v) for k, v in raw.items()}


def build_split() -> dict:
    registry = [r for r in load_registry() if r["role"] == "opponent"]
    glicko = load_glicko()

    def rating(entry: dict) -> float | None:
        for n in entry.get("bench_names") or []:
            if n in glicko:
                return glicko[n]
        return glicko.get(entry["dir"])

    agents = [{"dir": r["dir"], "archetype": r["archetype"], "glicko": rating(r),
               "clone": r["dir"].startswith(CLONE_PREFIX),
               "group": group_of(r["dir"]),
               "bench_names": r.get("bench_names") or []}
              for r in registry]

    externals = [a for a in agents if not a["clone"]]
    clones = [a for a in agents if a["clone"]]

    groups: dict[str, list[dict]] = defaultdict(list)
    for a in externals:
        groups[a["group"]].append(a)

    # Rules 3 + 4.
    side = {g: ("held_out" if g == HELD_OUT_GROUP else "train") for g in groups}

    # Rule 5: meta repair. An archetype the field cannot see is an archetype we
    # cannot make a generalization claim about.
    arch_groups: dict[str, set[str]] = defaultdict(set)
    for a in externals:
        arch_groups[a["archetype"]].add(a["group"])
    repairs: list[dict] = []
    for arch in sorted(arch_groups):
        gs = arch_groups[arch]
        if any(side[g] == "held_out" for g in gs) or len(gs) < 2:
            continue
        movable = sorted(gs, key=lambda g: (len(groups[g]), g))
        for g in movable:
            # Never strand the archetype: TRAIN must retain a pilot of it.
            others = [a for a in externals
                      if a["archetype"] == arch and a["group"] != g
                      and side[a["group"]] == "train"]
            if others:
                side[g] = "held_out"
                repairs.append({"archetype": arch, "group": g,
                                "moved": sorted(a["dir"] for a in groups[g])})
                break

    train_archetypes = {a["archetype"] for a in externals if side[a["group"]] == "train"}
    train: list[dict] = []
    held: list[dict] = []
    for a in externals:
        if side[a["group"]] == "train":
            train.append({**a, "tier": "train"})
        else:
            held.append({**a, "tier": "policy_novelty"})

    # Rule 2 + 6: clones follow their archetype.
    heldout_archetypes: set[str] = set()
    for a in clones:
        if a["archetype"] in train_archetypes:
            train.append({**a, "tier": "train"})
        else:
            heldout_archetypes.add(a["archetype"])
            held.append({**a, "tier": "deck_novelty"})

    return {
        "_doc": "Frozen TRAIN/HELD-OUT partition of agents/ for RL evaluation. "
                "Built by scripts/build_field_split.py; see its docstring for "
                "the rules. Do not hand-edit -- regenerate.",
        "clone_prefix": CLONE_PREFIX,
        "held_out_group": HELD_OUT_GROUP,
        "meta_repairs": repairs,
        "train_archetypes": sorted(train_archetypes),
        "heldout_archetypes": sorted(heldout_archetypes),
        "train": sorted(train, key=lambda m: (m["archetype"], m["dir"])),
        "held_out": sorted(held, key=lambda m: (m["tier"], m["archetype"], m["dir"])),
    }


def report(split: dict) -> None:
    def line(m: dict) -> str:
        g = f"{m['glicko']:8.1f}" if m["glicko"] is not None else "  UNRATED"
        tag = " [clone]" if m["clone"] else f"  ({m['group']})"
        return f"    {g}  {m['dir']}{tag}"

    for bucket, title in (("train", "TRAIN LEAGUE (RL may train against these)"),
                          ("held_out", "HELD-OUT FIELD (evaluation only)")):
        members = split[bucket]
        print(f"=== {title} -- {len(members)} agents")
        by_arch: dict[str, list[dict]] = defaultdict(list)
        for m in members:
            by_arch[m["archetype"]].append(m)
        for arch in sorted(by_arch):
            tiers = "/".join(sorted({m["tier"] for m in by_arch[arch]}))
            print(f"  {arch}  ({len(by_arch[arch])})  {tiers}")
            for m in sorted(by_arch[arch], key=lambda m: -(m["glicko"] or -1)):
                print(line(m))
        print()

    if split["meta_repairs"]:
        print("meta repair (rule 5) moved to held-out so the field covers them:")
        for r in split["meta_repairs"]:
            print(f"  {r['archetype']}: {', '.join(r['moved'])}")
        print()

    pol = [m for m in split["held_out"] if m["tier"] == "policy_novelty"]
    deck = [m for m in split["held_out"] if m["tier"] == "deck_novelty"]
    rated = [m["glicko"] for m in pol if m["glicko"] is not None]
    tr = [m["glicko"] for m in split["train"]
          if m["glicko"] is not None and not m["clone"]]
    print(f"primary   policy-novelty field: {len(pol)} external agents over "
          f"{len({m['archetype'] for m in pol})} archetypes, "
          f"{len({m['group'] for m in pol})} provenance groups")
    print(f"secondary deck-novelty field:   {len(deck)} clone pilots over "
          f"{len(split['heldout_archetypes'])} archetypes absent from training")
    if rated and tr:
        print(f"balance   train externals mean Glicko {sum(tr) / len(tr):.0f} (n={len(tr)}) "
              f"vs field {sum(rated) / len(rated):.0f} (n={len(rated)}) -- close "
              f"means the split did not stack one side")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    split = build_split()
    report(split)

    t, h = {m["dir"] for m in split["train"]}, {m["dir"] for m in split["held_out"]}
    assert not (t & h), f"train/held-out overlap: {sorted(t & h)}"
    tg = {m["group"] for m in split["train"] if not m["clone"]}
    hg = {m["group"] for m in split["held_out"] if not m["clone"]}
    assert not (tg & hg), f"provenance group spans the split: {sorted(tg & hg)}"

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2) + "\n")
    print(f"\nwrote {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
