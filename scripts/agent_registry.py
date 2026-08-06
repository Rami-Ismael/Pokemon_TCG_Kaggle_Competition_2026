"""What is each directory in agents/ actually FOR, and what deck does it pilot?

The models/ side of this question is scripts/model_registry.py; this is the
same idea for agents/. Two failures motivate it:

  1. Directory names say who wrote an agent, not what it is for. `il_agent`,
     `exploiter_regression` and `s2v2_arms/e0_s43` read as siblings when one is
     what we ship, one must NEVER be submitted, and one is an experiment arm.
  2. Directory names LIE about the deck. `prvsiyan_grimbelief_alakazam` and
     `prvsiyan_templates_alakazam` both ship a Grimmsnarl deck. So the archetype
     in an alias is always DERIVED from deck.csv here, never from the dir name.

NAMING RULE (same as model_registry.py)
  A directory name is an immutable ID -- the submission ledger, reports/ and
  notes/ cite it. Role is mutable, so it lives here and in the
  `--write-aliases` symlinks, never in the real dir name.

ROLES
  primary     our own line; the thing we would submit
  arm         an experiment arm -- exists to be compared, not shipped
  baseline    a deliberate floor or reference point of our own
  opponent    wired into the pool to be played AGAINST
  diagnostic  NEVER SUBMIT -- exists to expose a weakness
  family      a directory of arms, not an agent itself
  shared      library code vendored for other agents to import
  UNREGISTERED

Usage:
  uv run python scripts/agent_registry.py
  uv run python scripts/agent_registry.py --roles primary,diagnostic
  uv run python scripts/agent_registry.py --archetype grimmsnarl
  uv run python scripts/agent_registry.py --write-aliases   # agents/by-role/
  uv run python scripts/agent_registry.py --json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AGENTS = REPO / "agents"

# dir name (or "parent/child" for arms) -> role. Purpose is NOT declared here:
# it is read from the agent's own docstring, so this table cannot drift from
# what the code says about itself.
ROLES: dict[str, str] = {
    # --- ours ---------------------------------------------------------------
    "il_agent": "primary",
    "il_alldays_0804": "primary",
    "mcts_il_agent": "primary",
    "exploiter_regression": "diagnostic",
    "random_legal": "baseline",
    "grunt": "baseline",
    "improved_probabilistic": "baseline",
    "mega_lucario": "baseline",
    "selfplay_g1_final": "arm",
    "selfplay_g1_ref430k": "arm",
    "selfplay_g2_final": "arm",
    "selfplay_g3_final": "arm",
    # --- arm families: the container is not an agent; children are ----------
    "il_arms": "family",
    "s2_arms": "family",
    "s2v2_arms": "family",
    "ppo_arms": "family",
    "grid_cells": "family",
    "tb_shared": "shared",
    # --- wired pool opponents ----------------------------------------------
    # Ours, cloning a ladder deck:
    "wmh_alakazam": "opponent", "wmh_bellibolt": "opponent",
    "wmh_chandelure": "opponent", "wmh_froslass": "opponent",
    "wmh_garchomp": "opponent", "wmh_grimmsnarl": "opponent",
    "wmh_kangaskhan": "opponent", "wmh_megastarmie": "opponent",
    "wmh_mewtwo": "opponent", "wmh_ogerpon": "opponent",
    "wmh_trevenant": "opponent", "wmh_typhlosion": "opponent",
    # Vendored from other competitors' public notebooks / repos:
    "tb_abomasnow": "opponent", "tb_alakazam": "opponent",
    "tb_archaludon": "opponent", "tb_dragapult": "opponent",
    "tb_heuristic": "opponent", "tb_iono": "opponent",
    "tb_lucario": "opponent", "tb_rulecore": "opponent",
    "tb_search": "opponent", "tb_starmie": "opponent",
    "avikdas567_heuristic": "opponent", "biohack44_day2": "opponent",
    "daniilkrasnovvv_conservative_prob": "opponent",
    "dedquoc_rule_engine": "opponent", "jek1wantaufik_lucario": "opponent",
    "kiyotah_abomasnow": "opponent", "kiyotah_dragapult": "opponent",
    "kiyotah_iono": "opponent", "kojimar_lucario": "opponent",
    "makimakiai_rl": "opponent", "makthanithin_1084_baseline": "opponent",
    "makthanithin_improved_prob": "opponent", "mechi22_alakazam": "opponent",
    "pixiux_lucario_v63": "opponent", "plamen06_steel": "opponent",
    "pllinas_alakazam": "opponent", "prvsiyan_control_v11": "opponent",
    "prvsiyan_grimbelief_alakazam": "opponent",
    "prvsiyan_templates_alakazam": "opponent",
    "romanrozen_strong_start": "opponent",
    "ryotasueyoshi_alakazam": "opponent",
}
# Arm children inherit "arm" from their family rather than being listed here.
FAMILY_CHILD_ROLE = "arm"

ROLE_ORDER = ["primary", "arm", "baseline", "opponent", "diagnostic", "family",
              "shared", "UNREGISTERED"]
NO_ALIAS = {"UNREGISTERED", "family", "shared"}

# ---------------------------------------------------------------------------
# Deck identity: the ace / carry Pokemon.
#
# reports/deck_selection.md §1 and scripts/census_pool_archetypes.py rank by
# (evolution stage + ex/megaEx bonus, copy count). That key mislabels a deck
# whose carry is a BASIC ex running few copies behind a stage-1 support line:
# wmh_mewtwo is 4x Team Rocket's Spidops (stage 1, 130 HP) and 2x Team Rocket's
# Mewtwo ex (basic ex, 280 HP) -- both score 1, so copy count hands the label to
# Spidops. So this key inserts an ex-preference AHEAD of copy count.
#
# That is a deliberate divergence from the census script, which is left alone:
# its numbers back a published report and must stay comparable to it. These
# labels are names, and a name that points at the wrong card is worse than one
# that disagrees with a table.
#
# When two Pokemon still tie exactly, the deck genuinely has two aces and both
# are reported (tb_starmie is 3x Mega Froslass ex AND 3x Mega Starmie ex).
# ---------------------------------------------------------------------------
_CARD: dict | None = None


def _cards() -> dict:
    global _CARD
    if _CARD is None:
        for cand in (REPO / "data" / "external" / "cg-lib",
                     REPO / "agents" / "mega_lucario"):
            if (cand / "cg" / "api.py").exists():
                sys.path.insert(0, str(cand))
                break
        try:
            from cg.api import all_card_data
        except ImportError:
            print("warn: cg-lib not importable -- archetypes will read '?'. "
                  "Symlink data/external/cg-lib from the main checkout.",
                  file=sys.stderr)
            _CARD = {}
            return _CARD
        _CARD = {c.cardId: c for c in all_card_data()}
    return _CARD


def _is_pokemon(card) -> bool:
    return bool(card.basic or card.stage1 or card.stage2) or card.hp > 0


def ace_pokemon(deck_ids: list[int]) -> list[str]:
    """Every Pokemon tied for top ace rank -- usually one, two for a split deck."""
    cards = _cards()
    if not cards:
        return ["?"]
    ranked: dict[tuple[int, int, int], list[str]] = {}
    for cid, n in Counter(deck_ids).items():
        card = cards.get(cid)
        if card is None or not _is_pokemon(card):
            continue
        stage = 2 if card.stage2 else 1 if card.stage1 else 0
        bonus = 2 if card.megaEx else 1 if card.ex else 0
        key = (stage + bonus, 1 if (card.ex or card.megaEx) else 0, n)
        ranked.setdefault(key, []).append(card.name)
    if not ranked:
        return ["(none)"]
    return sorted(ranked[max(ranked)])


def slug(name: str) -> str:
    """'Mega Lucario ex' -> 'lucario'; the archetype word people actually say.

    Everything a card name carries is a prefix or a suffix around the Pokemon:
    a trainer possessive ("Marnie's Grimmsnarl ex"), a form ("Cornerstone Mask
    Ogerpon ex"), "Mega", and the trailing ex/V rank. So strip the possessive
    and the rank suffix and take the LAST word -- the deck is a Grimmsnarl deck,
    not a Marnie deck, and an Ogerpon deck, not a Cornerstone deck.
    """
    if name in ("?", "(none)"):
        return name.strip("()")
    stripped = re.sub(r"^.*['’]s\s+", "", name)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stripped or name) if w]
    rank = {"ex", "v", "vmax", "vstar", "gx"}
    core = [w for w in words if w.lower() not in rank]
    return (core[-1] if core else words[-1]).lower()


# ---------------------------------------------------------------------------


def bench_names() -> dict[str, list[str]]:
    """agents/ dir -> the name(s) benchmark_agents.py knows it by.

    A third naming layer: reports/glicko_ratings.json compounds history on the
    BENCHMARK name, not the directory name, and the two disagree (agents/
    mega_lucario is `rule_baseline` there). Read statically rather than by
    import -- benchmark_agents.py mutates sys.path on import.
    """
    src = REPO / "scripts" / "benchmark_agents.py"
    out: dict[str, list[str]] = {}
    if not src.exists():
        return out
    try:
        tree = ast.parse(src.read_text(errors="replace"))
    except SyntaxError:
        return out

    def parts(node) -> list[str] | None:
        """`REPO / "agents" / "x" / "agent_core.py"` -> ['agents', 'x', ...]."""
        got: list[str] = []
        while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if not isinstance(node.right, ast.Constant) or not isinstance(
                    node.right.value, str):
                return None
            got.insert(0, node.right.value)
            node = node.left
        return got if isinstance(node, ast.Name) and node.id == "REPO" else None

    seen: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "AGENT_FILES"
                   for t in node.targets):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            if not isinstance(k, ast.Constant):
                continue
            # A repeated key is silently resolved by Python to whichever came
            # last, so an edit to the earlier one would vanish without error.
            if k.value in seen:
                print(f"warn: benchmark_agents.py AGENT_FILES defines "
                      f"'{k.value}' twice (line {k.lineno}) -- Python keeps the "
                      f"last one silently", file=sys.stderr)
            seen.add(k.value)
            p = parts(v)
            if not (p and p[0] == "agents"):
                continue
            names = out.setdefault("/".join(p[1:-1]), [])
            if k.value not in names:
                names.append(k.value)
    return out


def _entry(d: Path) -> Path | None:
    for c in ("agent_core.py", "main.py"):
        if (d / c).exists():
            return d / c
    return None


def _purpose(src: Path | None) -> str:
    """First line of the module docstring -- what the code says about itself."""
    if src is None:
        return "no agent_core.py / main.py"
    try:
        doc = ast.get_docstring(ast.parse(src.read_text(errors="replace")))
    except SyntaxError:
        return "(unparseable)"
    if not doc:
        return "(no docstring)"
    return " ".join(doc.strip().splitlines()[0].split())


def _deck_path(d: Path, src: Path | None) -> Path | None:
    """This dir's deck.csv, or the one its wrapper borrows from another agent."""
    own = d / "deck.csv"
    if own.exists():
        return own
    if src is None:
        return None
    m = re.search(r'"agents"\s*/\s*"([^"]+)"\s*/\s*"deck\.csv"',
                  src.read_text(errors="replace"))
    if m:
        borrowed = AGENTS / m.group(1) / "deck.csv"
        if borrowed.exists():
            return borrowed
    return None


def scan() -> list[dict]:
    rows = []
    bench = bench_names()
    for d in sorted(AGENTS.iterdir()):
        if not d.is_dir() or d.name == "by-role":
            continue
        role = ROLES.get(d.name, "UNREGISTERED")
        children = []
        if role == "family":
            children = [c for c in sorted(d.iterdir())
                        if c.is_dir() and _entry(c) is not None]
        rows.append(_row(d, d.name, role, bench))
        for c in children:
            k = f"{d.name}/{c.name}"
            rows.append(_row(c, k, ROLES.get(k, FAMILY_CHILD_ROLE), bench))
    return rows


def _row(d: Path, key: str, role: str, bench: dict[str, list[str]]) -> dict:
    src = _entry(d)
    deck = _deck_path(d, src)
    ids = ([int(x) for x in deck.read_text().split() if x.strip()][:60]
           if deck else [])
    aces = ace_pokemon(ids) if ids else ["(no deck)"]
    arch = "+".join(slug(a) for a in aces) if ids else "nodeck"
    borrowed = bool(deck) and deck.parent != d
    return {
        "dir": key,
        "role": role,
        "name": f"{role}-{arch}-{key.replace('/', '__')}",
        "archetype": " + ".join(aces),
        "deck_from": deck.parent.name if borrowed else ("self" if deck else None),
        "n_cards": len(ids),
        "bench_names": sorted(bench.get(key, [])),
        "purpose": _purpose(src),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default=None, help="comma-separated roles to show")
    ap.add_argument("--archetype", default=None,
                    help="case-insensitive substring filter on the ace Pokemon")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-aliases", action="store_true",
                    help="create agents/by-role/<clear-name> symlinks")
    args = ap.parse_args()

    rows = scan()
    if args.roles:
        keep = set(args.roles.split(","))
        rows = [r for r in rows if r["role"] in keep]
    if args.archetype:
        needle = args.archetype.lower()
        rows = [r for r in rows if needle in r["archetype"].lower()]
    rows.sort(key=lambda r: (ROLE_ORDER.index(r["role"])
                             if r["role"] in ROLE_ORDER else 99, r["name"]))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if args.write_aliases:
        out = AGENTS / "by-role"
        out.mkdir(exist_ok=True)
        for old in out.iterdir():
            if old.is_symlink():
                old.unlink()
        n = 0
        for r in rows:
            if r["role"] in NO_ALIAS:
                continue
            # every link sits in agents/by-role/, so the hop up is always one
            # level regardless of how deep the target is.
            (out / r["name"]).symlink_to(Path("..") / r["dir"])
            n += 1
        print(f"wrote {n} alias(es) under agents/by-role/")
        return 0

    unreg = [r["dir"] for r in rows if r["role"] == "UNREGISTERED"]
    cur = None
    for r in rows:
        if r["role"] != cur:
            cur = r["role"]
            print(f"\n=== {cur.upper()} ===")
        deck = r["archetype"]
        if r["deck_from"] and r["deck_from"] != "self":
            deck += f" (deck from {r['deck_from']})"
        print(f"  {r['name']:52s} {deck}")
        print(f"      {r['purpose'][:150]}")
        # The benchmark's own name is what Glicko history compounds on, so
        # surface it wherever it is not just the directory name.
        odd = [b for b in r["bench_names"] if b != r["dir"].split("/")[-1]]
        if odd:
            print(f"      benchmark_agents.py calls this: {', '.join(odd)}")
    if unreg:
        print(f"\n{len(unreg)} UNREGISTERED: add a role in scripts/agent_registry.py")
    unwired = [r["dir"] for r in rows
               if r["role"] in ("primary", "opponent", "baseline", "arm")
               and not r["bench_names"]]
    if unwired:
        print(f"\n{len(unwired)} wired agent(s) with NO benchmark_agents.py entry "
              f"-- they cannot be played or rated:\n  " + "\n  ".join(unwired))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
