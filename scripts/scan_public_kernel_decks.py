"""Identify the DECK of each top public Kaggle agent, by reading the deck itself.

Why not keyword search: agent names do not track archetypes. Two agents wired as
`prvsiyan_*_alakazam` actually pilot Marnie's Grimmsnarl ex, and `wmh_mewtwo`
pilots Team Rocket's Spidops (scripts/census_pool_archetypes.py). Searching the
Code tab for "grimmsnarl" therefore misses agents and returns analysis notebooks.
This pulls the top kernels and identifies each deck from the card database, the
same ace-Pokemon key as reports/deck_selection.md.

Deck recovery, in order: an explicit 60-int list literal in the source; a
base64 tarball payload containing a deck.csv; a plain deck.csv output.

Downloads only; nothing is executed. Wiring an agent is a separate, manual step
that still requires the capability audit and the legality probe.

Usage:
    uv run python scripts/scan_public_kernel_decks.py --top 30 --want "Grimmsnarl"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
for _c in (REPO / "data" / "external" / "cg-lib", REPO / "agents" / "mega_lucario"):
    if (_c / "cg" / "api.py").exists():
        sys.path.insert(0, str(_c))
        break

from census_pool_archetypes import ace_pokemon  # noqa: E402

DECK_SIZE = 60
INT_LIST = re.compile(r"\[\s*(\d{1,4}\s*,\s*){55,79}\d{1,4}\s*,?\s*\]")
B64 = re.compile(r"['\"]([A-Za-z0-9+/]{2000,}={0,2})['\"]")


def _kaggle(*args: str) -> str:
    r = subprocess.run(["uv", "run", "kaggle", *args], cwd=REPO,
                       capture_output=True, text=True, timeout=180)
    return r.stdout


def wired_refs() -> set[str]:
    """source_refs already in the roster, so we don't re-pull them."""
    p = REPO / "data" / "opponent_pool.csv"
    if not p.exists():
        return set()
    import csv
    return {r["source_ref"] for r in csv.DictReader(p.open())}


def decks_from_source(src: str) -> list[list[int]]:
    """Every plausible 60-card list in a kernel's source."""
    out = []
    for m in INT_LIST.finditer(src):
        ids = [int(x) for x in re.findall(r"\d+", m.group(0))]
        if len(ids) == DECK_SIZE:
            out.append(ids)
    for m in B64.finditer(src):
        try:
            raw = base64.b64decode(m.group(1))
            if raw[:2] != b"\x1f\x8b":
                continue
            tf = tarfile.open(fileobj=io.BytesIO(raw))
            for mem in tf.getmembers():
                if mem.isfile() and mem.name.endswith("deck.csv"):
                    txt = tf.extractfile(mem).read().decode("utf-8", "ignore")
                    ids = [int(x) for x in txt.split() if x.strip().isdigit()]
                    if len(ids) >= DECK_SIZE:
                        out.append(ids[:DECK_SIZE])
                    break
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--want", default=None, help="substring of the ace name to flag")
    ap.add_argument("--sleep", type=float, default=1.0, help="politeness delay between pulls")
    ap.add_argument("--out", default="reports/public_kernel_decks.json")
    args = ap.parse_args()

    listing = _kaggle("kernels", "list", "--competition", "pokemon-tcg-ai-battle",
                      "--sort-by", "voteCount", "--page-size", str(args.top))
    refs = [ln.split()[0] for ln in listing.splitlines()[2:] if "/" in ln.split()[:1][0:1] or "/" in ln[:70]]
    refs = [r for r in refs if "/" in r][: args.top]
    known = wired_refs()

    work = REPO / ".scan_kernels"
    work.mkdir(exist_ok=True)
    rows = []
    for i, ref in enumerate(refs, 1):
        if ref in known:
            print(f"[{i}/{len(refs)}] {ref}  (already wired, skipped)")
            continue
        try:
            _kaggle("kernels", "pull", ref, "-p", str(work))
            time.sleep(args.sleep)
            src = ""
            for f in work.iterdir():
                if f.suffix == ".ipynb":
                    nb = json.loads(f.read_text(errors="ignore"))
                    src = "".join("".join(c["source"]) for c in nb["cells"]
                                  if c["cell_type"] == "code")
                elif f.suffix == ".py":
                    src = f.read_text(errors="ignore")
                f.unlink()
            if not src:
                print(f"[{i}/{len(refs)}] {ref}  (no source)")
                continue
            decks = decks_from_source(src)
            aces = Counter(ace_pokemon(d) for d in decks)
            aces.pop("(none)", None)
            label = ", ".join(f"{a}" for a, _ in aces.most_common(3)) or "no deck found"
            flag = ""
            if args.want and any(args.want.lower() in a.lower() for a in aces):
                flag = "   <<< MATCH"
            print(f"[{i}/{len(refs)}] {ref}\n      -> {label}{flag}")
            rows.append({"ref": ref, "aces": list(aces), "n_decks": len(decks)})
        except Exception as e:
            print(f"[{i}/{len(refs)}] {ref}  ERROR {type(e).__name__}: {str(e)[:60]}")
    (REPO / args.out).write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {args.out}  ({len(rows)} kernels read)")


if __name__ == "__main__":
    main()
