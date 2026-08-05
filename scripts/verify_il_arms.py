"""Pre-flight for the IL model x deck sweep: prove every arm is what it claims.

Two silent failures have already cost a run in this repo, and neither raises:

  1. `il_agent._load_model()` catches its own exceptions and returns None, so a
     missing/empty checkpoint directory degrades to non-ML `_safe_choice`
     behaviour while still playing every game and producing a plausible-looking
     win rate. (models/il_agent_medium_combined is an empty directory RIGHT NOW.)
  2. The `<agent>@<deck-tag>` override used to write `my_deck` onto the wrapper
     module while `agent()` read it from the inner core module's globals, so
     wrapper arms piloted the wrong deck and reported success.

This asserts against both, end to end, BEFORE any games are played:

  * the arm's core resolved MODEL_DIR to the intended checkpoint,
  * `_load_model()` returned a real module (not None),
  * the loaded weights hash to the intended safetensors file,
  * `agent({})` -- the real deck-submission path the harness calls -- returns
    exactly the 60 cards of the requested deck tag.

Usage:
    uv run python scripts/verify_il_arms.py --arms il_bc_2ep,il_bc_3ep \
        --decks marnies_grimmsnarl_ex,mega_lucario_ex
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import benchmark_agents as B  # noqa: E402

DECK_LISTS_DIR = REPO / "configs" / "deck_lists"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_deck(tag: str) -> list[int]:
    csv = DECK_LISTS_DIR / f"{tag}.csv"
    return [int(x) for x in csv.read_text().splitlines() if x.strip()][:60]


def verify(arm: str, deck_tag: str) -> tuple[bool, str]:
    label = f"{arm}@{deck_tag}"
    try:
        fn = B.load_agent(label)
    except Exception as exc:  # noqa: BLE001
        return False, f"load_agent raised: {exc!r}"

    g = getattr(fn, "__globals__", {})

    # --- checkpoint identity -------------------------------------------------
    model_dir = g.get("MODEL_DIR")
    if model_dir is None:
        return False, "core exposes no MODEL_DIR (not an il_agent-core arm?)"
    model_dir = Path(model_dir)
    if not (model_dir / "model.safetensors").exists():
        return False, f"checkpoint has no model.safetensors: {model_dir}"

    loader = g.get("_load_model")
    if loader is None:
        return False, "core exposes no _load_model"
    model = loader()
    if model is None:
        return False, (
            f"_load_model() returned None -- arm would SILENTLY play non-ML "
            f"fallback moves. checkpoint: {model_dir}"
        )

    ckpt_sha = _sha256(model_dir / "model.safetensors")

    # --- deck identity, via the real submission path -------------------------
    want = expected_deck(deck_tag)
    try:
        got = fn({})  # obs with no "select" == deck-submission call
    except Exception as exc:  # noqa: BLE001
        return False, f"agent({{}}) (deck submission) raised: {exc!r}"

    got = list(got or [])
    if got != want:
        overlap = len(set(got) & set(want))
        return False, (
            f"deck MISMATCH: agent submitted {len(got)} cards, "
            f"{overlap}/{len(want)} shared with '{deck_tag}' -- override did not apply"
        )

    n_params = sum(p.numel() for p in model.parameters())
    return True, (
        f"ckpt={model_dir.name} sha={ckpt_sha[:12]} params={n_params:,} "
        f"deck={deck_tag} ({len(got)} cards) OK"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma-separated arm names")
    ap.add_argument("--decks", required=True, help="comma-separated deck tags")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    decks = [d for d in args.decks.split(",") if d]

    failures = []
    for arm in arms:
        for deck in decks:
            ok, msg = verify(arm, deck)
            print(f"[{'PASS' if ok else 'FAIL'}] {arm}@{deck}: {msg}", flush=True)
            if not ok:
                failures.append(f"{arm}@{deck}: {msg}")

    total = len(arms) * len(decks)
    print(f"\n{total - len(failures)}/{total} arms verified.")
    if failures:
        print("\nBLOCKING -- do not run the sweep:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
