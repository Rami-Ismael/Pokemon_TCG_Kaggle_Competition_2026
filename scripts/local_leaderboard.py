"""Auto-generate the local leaderboard WITH full provenance, and track it over time.

A rating without provenance is unusable a week later: "il_alldays_3ep 1877" does
not say what it was trained on, which deck it piloted, whether the model actually
answered the moves, or what it scored on the real ladder. This joins every source
we have into one row per agent:

  local play   pooled Glicko + RD + GXE, field win% +/- sigma, games
  model        role + purpose (scripts/model_registry.py), checkpoint dir, sha256
  training     steps, params, hidden/layers/heads, lr, batch, seed, corpus
  offline      eval accuracy, top-3, ECE, vs the 0.381 majority baseline
  deck         which 60-card list, and its corpus episode count
  integrity    measured silent-fallback rate (0 = the model really played)
  ladder       real Kaggle score + submission ref, and whether we VERIFIED it

Every run appends a timestamped snapshot to reports/local_leaderboard_history.jsonl,
so ratings are trackable across runs the same way check_leaderboard.py tracks the
team's ladder position.

Usage:
  uv run python scripts/local_leaderboard.py \
      --merged reports/il_sweep/local_leaderboard.json \
      --fallbacks reports/il_sweep/fallback_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import model_registry as MR  # noqa: E402
from pool_predictiveness import LADDER_ANCHORS, VERIFIED_ANCHORS  # noqa: E402

MODELS = REPO / "models"
LEDGER = REPO / "reports" / "submission_ledger.jsonl"
HISTORY = REPO / "reports" / "local_leaderboard_history.jsonl"

# Corpus episodes per deck (train split 2026-07-26). See reports/deck_selection.md;
# local splits are stubs so this cannot currently be re-derived.
DECK_EPISODES = {
    "marnies_grimmsnarl_ex": 3488, "alakazam": 1542, "team_rockets_spidops": 784,
    "cynthias_garchomp_ex": 625, "mega_kangaskhan_ex": 600, "dragapult_ex": 318,
    "mega_lucario_ex": 1,
}


def arm_to_checkpoint() -> dict[str, str]:
    """Parse agents/il_arms/*/agent_core.py for the checkpoint each arm points at.

    Read from the wrapper rather than hardcoded, so this cannot drift out of sync
    with what actually gets benchmarked.
    """
    out: dict[str, str] = {}
    for d in sorted((REPO / "agents" / "il_arms").glob("*/agent_core.py")):
        m = re.search(r'"models"\s*/\s*"([^"]+)"', d.read_text())
        if m:
            out[d.parent.name] = m.group(1)
    # non-il_arms agents whose checkpoint is known
    out.setdefault("il_agent", "il_agent")
    out.setdefault("mcts_il_agent", "il_agent")
    return out


def training_meta(ckpt: str) -> dict:
    p = MODELS / ckpt / "train_metadata.json"
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}
    cfg = m.get("config") or {}
    return {
        "steps": m.get("total_steps") or m.get("step"),
        "epochs": m.get("epochs_equivalent") or cfg.get("epochs"),
        "params": m.get("n_params"),
        "eval_accuracy": m.get("eval_accuracy"),
        "eval_top3": m.get("eval_top3_accuracy"),
        "eval_ece": m.get("eval_ece"),
        "majority_baseline": m.get("majority_baseline"),
        "hidden": cfg.get("hidden_size"), "layers": cfg.get("num_layers"),
        "heads": cfg.get("num_heads"), "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"), "seed": cfg.get("seed"),
        "train_split": cfg.get("train_split"), "device": m.get("device_trained_on"),
        "git_sha": m.get("git_sha"),
    }


def ledger_scores() -> dict[str, dict]:
    """Latest ladder reading per submission ref, from the append-only ledger."""
    if not LEDGER.exists():
        return {}
    out: dict[str, dict] = {}
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ref = str(rec.get("ref") or "")
        if not ref:
            continue
        cur = out.setdefault(ref, {"ref": ref})
        for k in ("score", "status", "kaggle_description", "deck", "checkpoint_sha256"):
            if rec.get(k) is not None:
                cur[k] = rec[k]
        for r in (rec.get("readings") or []):
            if r.get("score") is not None:
                cur["score"] = r["score"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--fallbacks", type=Path, default=None)
    ap.add_argument("--out-md", type=Path,
                    default=REPO / "reports" / "local_leaderboard.md")
    ap.add_argument("--no-history", action="store_true")
    args = ap.parse_args()

    merged = json.loads(args.merged.read_text())
    glicko, agg = merged["glicko"], merged["aggregate"]
    pool = set(merged["pool"])
    a2c = arm_to_checkpoint()
    reg = {r["dir"]: r for r in MR.scan()}
    fallbacks = json.loads(args.fallbacks.read_text()) if (
        args.fallbacks and args.fallbacks.exists()) else {}
    subs = ledger_scores()
    sha_to_ref = {v.get("checkpoint_sha256"): v for v in subs.values()
                  if v.get("checkpoint_sha256")}

    rows = []
    for name in sorted(glicko, key=lambda n: -glicko[n]["rating"]):
        base, _, deck = name.partition("@")
        ckpt = a2c.get(base)
        r = reg.get(ckpt, {}) if ckpt else {}
        tm = training_meta(ckpt) if ckpt else {}
        anc = LADDER_ANCHORS.get(name) or LADDER_ANCHORS.get(base)
        sub = sha_to_ref.get(r.get("sha256"))
        fb = fallbacks.get(name)
        rows.append({
            "agent": name,
            "role": r.get("role") or ("pool" if name in pool else "—"),
            "clear_name": r.get("name"),
            "purpose": r.get("purpose"),
            "glicko": round(glicko[name]["rating"], 1),
            "rd": round(glicko[name]["rd"], 1),
            "gxe": round(glicko[name]["gxe"], 1),
            "field_win_pct": round(agg[name]["win_pct"], 1) if name in agg else None,
            "sigma": round(agg[name]["sigma"], 1) if name in agg else None,
            "games": agg[name]["games"] if name in agg else None,
            "checkpoint": ckpt, "checkpoint_sha256": r.get("sha256"),
            "deck": deck or None,
            "deck_corpus_episodes": DECK_EPISODES.get(deck) if deck else None,
            "training": tm,
            "fallback_rate_pct": fb.get("rate_pct") if fb else None,
            "fallback_decisions": fb.get("decisions") if fb else None,
            "ladder_score": anc[0] if anc else None,
            "ladder_provenance": anc[1] if anc else None,
            "ladder_verified": (name in VERIFIED_ANCHORS or base in VERIFIED_ANCHORS),
            "submission_ref": sub.get("ref") if sub else None,
        })

    ts = datetime.now(timezone.utc).isoformat()
    if not args.no_history:
        HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY.open("a") as fh:
            fh.write(json.dumps({"timestamp_utc": ts, "source": str(args.merged),
                                 "n_agents": len(rows), "rows": rows}) + "\n")

    # ---- render -----------------------------------------------------------
    L = []
    L.append("# Local leaderboard (auto-generated)\n")
    L.append(f"Generated {ts} by `scripts/local_leaderboard.py`. "
             f"Do not hand-edit — re-run the script.\n")
    L.append(f"Source: `{args.merged}`. History appended to "
             f"`reports/local_leaderboard_history.jsonl` ({len(rows)} agents this run).\n")
    L.append("> Local ratings are **hypotheses about the ladder**, not ladder claims. "
             "The `ladder` column is the real score where one exists; `verified` marks "
             "the ones we read ourselves rather than the ones their authors reported.\n")

    L.append("\n## Standings\n")
    L.append("| # | agent | role | Glicko | RD | GXE% | field win% | games | deck (corpus eps) | ladder |")
    L.append("|---:|---|---|---:|---:|---:|---|---:|---|---|")
    for i, r in enumerate(rows, 1):
        wr = f"{r['field_win_pct']:.1f} ± {r['sigma']:.1f}%" if r["field_win_pct"] is not None else "—"
        deck = f"{r['deck']} ({r['deck_corpus_episodes']})" if r["deck"] else "—"
        lad = "—"
        if r["ladder_score"]:
            lad = f"**{r['ladder_score']:.1f}**" + (" ✓" if r["ladder_verified"] else " *(alleged)*")
        L.append(f"| {i} | `{r['agent']}` | {r['role']} | {r['glicko']} | {r['rd']:.0f} | "
                 f"{r['gxe']} | {wr} | {r['games'] or '—'} | {deck} | {lad} |")

    L.append("\n## Provenance — how each of our checkpoints was trained\n")
    L.append("| checkpoint | role | steps | params | hidden/layers/heads | lr | batch | seed | offline acc | top-3 | ECE | sha256 |")
    L.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    seen = set()
    for r in rows:
        c = r["checkpoint"]
        if not c or c in seen or not r["training"]:
            continue
        seen.add(c)
        t = r["training"]
        arch = f"{t.get('hidden') or '—'}/{t.get('layers') or '—'}/{t.get('heads') or '—'}"
        def num(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"

        steps = f"{t['steps']:,}" if isinstance(t.get("steps"), int) else "—"
        params = f"{t['params']:,}" if isinstance(t.get("params"), int) else "—"
        L.append(
            f"| `{c}` | {r['role']} | {steps} | {params} | {arch} | "
            f"{t.get('lr') or '—'} | {t.get('batch_size') or '—'} | {t.get('seed') or '—'} | "
            f"{num(t.get('eval_accuracy'))} | {num(t.get('eval_top3'))} | "
            f"{num(t.get('eval_ece'))} | "
            f"`{(r['checkpoint_sha256'] or '')[:12]}` |")
    L.append("\nMajority-class baseline on the same eval rows: **0.381**.\n")

    L.append("\n## What each checkpoint is FOR\n")
    for r in rows:
        if r["purpose"] and r["clear_name"] and r["checkpoint"] not in (None,):
            key = (r["clear_name"], r["purpose"])
            if key in seen:
                continue
            seen.add(key)
            L.append(f"- **`{r['clear_name']}`** (`models/{r['checkpoint']}`) — {r['purpose']}")

    fb_rows = [r for r in rows if r["fallback_rate_pct"] is not None]
    if fb_rows:
        L.append("\n## Integrity — did the model actually play?\n")
        L.append("| agent | decisions measured | silent-fallback rate |")
        L.append("|---|---:|---:|")
        for r in fb_rows:
            L.append(f"| `{r['agent']}` | {r['fallback_decisions']:,} | {r['fallback_rate_pct']:.2f}% |")
        L.append("\nA non-zero rate means that share of moves came from `_safe_choice`, "
                 "not the checkpoint — the win rate would then be partly a measurement "
                 "of the fallback heuristic.\n")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(L) + "\n")
    print(f"wrote {args.out_md}  ({len(rows)} agents)")
    if not args.no_history:
        n = sum(1 for _ in HISTORY.open())
        print(f"appended snapshot -> {HISTORY}  ({n} snapshot(s) tracked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
