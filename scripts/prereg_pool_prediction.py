"""Pre-register the local pool's predicted ladder ordering — rl_pipeline_v4 §6.3, work item 7.

The anchored pool's Spearman rho = +0.929 is IN-SAMPLE: the pool has never been
tested on a submission it did not help select. This script is the only thing that
can turn it into an out-of-sample number, and it costs nothing — the prediction is
recorded BEFORE the ladder reads back, in git and in the ledger's `expects` field.

What is pre-registered, and why it is the ORDINAL prediction:

  A score-level map is available (OLS of ladder on local Glicko over the 9
  overlapping agents: ladder = -649.5 + 0.8955 * local, residual SD 171.6) but
  it is not worth registering as the primary claim. The residual SD is larger
  than the ~+/-100 same-build ladder noise band, and 5 of the 9 anchors are
  self-reported rather than ledger-verified ([[ladder-anchors-mostly-alleged]]).
  The verified-only slice that survives is n=2, which cannot support a fit at
  all. Spearman rho is an ORDINAL statistic, so registering the ordering is both
  what §6.3 asks for and the part of the claim the alleged anchors cannot move.

  The frozen coefficients are still recorded, with a deliberately wide band, so
  a later score-level check is possible without anyone re-fitting after seeing
  the outcome. That refit-after-the-fact is the specific failure this file
  exists to make impossible.

Workflow:

  1. lock    -- record a candidate + its local evidence + predicted rank, BEFORE
                submitting. Append-only; a second lock for the same candidate is
                an amendment and both survive in the file.
  2. expects -- print the one-line prediction string to pass to
                `submission_ledger.py log --expects` at submit time.
  3. bind    -- attach the Kaggle ref once the candidate is submitted.
  4. score   -- once >=4 bound candidates have settled reads, report the
                OUT-OF-SAMPLE Spearman rho of predicted vs actual ordering.

Usage:
  uv run python scripts/prereg_pool_prediction.py lock --candidate mega_lucario_restore \\
      --local-glicko 1711.7 --local-rd 30 --source reports/wmh_pool_calibration.json \\
      --note "byte-identical resubmit of the 804.0 bundle"
  uv run python scripts/prereg_pool_prediction.py show
  uv run python scripts/prereg_pool_prediction.py expects --candidate mega_lucario_restore
  uv run python scripts/prereg_pool_prediction.py bind --candidate mega_lucario_restore --ref 55281234
  uv run python scripts/prereg_pool_prediction.py score
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pokemon_tcg import config  # noqa: E402

PREREG = config.REPORTS_DIR / "pool_prereg.jsonl"
LEDGER = config.REPORTS_DIR / "submission_ledger.jsonl"

# >>> FROZEN 2026-08-05, before any of the pre-registered candidates read back.
# Fit by scripts/pool_predictiveness.py's anchor set over
# reports/wmh_pool_calibration.json (n=9). Do NOT refit these after a candidate
# settles -- refitting on the outcome is exactly what makes an in-sample rho
# look good and predict nothing.
CALIB = {
    "fit": "ladder = -554.4 + 0.8076 * local_glicko",
    "intercept": -554.4,
    "slope": 0.8076,
    "n_anchors": 19,
    "residual_sd": 198.7,
    "frozen_at_utc": "2026-08-05",
    "in_sample_rho": 0.718,
    "in_sample_p": 0.0007,
    "source": "reports/pool_prereg_source_glicko.json (sha256 0f0b7954fe7fb0b5)",
    "caveat": (
        "Fit on the COMPOUNDED local Glicko scale, snapshotted immutably as "
        "reports/pool_prereg_source_glicko.json so the fit is reproducible "
        "from this commit -- reports/glicko_ratings.json keeps compounding "
        "across runs and would not be. Most anchors are self-reported rather "
        "than ledger-verified ([[ladder-anchors-mostly-alleged]]); the "
        "verified-only slice cannot support a fit. Residual SD (198.7) is "
        "about twice the ~+/-100 same-build ladder noise band, so the score "
        "band is wide on purpose and the ORDINAL prediction is the registered "
        "claim."
    ),
}
# <<< FROZEN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read() -> list[dict]:
    if not PREREG.exists():
        return []
    return [json.loads(ln) for ln in PREREG.read_text().splitlines() if ln.strip()]


def _append(row: dict) -> None:
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    with PREREG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _fold(rows: list[dict]) -> dict[str, dict]:
    """Later-wins per field, per candidate — same convention as the ledger."""
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("kind") != "prediction":
            continue
        cur = out.setdefault(r["candidate"], {})
        for k, v in r.items():
            if v is not None:
                cur[k] = v
    return out


def _predicted_score(local_glicko: float) -> float:
    return CALIB["intercept"] + CALIB["slope"] * local_glicko


def cmd_lock(args: argparse.Namespace) -> None:
    rows = _read()
    if not any(r.get("kind") == "calibration" for r in rows):
        _append({"kind": "calibration", "locked_at_utc": _now(), **CALIB})

    folded = _fold(rows)
    pred = _predicted_score(args.local_glicko)
    band = 2 * CALIB["residual_sd"]

    # Predicted ordinal rank among everything locked so far (1 = strongest).
    # Recorded as the local Glicko itself so the ordering is reconstructable
    # even if candidates are locked out of order or amended later.
    row = {
        "kind": "prediction",
        "candidate": args.candidate,
        "locked_at_utc": _now(),
        "local_glicko": args.local_glicko,
        "local_rd": args.local_rd,
        "local_source": args.source,
        "predicted_ladder": round(pred, 1),
        "predicted_band_95": [round(pred - band, 1), round(pred + band, 1)],
        "note": args.note,
        "ref": None,
        "calibration": CALIB["fit"],
    }
    _append(row)

    folded[args.candidate] = row
    ordered = sorted(folded.values(), key=lambda r: -r["local_glicko"])
    rank = 1 + [r["candidate"] for r in ordered].index(args.candidate)
    print(f"locked {args.candidate}: local Glicko {args.local_glicko:.1f} "
          f"+/-{args.local_rd:.0f} -> predicted ladder {pred:.1f} "
          f"[{pred - band:.1f}, {pred + band:.1f}]")
    print(f"predicted rank {rank} of {len(ordered)} locked candidates")
    print(f"-> {PREREG}")


def cmd_show(args: argparse.Namespace) -> None:
    rows = _read()
    cal = next((r for r in rows if r.get("kind") == "calibration"), None)
    if cal:
        print(f"calibration frozen {cal['locked_at_utc']}: {cal['fit']}")
        print(f"  in-sample rho {cal['in_sample_rho']:+.3f} (n={cal['n_anchors']}, "
              f"p={cal['in_sample_p']:.4f}), residual SD {cal['residual_sd']:.1f}")
        print(f"  {cal['caveat']}\n")

    folded = _fold(rows)
    if not folded:
        print("no candidates locked yet")
        return
    ordered = sorted(folded.values(), key=lambda r: -r["local_glicko"])
    print(f"{'#':>2s} {'candidate':30s} {'local':>8s} {'pred':>8s} "
          f"{'ref':>10s}  note")
    for i, r in enumerate(ordered, 1):
        print(f"{i:>2d} {r['candidate']:30s} {r['local_glicko']:>8.1f} "
              f"{r['predicted_ladder']:>8.1f} {str(r.get('ref') or '-'):>10s}  "
              f"{r.get('note') or ''}")


def cmd_expects(args: argparse.Namespace) -> None:
    folded = _fold(_read())
    r = folded.get(args.candidate)
    if r is None:
        sys.exit(f"no locked prediction for {args.candidate!r}; run `lock` first")
    ordered = sorted(folded.values(), key=lambda x: -x["local_glicko"])
    rank = 1 + [x["candidate"] for x in ordered].index(args.candidate)
    lo, hi = r["predicted_band_95"]
    print(
        f"PRE-REGISTERED ({r['locked_at_utc']}, before any read-back): local pool "
        f"Glicko {r['local_glicko']:.1f} +/-{r['local_rd']:.0f} ranks this "
        f"candidate {rank} of {len(ordered)} in the pre-registered slate. "
        f"Ordinal prediction is the registered claim; the frozen map "
        f"({r['calibration']}, residual SD {CALIB['residual_sd']:.1f}) gives a "
        f"wide point estimate {r['predicted_ladder']:.1f} [{lo:.1f}, {hi:.1f}]. "
        f"Out-of-sample rho scored by scripts/prereg_pool_prediction.py score."
    )


def cmd_bind(args: argparse.Namespace) -> None:
    folded = _fold(_read())
    if args.candidate not in folded:
        sys.exit(f"no locked prediction for {args.candidate!r}")
    _append({"kind": "prediction", "candidate": args.candidate,
             "locked_at_utc": _now(), "ref": args.ref,
             "local_glicko": folded[args.candidate]["local_glicko"]})
    print(f"bound {args.candidate} -> ref {args.ref}")


def _ledger_scores() -> dict[str, float]:
    """Latest non-null score per ref.

    The ledger is append-only and chronological, and score readings are rows
    of the shape {at_utc, event, ref, score, status} — so last-wins over the
    file is the latest reading. Pending rows carry score=null and are skipped.
    """
    if not LEDGER.exists():
        return {}
    out: dict[str, float] = {}
    for ln in LEDGER.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        ref, score = str(r.get("ref") or ""), r.get("score")
        if ref and isinstance(score, (int, float)):
            out[ref] = float(score)
    return out


def cmd_score(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pool_predictiveness import spearman  # noqa: PLC0415

    folded = _fold(_read())
    scores = _ledger_scores()
    bound = [(c, r) for c, r in folded.items()
             if r.get("ref") and str(r["ref"]) in scores]
    if len(bound) < 4:
        print(f"{len(bound)} pre-registered candidate(s) have a settled ladder "
              f"read; need >=4 before an out-of-sample rho means anything.")
        for c, r in sorted(folded.items()):
            got = scores.get(str(r.get("ref") or ""))
            print(f"  {c:30s} ref {str(r.get('ref') or '-'):>10s} "
                  f"actual {('%.1f' % got) if got else 'pending'}")
        return

    local = [r["local_glicko"] for _, r in bound]
    actual = [scores[str(r["ref"])] for _, r in bound]
    rho = spearman(local, actual)
    print(f"OUT-OF-SAMPLE Spearman rho = {rho:+.3f} (n={len(bound)})")
    print("Predictions were locked before each read-back; this is the number "
          "that decides whether the pool may ever gate a submission.")
    for (c, r), a in zip(bound, actual):
        print(f"  {c:30s} predicted {r['predicted_ladder']:7.1f}  actual {a:7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lock", help="record a prediction BEFORE submitting")
    p.add_argument("--candidate", required=True)
    p.add_argument("--local-glicko", type=float, required=True)
    p.add_argument("--local-rd", type=float, required=True)
    p.add_argument("--source", required=True, help="benchmark result the Glicko came from")
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("show", help="table of locked predictions")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("expects", help="print the ledger --expects string")
    p.add_argument("--candidate", required=True)
    p.set_defaults(func=cmd_expects)

    p = sub.add_parser("bind", help="attach a Kaggle ref to a locked candidate")
    p.add_argument("--candidate", required=True)
    p.add_argument("--ref", required=True)
    p.set_defaults(func=cmd_bind)

    p = sub.add_parser("score", help="out-of-sample rho once >=4 have settled")
    p.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
