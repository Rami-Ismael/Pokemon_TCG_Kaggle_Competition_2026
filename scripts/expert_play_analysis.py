"""What slice of the legal action space do strong players actually use?

The transferable-lesson analysis (see .claude/skills/expert-play-analysis):
the episode corpus + manifest scores have value independent of whether IL
survives into the final agent. Filter episodes to matches where BOTH
players are highly rated (manifest ``min_score``) vs matches where both are
weak (``max_score = 2*avg - min``), then measure, decision by decision:

* which OptionTypes strong players pick from what the engine legally offers
  (pick share, pick-rate-given-legal, and the smallest set of option types
  covering 95% of picks -- the "effective decision space");
* macro-pattern probes: attach-when-available-and-unused by turn position
  ("energy first"), retreat rate conditional on active HP, END chosen while
  ATTACK was legal;
* per-SelectContext volume, forced-decision share, and decline usage.

Findings describe OFFLINE HUMAN/AGENT BEHAVIOR on the ladder, not our
agent's strength -- they are feature/masking candidates that survive a
pivot to self-play RL.

Data source is the ADR-001 shard corpus (Hub ``config.HF_EPISODES_REPO`` or
a ``--local-root`` with the same layout). Shards carry manifest scores next
to the raw ``episode_json`` bytes precisely so this filter never parses
unselected episodes. Days with no manifest scores (e.g. train/2026-07-26)
cannot be analyzed -- the script says so instead of proxying.

USAGE
-----
    uv run python scripts/expert_play_analysis.py                    # eval/2026-07-27
    uv run python scripts/expert_play_analysis.py --top 150 --bottom 150
    uv run python scripts/expert_play_analysis.py --split train --day 2026-07-01

Output: printed report + reports/expert_play_analysis-{split}-{day}.json.
Every rate carries a Wilson 95% CI; strong-vs-weak differences are flagged
only when the intervals are disjoint (repo standing rule: no uncertainty,
no claim).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

for _cand in [REPO / "data" / "external" / "cg-lib",
              *[p / "data" / "external" / "cg-lib" for p in REPO.parents]]:
    if (_cand / "cg" / "api.py").exists():
        sys.path.insert(0, str(_cand))
        break

from cg.api import OptionType, SelectContext  # noqa: E402

from pokemon_tcg import config  # noqa: E402
from pokemon_tcg.il_dataset import (  # noqa: E402
    _resolve_hub_shards,
    iter_episode_decisions,
)

MANIFEST = REPO / "data" / "episodes" / "manifest.csv"
MAIN_CTX = int(SelectContext.MAIN)
DECLINE = "DECLINE"


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """(rate, lo, hi) 95% Wilson interval; (nan, nan, nan) when n == 0."""
    if n == 0:
        return (float("nan"),) * 3
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, center - half, center + half


def fmt_rate(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "     n=0"
    return f"{100 * p:5.1f}% [{100 * lo:.1f},{100 * hi:.1f}] n={n}"


def disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when the two Wilson CIs do not overlap (a real difference)."""
    _, alo, ahi = wilson(*a)
    _, blo, bhi = wilson(*b)
    if any(math.isnan(x) for x in (alo, ahi, blo, bhi)):
        return False
    return ahi < blo or bhi < alo


def load_strata(split: str, day: str, top: int, bottom: int) -> dict[str, dict]:
    """episode_id -> stratum via manifest scores for one day.

    strong = top-N by min_score (BOTH seats strong, so every decision in the
    episode is a strong player's). weak = bottom-N by max_score (both seats
    weak). Filtering strong by avg alone would admit lopsided matches whose
    weak seat pollutes the stratum -- min/max sidestep seat attribution,
    which neither the manifest nor iter_episode_decisions provides.
    """
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST} (worktree missing the "
                 "data/episodes symlink? run the run-ptcg-battle doctor)")
    rows = []
    with MANIFEST.open(newline="") as f:
        for r in csv.DictReader(f):
            if not r["create_time"].startswith(day) or not r["avg_score"]:
                continue
            if r["agent_count"] != "2":
                continue
            avg, mn = float(r["avg_score"]), float(r["min_score"])
            rows.append({"id": r["episode_id"], "min": mn, "avg": avg,
                         "max": 2 * avg - mn})
    if not rows:
        sys.exit(f"no scored manifest rows for day {day} -- scores exist "
                 "only where the ingest recorded them (e.g. train/2026-07-26 "
                 "has none; use eval/2026-07-27 or train/2026-07-01)")

    by_min = sorted(rows, key=lambda r: r["min"], reverse=True)
    by_max = sorted(rows, key=lambda r: r["max"])
    strong = by_min[:top]
    strong_ids = {r["id"] for r in strong}
    weak = [r for r in by_max if r["id"] not in strong_ids][:bottom]

    def rng(rs, key):
        vals = sorted(r[key] for r in rs)
        return f"{vals[0]:.0f}-{vals[-1]:.0f}"

    print(f"day {day}: {len(rows)} scored episodes")
    print(f"  strong: {len(strong)} eps, min_score {rng(strong, 'min')} "
          f"(both seats at/above the floor)")
    print(f"  weak:   {len(weak)} eps, max_score {rng(weak, 'max')} "
          f"(both seats at/below the ceiling)")
    strata = {r["id"]: "strong" for r in strong}
    strata.update({r["id"]: "weak" for r in weak})
    return {"strata": strata,
            "meta": {"day": day, "split": split,
                     "n_scored": len(rows),
                     "strong_n": len(strong), "weak_n": len(weak),
                     "strong_min_score_range": rng(strong, "min"),
                     "weak_max_score_range": rng(weak, "max")}}


def iter_selected_episodes(split: str, day: str, wanted: set[str],
                           local_root: Path | None):
    """Yield (episode_id, parsed_episode) for wanted ids from the shards.

    Column projection reads scores/ids cheaply; episode_json is parsed only
    for selected rows. Hub shards land in the huggingface_hub cache, so the
    first pass pays the download and every later pass is SSD-speed.
    """
    import pyarrow.parquet as pq

    if local_root is not None:
        files = sorted(str(p.relative_to(local_root)) for p in
                       (local_root / split).glob(f"day={day}/shard-*.parquet"))
        paths = [local_root / f for f in files]
    else:
        files = _resolve_hub_shards(config.HF_EPISODES_REPO, split, [day])
        if not files:
            sys.exit(f"no shards for {split}/day={day} in "
                     f"{config.HF_EPISODES_REPO}")
        from huggingface_hub import hf_hub_download
        paths = []
        for f in files:
            t0 = time.monotonic()
            paths.append(hf_hub_download(config.HF_EPISODES_REPO, f,
                                         repo_type="dataset"))
            dt = time.monotonic() - t0
            note = "cache" if dt < 1 else f"{dt:.0f}s"
            print(f"  shard {f} ({note})")

    found = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=8,
                                     columns=["episode_id", "episode_json"]):
            ids = [str(v) for v in batch.column("episode_id").to_pylist()]
            blobs = batch.column("episode_json")
            for i, eid in enumerate(ids):
                if eid in wanted:
                    found += 1
                    yield eid, json.loads(blobs[i].as_py())
    if found < len(wanted):
        print(f"  note: only {found}/{len(wanted)} selected episodes were "
              "present in the shards")


class StratumStats:
    def __init__(self) -> None:
        self.episodes = 0
        self.decisions = 0
        self.forced = 0
        self.ctx = defaultdict(lambda: {"n": 0, "forced": 0, "opts": 0,
                                        "decline_legal": 0, "declined": 0})
        self.main_picks = Counter()          # OptionType name -> picks
        self.main_legal = Counter()          # name -> decisions where legal
        self.main_pick_given_legal = Counter()
        self.attach_first = Counter()        # bucket -> [k, n] flattened
        self.retreat_hp = Counter()
        self.end_while_attack = [0, 0]       # [END picks, ATTACK-legal n]

    def add(self, obs: dict, label: int) -> None:
        sel = obs["select"]
        opts = sel.get("option") or []
        n_opts = len(opts)
        ctx_name = SelectContext(sel.get("context", 0)).name \
            if sel.get("context", 0) < len(SelectContext) else str(sel["context"])
        self.decisions += 1
        c = self.ctx[ctx_name]
        c["n"] += 1
        c["opts"] += n_opts
        forced = n_opts == 1 and (sel.get("minCount") or 0) >= 1
        if forced:
            self.forced += 1
            c["forced"] += 1
        chosen_decline = label == n_opts
        if (sel.get("minCount") or 0) == 0:
            c["decline_legal"] += 1
            if chosen_decline:
                c["declined"] += 1

        if sel.get("context", -1) != MAIN_CTX or forced:
            return

        types = [OptionType(o["type"]).name for o in opts]
        legal_set = set(types)
        picked = DECLINE if chosen_decline else types[label]
        self.main_picks[picked] += 1
        for t in legal_set:
            self.main_legal[t] += 1
        if picked != DECLINE:
            self.main_pick_given_legal[picked] += 1

        cur = obs.get("current") or {}
        if "ATTACH" in legal_set and not cur.get("energyAttached"):
            bucket = "act0-1" if cur.get("turnActionCount", 0) <= 1 else "act2+"
            self.attach_first[f"{bucket}/n"] += 1
            if picked == "ATTACH":
                self.attach_first[f"{bucket}/k"] += 1
        if "RETREAT" in legal_set:
            hp_bucket = "hp?"
            you = (cur.get("players") or [{}, {}])[cur.get("yourIndex", 0)]
            active = (you.get("active") or [None])[0]
            if active and active.get("maxHp"):
                frac = (active.get("hp") or 0) / active["maxHp"]
                hp_bucket = ("hp<=1/3" if frac <= 1 / 3
                             else "hp<=2/3" if frac <= 2 / 3 else "hp>2/3")
            self.retreat_hp[f"{hp_bucket}/n"] += 1
            if picked == "RETREAT":
                self.retreat_hp[f"{hp_bucket}/k"] += 1
        if "ATTACK" in legal_set:
            self.end_while_attack[1] += 1
            if picked == "END":
                self.end_while_attack[0] += 1

    def coverage_95(self) -> tuple[int, list[str]]:
        total = sum(self.main_picks.values())
        if not total:
            return 0, []
        got, names = 0, []
        for name, k in self.main_picks.most_common():
            names.append(name)
            got += k
            if got / total >= 0.95:
                break
        return len(names), names


def report(stats: dict[str, StratumStats], meta: dict, out_path: Path) -> None:
    s, w = stats["strong"], stats["weak"]
    print("\n=== decisions (both seats of each episode) ===")
    for name, st in (("strong", s), ("weak", w)):
        forced_pct = 100 * st.forced / st.decisions if st.decisions else 0
        print(f"  {name}: {st.episodes} eps, {st.decisions} decisions "
              f"({forced_pct:.1f}% forced single-option -- excluded from "
              "MAIN metrics)")

    print("\n=== MAIN decisions: pick share by OptionType (of engine-legal "
          "options) ===")
    all_types = sorted(set(s.main_picks) | set(w.main_picks),
                       key=lambda t: -s.main_picks[t])
    s_tot, w_tot = sum(s.main_picks.values()), sum(w.main_picks.values())
    print(f"  {'type':<10} {'strong share':>24} {'weak share':>24}  sep?")
    for t in all_types:
        a, b = (s.main_picks[t], s_tot), (w.main_picks[t], w_tot)
        mark = "  ≠" if disjoint(a, b) else ""
        print(f"  {t:<10} {fmt_rate(*a):>24} {fmt_rate(*b):>24}{mark}")

    print("\n=== MAIN: P(pick type | type legal, not forced) ===")
    for t in sorted(set(s.main_legal) | set(w.main_legal)):
        a = (s.main_pick_given_legal[t], s.main_legal[t])
        b = (w.main_pick_given_legal[t], w.main_legal[t])
        mark = "  ≠" if disjoint(a, b) else ""
        print(f"  {t:<10} {fmt_rate(*a):>24} {fmt_rate(*b):>24}{mark}")

    print("\n=== macro-pattern probes (MAIN, not forced) ===")
    def probe(title, sk, sn, wk, wn):
        mark = "  ≠" if disjoint((sk, sn), (wk, wn)) else ""
        print(f"  {title:<38} {fmt_rate(sk, sn):>24} "
              f"{fmt_rate(wk, wn):>24}{mark}")

    for bucket in ("act0-1", "act2+"):
        probe(f"attach | attach legal+unused, {bucket}",
              s.attach_first[f"{bucket}/k"], s.attach_first[f"{bucket}/n"],
              w.attach_first[f"{bucket}/k"], w.attach_first[f"{bucket}/n"])
    for hp in ("hp<=1/3", "hp<=2/3", "hp>2/3", "hp?"):
        probe(f"retreat | retreat legal, {hp}",
              s.retreat_hp[f"{hp}/k"], s.retreat_hp[f"{hp}/n"],
              w.retreat_hp[f"{hp}/k"], w.retreat_hp[f"{hp}/n"])
    probe("END picked | ATTACK legal",
          s.end_while_attack[0], s.end_while_attack[1],
          w.end_while_attack[0], w.end_while_attack[1])

    for name, st in (("strong", s), ("weak", w)):
        k, names = st.coverage_95()
        print(f"\n  {name}: {k} option types cover >=95% of MAIN picks: "
              f"{names}")

    print("\n=== per-context volume (strong stratum) ===")
    print(f"  {'context':<28} {'n':>7} {'forced%':>8} {'mean opts':>10} "
          f"{'declined/decline-legal':>24}")
    for ctx_name, c in sorted(s.ctx.items(), key=lambda kv: -kv[1]["n"])[:12]:
        print(f"  {ctx_name:<28} {c['n']:>7} "
              f"{100 * c['forced'] / c['n']:>7.1f}% "
              f"{c['opts'] / c['n']:>10.1f} "
              f"{c['declined']:>12}/{c['decline_legal']}")

    payload = {"meta": meta}
    for name, st in stats.items():
        payload[name] = {
            "episodes": st.episodes, "decisions": st.decisions,
            "forced": st.forced,
            "main_picks": dict(st.main_picks),
            "main_legal": dict(st.main_legal),
            "main_pick_given_legal": dict(st.main_pick_given_legal),
            "attach_first": dict(st.attach_first),
            "retreat_hp": dict(st.retreat_hp),
            "end_while_attack": st.end_while_attack,
            "coverage_95": st.coverage_95(),
            "per_context": {k: dict(v) for k, v in st.ctx.items()},
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nreport JSON -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--day", default="2026-07-27")
    ap.add_argument("--top", type=int, default=150,
                    help="strong stratum size (episodes, by min_score)")
    ap.add_argument("--bottom", type=int, default=150,
                    help="weak stratum size (episodes, by max_score)")
    ap.add_argument("--local-root", type=Path, default=None,
                    help="read shards from this dir instead of the Hub "
                         "(layout {split}/day=.../shard-*.parquet)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sel = load_strata(args.split, args.day, args.top, args.bottom)
    strata, meta = sel["strata"], sel["meta"]

    stats = {"strong": StratumStats(), "weak": StratumStats()}
    t0 = time.monotonic()
    for eid, episode in iter_selected_episodes(
            args.split, args.day, set(strata), args.local_root):
        st = stats[strata[eid]]
        st.episodes += 1
        for obs, label, _exclude in iter_episode_decisions(episode):
            st.add(obs, label)
    meta["elapsed_s"] = round(time.monotonic() - t0, 1)
    print(f"\nparsed {stats['strong'].episodes + stats['weak'].episodes} "
          f"episodes in {meta['elapsed_s']}s")

    out = args.out or (REPO / "reports" /
                       f"expert_play_analysis-{args.split}-{args.day}.json")
    report(stats, meta, out)


if __name__ == "__main__":
    main()
