"""Does the local benchmark pool predict the real ladder? Measure it, don't assert it.

Correlates local Glicko ratings (from a benchmark run's result JSON) against
real Kaggle-ladder readings for every agent that has both. This is the running
answer to the leaderboard-check skill's question -- two confirmed local-vs-ladder
inversions (2026-08-02, 2026-08-03) are why this number exists.

Ladder anchor table caveats (all matter, stated so the rho isn't over-read):
- Readings span DIFFERENT dates/eras of a live ladder (field grew ~4.5k -> 6.2k
  teams July->August; Elo scale drifts). Same-build spread is ~ +/-100.
- wmh_* values are self-reported in github.com/wmh/ptcg-abc docs (July readings).
- tb_* values are TomBombadyl's decoded-submission catalog mu.
- Ours are read from reports/submission_ledger.jsonl history.
A high rho here is necessary, not sufficient, for trusting local ordering.

Usage:
  uv run python scripts/pool_predictiveness.py [--result reports/wmh_pool_calibration.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# agent -> (ladder score, provenance)
LADDER_ANCHORS: dict[str, tuple[float, str]] = {
    "tb_archaludon": (1196.1, "TomBombadyl catalog mu (his leader)"),
    "romanrozen_strong_start": (950.0, "notebook title 'LB 960'/~950 claim"),
    "makthanithin_1084_baseline": (1084.5, "notebook's scored-target claim"),
    "wmh_megastarmie": (871.5, "wmh docs: megastarmie v3, 2026-06-24"),
    "wmh_alakazam": (860.3, "wmh docs: Alakazam v3 new-high, 2026-07-08"),
    "wmh_bellibolt": (836.0, "wmh README ladder Elo"),
    "wmh_garchomp": (713.8, "wmh docs: garchomp first run, 2026-07-08"),
    "wmh_typhlosion": (532.0, "wmh README ladder Elo"),
    "tb_dragapult": (880.9, "TomBombadyl catalog mu"),
    "tb_alakazam": (659.0, "TomBombadyl catalog mu"),
    "tb_starmie": (277.5, "TomBombadyl catalog mu"),
    "tb_search": (660.5, "TomBombadyl catalog mu"),
    "tb_heuristic": (633.0, "TomBombadyl catalog mu"),
    "tb_rulecore": (535.6, "TomBombadyl catalog mu"),
    "improved_prob_main": (701.6, "our sub 55169814, 2026-08-01"),
    # CORRECTED 2026-08-05. 804.0 was ONE early read of 55162376 and is not what
    # this build polls at now. Four same-build resubmits settled at 699.0
    # (55191752), 692.7 (55219194), 666.1 (55224682), 683.2 (55228113) -> mean
    # 685.3. Using best-ever here silently inflated the anchor by ~119 points and
    # produced a false local-vs-ladder "inversion" against improved_prob_main.
    "agent_core_improved": (685.3, "our subs 55191752/55219194/55224682/55228113 mean of settled reads"),
    "il_agent": (398.7, "our subs 55190924/55149903 mean"),
    "s2_e1_s43": (395.0, "our sub 55196434"),
    "ppo_u120832": (275.1, "our sub 55215267 (drifting read)"),
    # Fresh reads for agents that are IN this study's local leaderboard, so the
    # correlation includes points where both sides are ours and measured.
    "il_alldays_3ep@mega_lucario_ex": (422.8, "our sub 55248985, settled read 3 (600.0 was the mu0 prior)"),
    "mcts_il_agent": (291.4, "our sub 55248781, latest of 4 reads"),
    # same weights as `il_agent` on the same deck -> same verified anchor
    "il_bc_3ep@mega_lucario_ex": (398.7, "our subs 55190924/55149903 mean (il_agent weights, Lucario)"),
    "il_alldays_3ep@marnies_grimmsnarl_ex": (311.3, "our sub 55270787, settled over 3 reads"),
    # Self-play arms, added 2026-08-06. These are the anchors that let the pool
    # be scored on the self-play family specifically -- the slice where it
    # performs worst (rho +0.000 over these two plus the two IL arms above).
    "selfplay_g1_ref430k": (254.9, "our sub 55253900, latest of 5 reads"),
    "selfplay_g3_final": (405.4, "our sub 55284059, read 3; reads 1-2 said 564.9 -- see below"),
}

# ⚠️ selfplay_g3_final is the cautionary anchor. Submitted 2026-08-06 to settle
# whether self-play beats imitation training, it read 564.9 TWICE over the first
# half hour (which would have meant a +147-point win over its own init) and then
# settled to 405.4 -- level with that init's 418.0. Two agreeing early readings
# were still 160 points wrong. Any anchor here sourced from a submission less
# than a few hours old should be treated as provisional, and any rho computed
# from it inherits that: this one moved -0.500 -> +0.400 -> +0.000 as the single
# number settled.

# Which anchors WE measured on the real ladder vs which are third-party claims.
# The tb_* catalog mu values, the wmh_* README/doc numbers and the notebook-title
# claims are all SELF-REPORTED and unverified by us -- they are alleged, not read.
# Correlations computed over them inherit that uncertainty.
VERIFIED_ANCHORS = {
    "improved_prob_main", "agent_core_improved", "il_agent", "s2_e1_s43",
    "ppo_u120832", "il_alldays_3ep@mega_lucario_ex", "mcts_il_agent",
    "il_bc_3ep@mega_lucario_ex", "il_alldays_3ep@marnies_grimmsnarl_ex",
    "selfplay_g1_ref430k", "selfplay_g3_final",
}


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):  # mean rank for ties
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mean = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mean
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def permutation_p(xs: list[float], ys: list[float], observed: float, n_perm: int = 20000) -> float:
    import random

    rng = random.Random(42)
    ys2 = list(ys)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(ys2)
        if abs(spearman(xs, ys2)) >= abs(observed):
            hits += 1
    return (hits + 1) / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--result", type=Path,
                    default=REPO / "reports" / "wmh_pool_calibration.json",
                    help="benchmark result JSON (needs a 'glicko' block)")
    ap.add_argument("--verified-only", action="store_true",
                    help="use ONLY anchors we read off the ladder ourselves, dropping the\n"
                         "self-reported tb_*/wmh_*/notebook claims")
    args = ap.parse_args()

    res = json.loads(args.result.read_text())
    glicko = res["glicko"]

    rows = []
    for name, g in glicko.items():
        if args.verified_only and name not in VERIFIED_ANCHORS:
            continue
        if name in LADDER_ANCHORS:
            ladder, src = LADDER_ANCHORS[name]
            rows.append((name, g["rating"], g["rd"], ladder, src))
    rows.sort(key=lambda r: -r[3])

    print(f"{len(rows)} agents with BOTH a local Glicko (this run) and a ladder reading:\n")
    print(f"{'agent':40s} {'local glicko':>14s} {'RD':>6s} {'ladder':>8s} {'src':>9s}  provenance")
    for name, rating, rd, ladder, src in rows:
        tier = "VERIFIED" if name in VERIFIED_ANCHORS else "alleged"
        print(f"{name:40s} {rating:>10.1f} {'+/-':>3s}{rd:>4.0f} {ladder:>8.1f} {tier:>9s}  {src}")

    if len(rows) < 4:
        print("\nToo few overlapping agents for a correlation; wire more ladder-scored anchors.")
        return

    local = [r[1] for r in rows]
    ladder = [r[3] for r in rows]
    rho = spearman(local, ladder)
    p = permutation_p(local, ladder, rho)
    print(f"\nSpearman rho (local Glicko vs ladder) = {rho:+.3f}  "
          f"(n={len(rows)}, permutation p={p:.4f})")
    n_ver = sum(1 for r in rows if r[0] in VERIFIED_ANCHORS)
    print(f"Anchor provenance: {n_ver} VERIFIED (we read them off the ladder), "
          f"{len(rows) - n_ver} alleged (self-reported by their authors, unverified).")
    print("Read with the header caveats: mixed-era ladder readings, ~+/-100 same-build drift.")

    # The two most decision-relevant slices: does the pool order OUR agents right?
    ours = [r for r in rows if r[0] in ("improved_prob_main", "agent_core_improved",
                                        "il_agent", "s2_e1_s43", "ppo_u120832")]
    if len(ours) >= 3:
        rho_ours = spearman([r[1] for r in ours], [r[3] for r in ours])
        print(f"Ours-only rho = {rho_ours:+.3f} (n={len(ours)}) -- the slice the "
              f"2026-08-02/03 inversions live in.")


if __name__ == "__main__":
    main()
