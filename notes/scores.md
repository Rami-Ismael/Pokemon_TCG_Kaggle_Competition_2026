# Submission scores table

Per the protocol's Gate 5: one row per submission, appended after the public
score is read back from the leaderboard. Never edit a row's score after
first read-back; append a new row instead if a submission is rebuilt.

| Date | Submission ref | Public score | Checkpoint sha256 | Git SHA | Deck | Config | Local result | What changed |
|---|---|---|---|---|---|---|---|---|
| 2026-08-01 | 55162376 | 827.8 | 758dd7bc55fb...884a67a | e42f3bb | Mega Lucario ex | USE_SEARCH=1 (dead — see notes/phase0_gate0_report.md), USE_BC_PRIOR=1 (architecturally unreachable, same reason) | n/a at submit time, retroactively: behaviorally = pure `AdvancedPolicy` heuristic | Baseline to beat, established before this session |
| 2026-08-02 | 55190924 | **PENDING** | bce726e56bb2...d738e132 | 8cf2dad9 | Alakazam (`ryotasueyoshi_alakazam`) | Pure BC cloning (`il_agent`), no search, no heuristic | 62.5% [38.6,81.5] vs makthanithin_improved_prob (n=16); 62.5% [38.6,81.5] vs rule_baseline (n=16) — distinguishable from same checkpoint's 0.0% on Lucario deck | First deck-embedding-informed submission; see `notes/phase0_gate0_report.md` A0'' + `reports/bc_standalone_deck_test.json` |
| 2026-08-02 | 55190932 | **PENDING** | bce726e56bb2...d738e132 | 8cf2dad9 | Alakazam (`ryotasueyoshi_alakazam`) | Search (USE_SEARCH=1, real `rank_all()` candidates) + BC prior (USE_BC_PRIOR=1, BC_PRIOR_C=0.75), forced on via main.py | 10.0% [1.8,40.4] vs makthanithin_improved_prob (n=10) — weaker than, and tied with at this N, the no-search control's 20.0% [5.7,51.0] | Companion submission to 55190924, deliberately weaker local evidence, submitted to get a real ladder read on the search+prior hybrid rather than assume the local benchmark predicts correctly |

## To do when scores post
- Read back public score for 55190924 and 55190932 (`kaggle competitions submissions -c pokemon-tcg-ai-battle`).
- Compare against the 827.8 baseline and against each other.
- If either beats 827.8, that is the new baseline for Phase 5/6 going forward.
- Record the local-Glicko-vs-ladder residual once both scores are in — this is
  the number that tells us whether the Phase 1 benchmark pool is actually
  predictive, per the protocol's own framing.
