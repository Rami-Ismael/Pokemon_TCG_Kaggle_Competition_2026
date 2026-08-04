# Submission scores table

Per the protocol's Gate 5: one row per submission, appended after the public
score is read back from the leaderboard. Never edit a row's score after
first read-back; append a new row instead if a submission is rebuilt.

| Date | Submission ref | Public score | Checkpoint sha256 | Git SHA | Deck | Config | Local result | What changed |
|---|---|---|---|---|---|---|---|---|
| 2026-08-01 | 55162376 | 827.8 | 758dd7bc55fb...884a67a | e42f3bb | Mega Lucario ex | USE_SEARCH=1 (dead — see notes/phase0_gate0_report.md), USE_BC_PRIOR=1 (architecturally unreachable, same reason) | n/a at submit time, retroactively: behaviorally = pure `AdvancedPolicy` heuristic | Baseline to beat, established before this session |
| 2026-08-02 | 55190924 | 397.3 (read 2026-08-03; intra-day read on 08-02 was 437.3 per 55191752's submit msg) | bce726e56bb2...d738e132 | 8cf2dad9 | Alakazam (`ryotasueyoshi_alakazam`) | Pure BC cloning (`il_agent`), no search, no heuristic | 62.5% [38.6,81.5] vs makthanithin_improved_prob (n=16); 62.5% [38.6,81.5] vs rule_baseline (n=16) — distinguishable from same checkpoint's 0.0% on Lucario deck | First deck-embedding-informed submission; see `notes/phase0_gate0_report.md` A0'' + `reports/bc_standalone_deck_test.json` |
| 2026-08-02 | 55190932 | 190.3 (read 2026-08-03; intra-day read on 08-02 was 252.8 per 55191752's submit msg) | bce726e56bb2...d738e132 | 8cf2dad9 | Alakazam (`ryotasueyoshi_alakazam`) | Search (USE_SEARCH=1, real `rank_all()` candidates) + BC prior (USE_BC_PRIOR=1, BC_PRIOR_C=0.75), forced on via main.py | 10.0% [1.8,40.4] vs makthanithin_improved_prob (n=10) — weaker than, and tied with at this N, the no-search control's 20.0% [5.7,51.0] | Companion submission to 55190924, deliberately weaker local evidence, submitted to get a real ladder read on the search+prior hybrid rather than assume the local benchmark predicts correctly |
| 2026-08-02 | 55191752 | 699.0 (read 2026-08-03) | same bundle as 55162376 (submit msg: "unchanged from the build that scored 804.0") | see 55162376 row | Mega Lucario ex | AdvancedPolicy heuristic, USE_SEARCH=0 default | n/a — same-build resubmit | Deliberate resubmit of the top build after the Alakazam ablations. Its twin reads 804.0 → same-build spread ~100 pts, the ladder-variance yardstick |
| 2026-08-02 | 55196434 | 395.0 (read 2026-08-03) | not recorded in submit msg (arm `s2_e1_s43`) | not recorded in submit msg | Alakazam | Stage-2 REWEIGHT winners-only BC (arm E1, seed 43): 3.32M-param option-scoring transformer warm-started from 3-epoch il_agent PRIOR, 12,900 steps @ LR 1e-4 over 822k winner-seat rows | Rung-1 75.9% top-1 held-out day (PRIOR 75.3%, majority 38.1%); Rung-2 pooled 3 seeds beats PRIOR 62.5% [51.0,72.8]; Glicko 1460 in 15-agent round-robin | First Stage-2 arm on the ladder |
| 2026-08-03 | 55215267 | 265.5 (read 2026-08-03 ~16:50 UTC; earlier same-day read 232.1 — live rating) | not recorded in submit msg (run `ppo_u120832`) | not recorded in submit msg | Alakazam (s2 lineage) | Stage-3 self-play: PufferLib 3.0 PPO fine-tune of s2_e1_s43, ~121k on-policy steps, league mix (50% mirror / 30% PRIOR / 20% public trio), lr 3e-5, masked Discrete(48) | Beats s2_e1_s43 head-to-head 87.5% [69.0,95.7], PRIOR 79.2%; Glicko 1478±71 vs 1300±71 (non-overlap); rung-3 clean | First Stage-3 (RL) submission — and second confirmed local-vs-ladder inversion (see below) |
| 2026-08-04 | 55248781 | 600.0 (read 2026-08-04 ~19:55, first reading — live rating, same-build spread ~±100) | bce726e56bb2...d738e132 (same il_agent checkpoint as 55190924) | 5b0cf63 | Grimmsnarl ex (il_agent's list) | IL-prior MCTS via official Search API: SEARCH_COUNT=30, determinized lookahead (public counts only), prior-only (critic failed calibration 0.266 vs 0.250 and was not shipped) | Beats plain il_agent 67.2% [58.4,75.0] pooled n=119 decided (2 independent post-fix runs), 0 fallbacks/15.8K decisions; 209 ms/decision forced-CPU | Best il-family ladder read yet (il_agent lineage was 397–400): local gain TRANSFERRED, +200 over base — contrast with the 08-02/03 inversions. Displaced the 804-family active resubmit (683–696) per plan; recovery = same-build 804 resubmit. Reports: search_prior_phase3.md |

## Read-back 2026-08-03 (`scripts/check_leaderboard.py`)

- **Standing: rank 5408 / 6202, team score 395.0.** The leaderboard counts only
  **2** submissions for the team — recent submissions displace older ones from
  the active set, so the 804.0 agent no longer represents us. The two
  experimental submissions of 08-02/08-03 cost ~4,500 ranks (was ~865 at 804.0).
- **Local-vs-ladder residual, the number this table exists to produce:** the
  benchmark pool is **not predictive across method families**. Two confirmed
  inversions: improved_prob_main (local Glicko 1720, ladder 701.6 vs 804.0 for
  the agent it "beat") and ppo_u120832 (87.5% head-to-head over s2_e1_s43
  locally, 232.1–265.5 vs 395.0 on the ladder). The one ordering that held
  (55190924 > 55190932) was a same-checkpoint ablation.
- **Scores drift after posting** (live ratings): 55162376 827.8 → 804.0;
  55190924 437.3 → 397.3; 55190932 252.8 → 190.3; 55215267 232.1 → 265.5 within
  hours. Same-build spread observed ~±100 (804.0 vs 699.0). A row records the
  score at its stated read time; per protocol, never edited afterward.
- Baseline to beat remains **804.0** (55162376 lineage). Nothing since has
  approached it. Process guard now lives in the `leaderboard-check` skill +
  `scripts/check_leaderboard.py`; snapshots append to
  `reports/leaderboard_history.jsonl`.
