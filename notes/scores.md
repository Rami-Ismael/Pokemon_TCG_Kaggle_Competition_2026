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
| 2026-08-03 | 55219194 | 685.5 (read 2026-08-03 evening) | 758dd7bc55fb...884a67a (byte-identical to 55162376) | e42f3bb | Mega Lucario ex | AdvancedPolicy heuristic (USE_SEARCH=1/USE_BC_PRIOR=1 present but dead) | n/a — same-build resubmit | Restored the 804.0 build to the active set after 08-02/08-03 experiments displaced it (team had fallen to 395.0 / rank ~5408) |
| 2026-08-03 | 55224682 | 600.0 (read 2026-08-04 ~01:30 UTC, first read after COMPLETE) | 758dd7bc55fb...884a67a (byte-identical to 55162376; tarball sha256 3b90d8c8...9b28) | e42f3bb | Mega Lucario ex | AdvancedPolicy heuristic (USE_SEARCH=1/USE_BC_PRIOR=1 present but dead) | n/a — same-build resubmit | Filled the second active slot, displacing ppo_u120832 (275.1) so both slots hold the best build. Lowest same-build roll yet: lineage now 827.8→804.0, 699.0, 685.5, 600.0 — observed same-build spread widened to ~230 pts |

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

## Read-back 2026-08-04 ~01:30 UTC (after 55219194 + 55224682)

- **Standing recovered: rank 2496 / 6215, team score 685.5** (was 5408 @ 395.0
  before the two same-build resubmits). Both active slots now hold the
  804.0-lineage bundle; the team score rests on 55219194's 685.5 roll.
- **Same-build spread is wider than the earlier ±100 estimate.** Four rolls of
  the byte-identical bundle: 827.8→804.0, 699.0, 685.5, 600.0 — a ~230-pt
  range. Any single-submission ladder delta inside ~±150 of a baseline roll is
  noise; treat only repeated reads or margins beyond that as signal.
- Top-8 cutoff 1122.3; gap from our team score +436.8 — beyond what same-build
  rerolling can close. Closing it needs a genuinely stronger agent, and the
  local benchmark pool is not currently predictive across method families.

## Submitted 2026-08-05 ~13:10 UTC — 55270787, deck-switch test (score PENDING)

- **What:** `il_alldays_0804` (all-days BC, 127,748 steps) piloting **Marnie's
  Grimmsnarl ex** instead of Mega Lucario ex. Full detail:
  `submission_ledger.py show --ref 55270787`; study in
  `reports/il_model_deck_selection.md`.
- **Why it is a clean test:** 55248985 is the *same checkpoint* on Mega Lucario ex
  and settled at **418.0**. The ONLY variable changed is the deck, so this isolates
  the deck effect against a real ladder baseline rather than a local proxy.
- **Local evidence:** 86.6 ± 1.9% (320 games) on Grimmsnarl ex vs 55.0 ± 3.9% for
  identical weights on Lucario, against an 8-agent pool anchored to ladder
  532–1196. Pooled Glicko 1877.1 vs 1551.2. The shipped `il_bc_3ep` on Lucario is
  28.8 ± 3.6% — a 544-Glicko spread from the same weights on a different deck.
  0 silent fallbacks in 4,694 measured decisions.
- **Pre-registered prediction: >450.** Every prior BC-family submission landed
  395–450 (il_agent 397.3/400.0, s2_e1_s43 395.0, 55248985 418.0). Landing in that
  band again falsifies the transfer and would be the 4th local-vs-ladder divergence.
- **Displaces** 55248985 (418.0); active set becomes this + 55253900 (267.4, a
  concurrent session's self-play candidate). Team = max(this, 267.4), so this must
  clear 418.0 to avoid a net loss. Best-ever 804.0 still displaced.
- **Score: PENDING** — fill once, on read-back.
