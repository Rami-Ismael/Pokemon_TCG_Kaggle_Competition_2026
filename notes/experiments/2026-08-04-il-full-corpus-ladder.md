# IL retrained on the full Hub corpus — third point on the data-scaling curve, ladder-adjudicated

Design: **controlled comparison + scaling-study point** (not a sweep — two
smaller-corpus points already exist, this adds the third and takes the ladder
measurement the middle point never got). Proposed by Rami 2026-08-04
("retrain the IL model with all the days so far and release to Kaggle…
I expect the performance to increase dramatically"), coached into this card.

- **Hypothesis:** behavior cloning on ALL 15,032 Hub train episodes (8 days,
  07-01 → 08-03) beats the 4,554-episode PRIOR on the real ladder, because
  (a) 3.3× more human data reduces overfitting and (b) the added days are
  RECENT — they cover the current meta the ladder actually plays, which the
  July-only corpus does not.
- **Independent variable:** training-corpus size/recency (4,554 → 15,032
  episodes). Architecture (192/6/6, 3.32M), recipe (fresh init, 3 epochs,
  lr 3e-4 cosine, batch 64, seed 42), encoder, and deck all held fixed.
- **Baseline:** `models/il_agent` (PRIOR) — ladder-settled **400.0**
  (submission 55149903). Middle point: `il_agent_hfstream_combined_3ep`
  (9,820 eps): offline accuracy 0.7527 vs PRIOR's 0.7534 — **2.16× data
  moved offline accuracy 0.0 points** (never ladder-tested).
- **Metric & protocol:** (1) offline accuracy on the held-out day — expect
  ~flat, recorded but NOT a decision metric; (2) fallback diagnostic must be
  clean (hard rule — a silent-fallback regression would invalidate the
  comparison); (3) quick local head-to-head vs PRIOR (50 mirrored pairs) for
  the record — local results do not gate the submission, which Rami ordered
  regardless; (4) **the decision metric: Kaggle score after ~3-day settle**
  (first reads carry ±150-point noise, measured).
- **Pre-registered decision:** "dramatic increase" (Rami's expectation) =
  settled **≥ 500**. Real-but-modest = settled clearly above 400. Null/
  refuted = settled ≤ 400. Claude's registered prediction, for calibration:
  modest at best — the 2.16× step moved offline accuracy 0.0, and 1.53× more
  is small on a log axis; the live mechanism to watch is data RECENCY (which
  offline accuracy on a July eval day cannot see, and the ladder can).
- **Scope guard (Rami's explicit instruction):** the unrated new days do NOT
  enter the skill-gate arm (e3 stays on rated 07-01+07-26 data).
- **Cost estimate:** ~127,500 steps (2.72M rows/epoch × 3 / batch 64) ≈
  2.5–3 h MPS, queued behind the arm tournament; ~13 MB checkpoint; one
  Kaggle submission slot — **displaces one of the two active 804-lineage
  heuristic slots (currently reading 670.4 / 600)** for ~days.
- **Prior work checked:** hfstream_combined_3ep metadata (the flat 2.16×
  point); rl_pipeline_v2 corrections table (first-read noise, settle rule);
  Metamon paper (data scale is their central axis — but across orders of
  magnitude, not +0.18 dex); ADR-001 (hub streaming = the corpus path).

## Result (updating as readings land)
- **Observed so far (2026-08-04):** training complete (127,748 steps).
  Offline accuracy **0.7583** — above PRIOR (0.7534) and the 2-day point
  (0.7527): the scaling curve's third point bends UP where the second was
  flat (likely the recency mechanism; +0.5pt, small but the first offline
  gain data has bought this architecture). Independent robustness datapoint
  from the exploiter session: their PPO exploiter wins 95.5% vs PRIOR and
  92–97% vs unfamiliar imitation seeds, but only **38.5% [32.0, 45.4]** vs
  this model — far less exploitable. Submitted as **55248985** (by the
  concurrent session; this session's duplicate attempt bounced on the
  daily limit — ledger-mediated dedup). First read 600.0. NOTE: its two
  submissions displaced continued-imitation (55246108) from the active set
  after ~3h (readings 600 → 320.4, truncated settle — weak evidence).
- **Decision:** pending the ~08-07 settle vs the pre-registered bars
  (≥500 dramatic / >400 modest / ≤400 null). Provisional Stage-3 base
  switched to THIS model (robustness evidence + it holds an active slot).
- **What we learned (so far):** offline accuracy CAN move with data at
  this scale once recency enters; and exploitability — not offline
  accuracy — separated the imitation models most sharply.
