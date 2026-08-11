# IL policy + UCB1 depth-1 re-rank (controlled comparison, add-one-in)

- **Hypothesis:** Re-ranking the IL policy's top-8 MAIN-decision candidates with
  the repaired UCB1 depth-1 bandit (real-engine one-turn rollouts scored by
  `evaluate_state`) beats the same IL policy without search, because the engine
  rollout catches tactical blunders (bad attacks, wasted energy, lethal
  oversights) that one-shot logits miss.
- **Independent variable:** the search layer only — on vs off. Same checkpoint,
  same deck, same everything else.
- **Baseline:** `bc_alldays52_jun16_aug07_seed42` (current best BC, all-52-days
  corpus), bundled Mega Lucario ex deck, no search.
- **Arm:** `agents/search_arms/bc_alldays52_ucb1_rerank/` — same checkpoint;
  MAIN decisions with ≥2 options route through `flat_monte_carlo_search`
  (improved_probabilistic lineage, repaired 2026-08-11) with `base_order` =
  argsort of the IL logits; candidates = top 8; budget 1.5 s; rollout policy =
  the deck-matched Mega Lucario heuristic; leaf = `evaluate_state`. Non-MAIN
  decisions and every fallback path are byte-identical BC behavior.
- **Metric & protocol:** staged.
  - **Stage 1 — mechanism gate (pre-registered from the tb-tree negative's
    reopen condition):** ≥200 real MAIN decisions sampled from BC-vs-BC games;
    measure decision-change rate = fraction where search top-1 ≠ IL top-1.
    Also record sims/decision and the BC fallback counters (rule out the
    silent-fallback confound).
  - **Stage 2 — strength (only if Stage 1 passes):** 25 mirrored pairs
    (50 games) arm vs control, head-to-head. Head-to-head is the cheapest
    falsifier; a win here still needs the anchored pool + `leaderboard-check`
    before any "better" claim. Labeled subset — not comparable to pool numbers.
- **Primary metric:** Stage 2 win rate with binomial σ.
- **Guardrail metric:** root-decision agreement with the IL policy
  (= 1 − decision-change rate, from Stage 1) — search overriding the teacher
  on most decisions while winning would be a different finding than a
  light-touch correction.
- **Pre-registered decision:**
  - Stage 1: decision-change rate < 10% → **drop** (search is decoration; do
    not spend games). ≥ 10% → proceed to Stage 2.
  - Stage 2: arm ≥ 60% (30/50) → adopt for anchored-pool testing;
    ≤ 50% → drop; in between → inconclusive, do not scale up without a
    leaf-value fix.
- **Cost estimate:** Stage 1 ≈ 10 min CPU. Stage 2 ≈ 1.5–2 h CPU (search adds
  ≤1.5 s × ~50 MAIN decisions/game). No MPS use (inference on CPU), no disk.
- **Prior work checked:** memory `search-bc-prior-no-measurable-gain` (BC as
  PUCT exploration bonus on the heuristic's candidates: null, 900 games — NOT
  this design; there the heuristic stayed the ranker), `search-prior-arm-pool-
  negative` (evaluate_state's rollout scorer named the broken part), `il-mcts-
  tb-tree-negative` (IL prior on the tb tree: null ×4; its reopen condition is
  the ≥10% mechanism gate used here), `prior-only-mcts-is-decoration`,
  `search-leaf-value-must-be-centered`, `il-prior-mcts-beats-il-agent-locally`
  (3rd local-vs-ladder inversion — no transfer claims from local numbers),
  agent_core_improved's live-search benchmark (repaired bandit lost 35.0% to
  its own heuristic — leaf blamed). What is NEW here: the IL model is the
  candidate generator itself, on the freshly repaired depth-1 bandit.
- **Stated risk / attribution scope:** if Stage 2 is negative, the supported
  conclusion is "the evaluate_state leaf loses the trades the IL ranker sets
  up", not "search can't help IL". The clean follow-up would swap the leaf
  (centered `critic_trainday` through the same seam), one variable at a time.
- **Scale confound:** the base checkpoint is trained on the full 52-day corpus
  (no longer the 9-day regime), and the search layer is inference-time only —
  a negative here mostly survives the scale confound, except that
  `evaluate_state` itself was hand-tuned against weaker opponents.
- **Deck scope:** everything is the Mega Lucario ex mirror (both arms, and the
  rollout heuristic is deck-matched by design). The meta-deck (Grimmsnarl)
  cell stays empty — per `all-search-and-rl-measured-on-lucario`.

## Stage 1 result (2026-08-11)
- **Observed:** decision-change rate **188/200 = 94.0% ± 1.7%** (1 SE), on 200
  MAIN decisions from 2 BC-vs-BC control games. Mechanism fully live: mean
  5.8 candidates and ~4,469 sims per decision, 1.50 s/decision, 0 IL-rank
  failures, 0 search failures, 0 BC fallbacks.
  Command: `scratchpad/stage1_mechanism_gate.py` (session 2026-08-11).
- **Gate verdict:** PASS (≥10% pre-registered) → Stage 2 proceeds.
- **Guardrail reading (bad):** 94% override means the arm is NOT "IL with
  tactical corrections" — it is evaluate_state-driven play with IL candidate
  pruning. The teacher's ranking survives only 6% of searched decisions.
  Prediction going into Stage 2, from the sibling lineage's evidence (bandit
  overriding a strong policy via this same leaf lost 35.0%): negative.

## Result (fill after)
- **Observed:**
- **Decision:**
- **What we learned:**
- **Belief update:** <Rami's one-liner, not a paraphrase>
