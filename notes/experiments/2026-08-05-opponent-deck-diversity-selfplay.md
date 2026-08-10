# Opponent-deck diversity in self-play (generalization test)

- **Hypothesis:** Sampling the opponent's deck per episode from a ~20-deck meta
  pool during self-play RL will raise the learner's win rate against decks it
  never trained against, because the current opponent distribution is ~80%
  Lucario-mirror (mirror 50% + league checkpoints 30%, all piloting il_agent's
  Lucario deck; the remaining 20% is only 3 public decks), so the policy
  overfits to Lucario-mirror dynamics. Supporting prior: the exploiter result —
  a policy trained on narrow data was 95% exploitable in the Lucario mirror but
  the exploit died when training data got diverse
  (exploit-transfers-across-seeds-not-across-data).
- **Design:** Controlled comparison + generalization test (held-out decks).
  Not an ablation — the question is "does adding X help off-distribution",
  not attribution among existing components.
- **Independent variable:** The opponent deck distribution during continued
  self-play. Arm A (control): continue the best RL checkpoint, incumbent
  opponent mix, unchanged decks. Arm B: identical steps/config/seeds, but
  checkpoint-opponents' decks sampled uniformly per episode from the TRAIN
  deck pool. Learner's own deck stays fixed (il_agent Lucario) in BOTH arms —
  piloting diverse decks is a separate variable for a separate experiment.
- **Baseline:** `models/selfplay_g1/u430080` — the promoted generation-1
  anchored-self-play checkpoint (PR #42's ref430k, submitted as 55253900,
  ladder read 267.4 on 2026-08-05), continued for the same number of extra
  steps as arm B (equal-steps rule).
- **As-built protocol (2026-08-05):** the deck-pool machinery from the
  08-04 exploitability sweep is reused; what was added is the opponent-only
  axis (`PTCGGym(opp_deck_pool=...)`, trainer `--opp-deck-pool`, eval
  `--opp-deck-only`, tests in `tests/test_opp_deck_pool.py`). Split is at
  MULTISET identity, not file bytes: the 33 legality-verified refs collapse
  to **27 truly distinct decks** (6 were reorderings — a leak if split
  naively). Pins into train: il_agent's own deck + the 3 module-pool decks
  (kiyotah_dragapult, mechi22_alakazam, plamen06_steel — the base checkpoint
  trained against them, so they can never be "held out"). Seed-42 split of
  the rest → **19 train / 8 holdout**
  (`configs/deck_pools/opp_{train,holdout}.txt`). Module opponents keep
  their bundled decks (their rules are written around their cards); only
  ckpt opponents (mirror + league) are re-decked. Caveat: holdout decks are
  unseen LISTS but partly seen ARCHETYPES (e.g. Alakazam variants exist on
  both sides) — this battery measures list-level, not archetype-level,
  generalization. Both arms: 500k steps, `--mix 0.5,0.3,0.2` (g1's regime),
  league = {il_alldays_0804, g1/u430080}, KL anchor = u430080, seed 42,
  SINGLE seed (provisional). Driver: `scripts/run_deckdiv_experiment.sh`,
  chained behind the hfstream-v2-3ep IL run so MPS is never contended.
- **Metric & protocol:**
  1. Primary: win rate vs a HELD-OUT deck battery — ~8 decks excluded from the
     train pool, piloted by the same frozen opponent policy, ≥50 mirrored
     pairs per deck per arm. Both arms, same seeds, same eval set.
  2. Guardrail: the ladder-anchored 8-agent pool (rho +0.929) — arm B must not
     regress beyond overlapping Glicko RD intervals.
  3. Confound check: run-fallback-diagnostic on both arms before reading any
     number.
- **Pre-registered decision (PROPOSED — Rami to confirm before launch):**
  adopt if arm B beats arm A on the held-out battery by ≥5 pts (outside
  binomial noise at ~800 games, σ≈1.8 pts) AND the anchored pool doesn't
  regress; drop if Δ < 3 pts; inconclusive band 3–5 pts → one more seed.
- **Phase placement:** This experiment IS the "after phase 3" placement
  (fine-tune the existing checkpoint). "Diversity from the start of phase 3"
  is a follow-up arm only if this shows signal — it costs a full phase-3 run,
  so cheapest-first ordering applies.
- **Cost estimate:** code: ~30 lines in puffer_env.py (SamplingPolicy already
  takes a deck param; PTCGGym hardcodes load_deck()). Runs: 2 arms × extra
  steps at current s3 throughput + ~1,600 eval games. Single MPS job at a
  time, chained.
- **Prior work checked:** exploiter diagnostic (57.6% across 33-deck pool =
  the baseline generalization number), exploit-transfers memory,
  deck-matters-more-than-checkpoint (disputed by ladder 08-05),
  identify_opponent_deck.py (exact lists NOT recoverable from public zones —
  but il_dataset.py:778 confirms raw episodes DO record the deck-submission
  step, so mining the ladder meta from the HF corpus is feasible as an
  extension), public-deck-cg-legality memory (any mined/mutated deck must
  pass a battle_start probe before entering the pool).
- **Explicitly deferred:** (a) learner piloting diverse decks — second
  variable; (b) one-card deck mutations — tiny variation relative to 29 real
  decks, uncontrolled opponent-strength change, and cg-legality risk; revisit
  only if the real-deck pool plateaus; (c) ladder-meta frequency weighting
  from episode mining — extension after a positive result.

## Result (2026-08-07)

Training: both arms 500,736 steps from g1/u430080, KL-to-anchor at finish
0.083 (A) / 0.085 (B). Arm B's first attempt was jetsam-killed at 271k under
parallel-session memory pressure and rerun from scratch (not resumed — a
mid-run resume would have given B a second critic cold-start A never had);
metrics of the killed attempt archived at
`reports/deckdiv/armB_attempt1_metrics.jsonl`.

**Holdout battery** (8 held-out decks × 50 mirrored pairs = 800 games each,
frozen il_agent piloting, learner on its fixed Lucario deck,
`reports/deckdiv/holdout_{base,armA,armB}.json`):

| checkpoint | win rate | Wilson 95% |
|---|---|---|
| base g1/u430080 | 0.576 | [0.542, 0.610] |
| arm A (control continuation) | 0.605 | [0.571, 0.638] |
| arm B (opponent-deck pool) | 0.561 | [0.527, 0.595] |

**B − A = −4.4 pts.** Validity: learner fallbacks 0/55,398 (A) and 1/55,227
(B); opponent fallbacks ≤0.05% — not a fallback artifact. Per-deck, B gains
nowhere meaningful and loses most on wmh_mewtwo (−12), pixiux_lucario_v63
(−10), wmh_garchomp (−9). Per-deck spread is huge for every checkpoint
(0.07–0.98); pllinas_alakazam and wmh_alakazam are near-total losses for all
three (~7–18%) — matchup, not checkpoint, dominates those cells, consistent
with the deck-matters-more-than-checkpoint finding.

- **Observed:** opponent-deck diversity during continued self-play did not
  improve held-out-deck generalization; the point estimate moved the wrong
  way (−4.4 pts, below even the +3 drop line; CIs of A and B overlap).
  The control continuation alone gained +2.9 over base (overlapping CIs —
  weak evidence that more anchored self-play helps holdout play at all).
- **Decision: VOIDED by Rami (2026-08-07), superseding the mechanical DROP.**
  The pre-registered rule fired at drop (< +3), but the run has a validity
  hole: arm A trained under heavy CPU contention (~44 sps, parallel-session
  load, jetsam pressure on the box) while arm B's rerun trained on a quiet
  machine (~102 sps). Equal steps, unequal machine conditions — contention
  can starve env workers into timeout losses that poison rollouts (the
  preflight's own warning), so the A-vs-B comparison is confounded and the
  −4.4 is not a trustworthy effect size. The jetsam-killed arm B attempt
  itself contributed no data (discarded, rerun from scratch). What survives:
  the base-generalization reading (57.6% on unseen lists) — all three
  batteries ran on the same quiet machine — and the instrument itself.
  **Reopen condition:** rerun both arms from the new all-episodes BC model
  (il_agent_hfstream_v2_3ep lineage) once it is gated, BOTH arms on a quiet
  machine (preflight enforced, no --skip-preflight), same 19/8 split, same
  battery.
- **What we learned:**
  1. The hypothesis's mechanism was already half-refuted by the 08-04
     competence control (il_agent plays *better* off its own deck):
     board-state generalization across opponent decks appears to come from
     the BC corpus's diversity, not from self-play opponent variety. Adding
     opponent-deck randomness at 500k-step scale under a 0.05 KL leash
     bought nothing and may have diluted the mirror/league signal.
  2. Transferable: when a diversity intervention targets a failure mode
     (overfitting to one opponent distribution), first measure whether the
     failure mode exists at the intervention point — the base checkpoint
     already sat at 57.6% on unseen lists, so there was less headroom than
     the "80% Lucario mirror" framing implied. The cheap pre-experiment
     (evaluate base on holdout FIRST, before training any arm) would have
     sized the opportunity honestly.
  3. The battery instrument is cheap (~90 s for 800 games on a quiet
     machine) — holdout-deck evaluation should be a standing gate for every
     future checkpoint, regardless of this arm being dropped.
