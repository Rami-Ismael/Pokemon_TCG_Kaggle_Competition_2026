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

## Result (fill after)
- **Observed:**
- **Decision:**
- **What we learned:**
