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
  self-play. Arm A (control): continue current best RL checkpoint, incumbent
  opponent mix, unchanged decks. Arm B: identical steps/config/seeds, but
  checkpoint-opponents' decks sampled uniformly per episode from the TRAIN
  deck pool (~21 of the 29 distinct decks on disk). Learner's own deck stays
  fixed (il_agent Lucario) in BOTH arms — piloting diverse decks is a separate
  variable for a separate experiment.
- **Baseline:** Current best s3 checkpoint, continued for the same number of
  extra steps as arm B (equal-steps rule).
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
