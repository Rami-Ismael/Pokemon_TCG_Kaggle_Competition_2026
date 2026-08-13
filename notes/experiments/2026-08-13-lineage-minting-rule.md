# Lineage minting rule: 70% tryout vs fixed cadence (controlled comparison)

- **Hypothesis:** In v6 lineage self-play (rl_pipeline_v6.md §1.2), minting a
  checkpoint into the lineage only when it wins >70% of decisive tryout games
  against the newest lineage member produces a stronger final agent than
  minting on a fixed cadence, because copies enter the fictitious-play mixture
  only when they represent real progress, so training never spends games
  against redundant opponents.
- **Independent variable:** the minting rule alone — `--mint-rule tryout`
  (arm A) vs `--mint-rule cadence` (arm B) in
  `scripts/run_lineage_selfplay.py`. Same initial lineage, same deck pool
  (both seats sampled), same `p_opt`, same games per generation, same resume
  training config, same seed.
- **Baseline:** arm B (cadence) — it is also the v5 precedent
  (unconditional retarget after the 70% ratchet fired 0/31).
- **Metric & protocol:** each arm's final checkpoint plays the anchored
  8-agent pool (the rho +0.929 pool), ≥50 games per pairing, paired decks;
  Glicko with RD. Fallback diagnostic (`run-fallback-diagnostic`) on both
  checkpoints before interpreting anything.
- **Primary metric:** Glicko vs the anchored pool.
- **Guardrail metric:** KL to the never-updated imitation reference
  (`bc_alldays52_jun16_aug07_seed42`) on a fixed eval batch, per generation
  (v6 §5) — a strength gain that torches the human prior is a different
  result than a clean gain.
- **Pre-registered decision (committed by Rami 2026-08-13):** adopt the 70%
  tryout rule only if arm A's final agent beats arm B's on the anchored pool
  with non-overlapping Glicko intervals. If intervals overlap, OR the tryout
  never fires in arm A, ship cadence (simpler, v5 precedent). Built-in early
  read: if the tryout fires zero times in the first half of arm A's budget,
  the 0/31 precedent has repeated and arm A has collapsed to "no lineage
  growth" — that alone answers the question at half cost.
- **Design note:** controlled comparison, not an ablation — the question is
  which trigger, an attribution question between two rules, not whether the
  lineage pool matters at all (that would be a separate mirror-only vs
  growing-lineage arm).
- **Cost estimate:** 2 × (one generation of self-play games + one train_il
  resume). Games run on CPU via the direct battle API; resume runs on MPS —
  chain the arms, never overlap MPS jobs. Disk: episodes are written straight
  to zstd parquet shards (no raw JSON unless --keep-raw), so one generation
  is a few hundred MB, inside the ~45 GB free budget.
- **Scale confound:** any negative result here is a 9-days-of-data datum on a
  laptop-scale run; it constrains the minting rule at this scale only, not
  the method. The Metamon replication is not up for review in this
  experiment.
- **Prior work checked:** rl_pipeline_v6.md §1.2/§3 (growing lineage is plan
  of record; no minting trigger committed); v4/v5 promotion logs (70% ratchet
  fired 0/31 at the PPO+KL config — different regime, offline resume may
  differ); notes/experiments/2026-08-05-selfplay-deck-variation.md
  (lineage-only league decision); selfplay-mechanism-taxonomy and
  ByteRL/OSFP (arXiv:2303.04096).

## Result (fill after)
- **Observed:**
- **Decision:**
- **What we learned:**
- **Belief update:**
