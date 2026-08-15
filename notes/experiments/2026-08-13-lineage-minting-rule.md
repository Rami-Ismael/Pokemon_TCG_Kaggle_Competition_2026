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

## Result (filled 2026-08-15)
- **Observed:** Both arms ran 1 generation (1000 games, p_opt 0.5, identical
  OSFP draws: 480 mirror / 272 bc / 248 binaryadv; fallback 0.03% both) +
  0.25-epoch binary-advantage resume (184,887 steps, LR fully annealed).
  Arm A's tryout did NOT fire: 47W/53L = 0.470 vs the >0.70 bar, lineage
  stayed at 2; arm B minted unconditionally, lineage 3. Anchored-pool
  battery (50 games/pairing, 400 games/agent, same Ogerpon deck, fallback
  tracking on): arm A 73.2% ± 2.2 field WR (fresh Glicko 1710.4, RD 90.5),
  arm B 75.5% ± 2.2 (1671.2, RD 75.9) — intervals overlap on both metrics.
  Neither arm separates from their shared init binaryadv (72.5% ± 2.2);
  per §3 that reads "no measurable gain from one self-play generation".
  bc_alldays52 baseline 62.7% ± 2.4 (its first battery row was 100%
  fallback — missing worktree model/, caught by the diagnostic, rerun clean).
- **Decision:** SHIP CADENCE — both pre-registered clauses fired
  independently (overlapping intervals; tryout never fired). The cadence
  arm's checkpoint (lineage_selfplay_cadence_gen1_seed42) is the v6
  self-play line's ship candidate.
- **What we learned:** The 70% bar stays unfired in the offline-resume
  regime too (cumulative 0/94 PPO tryouts in v4/v5, now 0/1 here) — a
  0.25-epoch resume over human ∪ 1000 self-play episodes moves the policy
  ~nowhere relative to its own reference (47/100 decisive). At one
  generation the rules differ by a single lineage member, so a real
  minting-rule effect needs multi-generation budgets; this run bounds the
  one-generation effect at < ~2σ.
- **Belief update:** Rami's tryout design isn't wrong, it's unreachable at
  this dose — the bar asks for a 70/30 edge that one cheap generation
  never produces. If v7 wants a growing lineage, either mint on cadence or
  drop the bar toward ~55%.
- **Guardrail caveat:** KL-to-imitation-reference was never implemented in
  the driver (grep 0 hits) — NOT computed as pre-registered. Proxy used:
  holdout (2026-08-09) top-1 vs the human corpus: bc .7606 → binaryadv
  .7433 → arms .726, top-3 stable ~.95 — mild monotone drift from the
  human prior, no collapse. Gap noted; wire real KL before v7 relies on it.
- **Scale confound:** one generation, 1000 games, laptop-scale, single
  seed — bounds this minting rule at this dose only; the Metamon
  replication is untouched by any read here.
