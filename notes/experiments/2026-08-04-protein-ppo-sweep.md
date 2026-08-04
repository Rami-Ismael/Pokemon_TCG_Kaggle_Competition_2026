# Protein hyperparameter sweep over Stage-3 PPO (design: hyperparameter sweep)

Design: **hyperparameter sweep**, not an ablation — the question is *sensitivity
of the operating point*, not attribution of components. Optimizer: PufferLib
3.0's `Protein` (the sweep algorithm shipped with the exact trainer we use,
`.venv-ppo`), a cost-aware GP/Pareto local search. Its
`seed_with_search_center=True` behavior makes **run #1 exactly the incumbent
config** — the baseline arm falls out of the sweep for free.

- **Hypothesis:** The incumbent Stage-3 config (lr 3e-5, ent_coef 0.001,
  kl_coef 0.05, update_epochs 3, minibatch 512, gae_lambda 0.95) is not at a
  local optimum: Protein search over these 6 knobs will find a config whose
  60k-step policy beats the incumbent-config 60k-step policy head-to-head.
  Mechanism: PPO fine-tuning from a BC prior lives on a narrow ridge between
  "KL/lr too conservative → never leaves the prior" and "too aggressive →
  mirror-exploitation collapse"; hand-picked values were never searched, so
  the ridge top is likely elsewhere.
- **Independent variable:** the 6-knob hyperparameter vector suggested by
  Protein (one axis: "config"). Everything else fixed: init + KL prior =
  `models/s2/e1_seed43`, 8×8 env topology, bptt_horizon 128, gamma 1.0,
  vf_coef 0.5, max_grad_norm 1.0, promotion gate OFF, total_timesteps 60,000
  (equal steps, per repo rule 4), seed 42, same league mix, same (Lucario
  mirror) deck.
- **Search space** (center = incumbent):
  | knob | distribution | min | center | max |
  |---|---|---|---|---|
  | learning_rate | log_normal | 3e-6 | 3e-5 | 3e-4 |
  | ent_coef | log_normal | 1e-5 | 1e-3 | 3e-2 |
  | kl_coef | log_normal | 5e-3 | 5e-2 | 5e-1 |
  | update_epochs | int_uniform | 1 | 3 | 4 |
  | minibatch_size | uniform_pow2 | 128 | 512 | 1024 |
  | gae_lambda | logit_normal | 0.8 | 0.95 | 0.99 |

  kl_coef=0 is deliberately excluded: it switches trainer class (stock
  PuffeRL, no anchor) — that is the separate KL-ablation arm, not a point on
  this surface. minibatch max 1024 = batch_size (8 envs × 128 horizon), the
  PuffeRL hard constraint.
- **Baseline:** sweep run #1 (Protein's search center) = incumbent config at
  60k steps. NOT ppo_u120832 (121k steps — unequal-steps comparison).
- **Metric & protocol:** per run — win rate over decisive games, 50 mirrored
  pairs (100 games) vs frozen `s2/e1_seed43`, `promotion.evaluate_gate`
  protocol (both sides sample at T=1.0, CPU workers). Protein observes
  score = that win rate, cost = training wall-clock seconds. ~20 runs total,
  sequential (MPS is single-tenant). Runs that stop before 60k steps (crash,
  --max-seconds cap 3600s) are observed as failures, never scored.
- **Pre-registered decision:** after the sweep, play best-config checkpoint
  vs run-#1 (incumbent-config) checkpoint directly, 150 mirrored pairs (300
  games). **Adopt** the swept config as the operating point for the next
  full-length Stage-3 run iff the 95% Wilson CI of its decisive-game win rate
  excludes 0.5. **Drop** otherwise — and record "operating point robust at
  60k scale" as the finding. Secondary deliverable either way: the
  sensitivity picture (score vs each knob from the sweep log).
- **Cost estimate:** 60k steps at ~60-75 steps/s ≈ 13-17 min + ~1 min startup
  + ~2-4 min eval ≈ ~20 min/run → **~6.5 h for 20 runs**, + ~15 min
  validation head-to-head. Disk ~55 MB/run ≈ 1.1 GB (98 GB free). MPS:
  sweep launcher waits for the in-flight `train_il` Hub run (started 12:06,
  ~2 h expected) before its first training run; runs chain, never overlap.
- **Prior work checked:** `notes/scores.md` (ppo_u120832: 87.5% local vs
  s2_e1_s43 but a confirmed local-vs-ladder INVERSION — see caveat),
  `notes/exploiter_experiment_state.md` (~75 steps/s throughput), PufferLib
  3.0 `config/default.ini` [sweep] ranges (their canonical lr/ent/gae bands,
  rescaled to our operating point), ADR-metamon-grid memory (rescale grids to
  our budget, don't copy). No prior hyperparameter search over Stage-3 PPO
  exists in notes/.

**Scope caveat (pre-registered):** the sweep metric is local. Within-family,
same-checkpoint-lineage comparisons are the one class of local ordering that
has held up on the ladder, so it can choose *among PPO configs* — but a sweep
"win" earns only a full-length run whose submission goes through the
`leaderboard-check` skill; it is not a ladder claim. Single seed per sweep
run (seed 42, shared): treat per-run scores as provisional; the adoption
decision rests on the 300-game validation, not on any single sweep score.

## Result (fill after)
- **Observed:**
- **Decision:**
- **What we learned:**
