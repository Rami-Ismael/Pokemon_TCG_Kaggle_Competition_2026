# Batch size vs training throughput (scaling study, stage 1 of 2)

- **Hypothesis:** Raising the BC training batch size above 64 raises training
  throughput (rows/sec) on MPS, because at batch 64 the step is
  kernel-launch-bound and larger batches amortize that overhead. It fails if
  the streaming DataLoader (4 workers over hub shards) is the bottleneck, in
  which case rows/sec plateaus regardless of batch size.
- **Independent variable:** batch size ∈ {64, 128, 256, 512, 1024}; everything
  else fixed (model 192/6/6 as in production BC, train_combined_v4 corpus,
  4 workers, seed 42, same shard stream per arm).
- **Baseline:** batch 64 (current train_il.py default; measured 10.86 steps/s
  ≈ 695 rows/s on this model per train_il.py's own header note).
- **Metric & protocol:** rows/sec for (a) loader-only pass (dataloader
  ceiling) and (b) full train pass (forward+backward+AdamW step), 30 warmup
  steps discarded, ≥20 steps and ≥12,800 rows timed per arm; peak RSS of the
  process tree sampled throughout. Command:
  `uv run python scripts/benchmark_il_throughput.py --out <json>`.
  Machine must be otherwise idle (no MPS contention) or the numbers are void.
- **Primary metric (stage 1):** train rows/sec vs batch size curve. This
  stage makes no strength claim; strength enters only in stage 2.
- **Guardrail metric (stage 2, only if stage 1 passes):** eval loss / top-1 at
  EQUAL WALL-CLOCK vs the batch-64 baseline, LR scaled linearly with batch —
  non-inferiority guardrail on optimization, not a strength claim. Any
  "stronger agent" claim would additionally need the anchored pool +
  leaderboard-check.
- **Pre-registered decision (committed by Rami 2026-08-13, "run it when stuff
  frees up"):**
  - Proceed to stage 2 with batch B if train rows/sec at B ≥ 1.5× batch 64.
  - Drop the idea if rows/sec at 512 < 1.5× batch 64 → we are
    dataloader-bound; the follow-up knob is num_workers/prefetch, not batch.
  - Adopt B only if stage 2 eval loss at equal wall-clock is within noise of
    baseline.
- **Cost estimate:** stage 1 ≈ 15–25 min wall-clock, no disk growth (HF cache
  already warm), peak RAM measured as part of the result (24 GB machine).
  Stage 2 if funded: 2 runs × ~1 h wall-clock each.
- **Prior work checked:** PPO minibatch sweep negative result (different axis:
  on-policy PPO minibatch, not BC batch); model-size scaling study
  (training-parity confound → equal wall-clock control here); direct-battle
  measurement that MPS is 2.4× faster at learner-sized batches (supports the
  mechanism); train_il.py header's measured 10.86 steps/s at batch 64.

## Result
- **Observed:** (idle machine except one VM at ~0.7 of one core, shared by all
  arms; full data in 2026-08-13-batch-size-throughput-results.json)

  | batch | loader-only rows/s | train rows/s | × vs batch 64 |
  |---|---|---|---|
  | 64 | 17,703 | 882.6 | 1.00 |
  | 128 | 9,875 | 950.5 | 1.08 |
  | 256 | 18,217 | 989.0 | 1.12 |
  | 512 | 20,320 | 932.1 | 1.06 |
  | 1024 | 15,551 | 294.7 | 0.33 |

  Command: `uv run python scripts/benchmark_il_throughput.py` (arms 64–512 in
  one process; both 1024 arms rerun in a fresh process after a macOS libshm
  DataLoader-teardown flake killed the first process on its last arm).
- **Decision:** DROPPED per the pre-registered rule — best arm is 1.12× at
  batch 256, below the 1.5× bar; batch 1024 is a 3× regression. Stage 2 not
  funded. The follow-up worker/prefetch sweep is ALSO dead by the same data:
  the 4-worker loader ceiling is 10–20× above what training consumes at every
  batch size, so training is MPS-compute-bound, not dataloader-bound, and
  more workers raise a ceiling nobody hits.
- **What we learned:** MPS compute on this model (192/6/6, 3.3M params) is
  already saturated at batch 64 — the launch-overhead-amortization mechanism
  was already exhausted, and very large batches actively hurt (the 1024 step
  falls off a throughput cliff). Free side-finding: batch 64 now trains at
  ~883 rows/s (13.8 steps/s) vs the 10.86 steps/s in train_il.py's header —
  warm HF cache + idle machine; use ~13.8 steps/s for wall-clock projections.
- **Scale confound:** none — this is a hardware/pipeline throughput
  measurement, independent of training-data scale. Reopens only on different
  hardware (CUDA), a much larger model, or a changed input pipeline.
- **Belief update (Rami):** _pending — what did this change about your
  beliefs?_

# Follow-up: mixed precision + torch.compile (stage 1b, same harness)

Requested by Rami 2026-08-13 after the batch-size drop ("run the mixed
precision benchmark and add torch.compile").

- **Hypothesis:** bf16 autocast and/or torch.compile raise train rows/sec on
  MPS, because the fp32 eager step is memory-bandwidth- and
  dispatch-overhead-bound (the flat batch curve is consistent with that), and
  half-width activations / fused kernels cut both.
- **Independent variable:** train-step implementation ∈ {bf16, compile,
  bf16+compile}, at batch ∈ {64, 256}; fp32-eager baselines from stage 1.
- **Metric & protocol:** train rows/sec, same harness
  (`benchmark_il_throughput.py --modes train --precision ... [--compile]`),
  one arm per process, warmup 40 steps for compile arms (compile time lands
  in warmup, excluded from timing). torch 2.13.0.
- **Pre-registered decision:** adopt for production training only if some arm
  is ≥1.3× its same-batch fp32 baseline AND (for bf16 arms) a follow-up
  equal-wall-clock quality check shows loss parity — mixed precision changes
  numerics, so a throughput win alone does not change production. Drop
  otherwise. torch.compile failing to compile on MPS is recorded as a result,
  not retried.

## Result
- **Observed:** (fp32-eager baselines from stage 1; × is vs the same-batch
  fp32 baseline)

  | arm | batch 64 rows/s (×) | batch 256 rows/s (×) |
  |---|---|---|
  | fp32 eager (baseline) | 882.6 (1.00) | 989.0 (1.00) |
  | bf16 autocast | 805.6 (0.91) | 786.8 (0.80) |
  | torch.compile | 388.8 (0.44) | 795.3 (0.80) |
  | bf16 + compile | 322.3 (0.37) | 972.3 (0.98) |

- **Decision:** DROPPED per the pre-registered rule — no arm reached 1.3×;
  none even reached 1.0×. The bf16 loss-parity check is moot.
- **What we learned:** fp32 eager is already the fastest configuration for
  this model on MPS/torch 2.13. Autocast's cast overhead and inductor's
  MPS codegen both cost more than they save at 3.3M params. Combined with
  stage 1: the laptop's training ceiling is ~989 rows/s (batch 256, fp32,
  eager, 4 workers) ≈ 13.5 h per epoch over the 53-day corpus. The remaining
  throughput lever is hardware (CUDA), not configuration.
- **Belief update (Rami):** _pending_
- **Adoption note (2026-08-13, Rami's call):** train_il.py defaults changed
  to the fastest measured config — batch 256, 4 workers (fp32 eager
  unchanged, lr unchanged at 3e-4). This is a THROUGHPUT-ONLY adoption at
  1.12×: the stage-2 quality check at batch 256 was never run (unfunded at
  <1.5×). If future batch-256 runs look worse per-sample than historical
  batch-64 runs, this default is the first suspect — the clean comparison is
  equal wall-clock, batch 64 vs 256, lr scaling as a second axis.
