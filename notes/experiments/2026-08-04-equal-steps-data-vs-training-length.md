# Was it the data or the training length? (equal-steps controlled comparison)

Disentangles the confound left by
`2026-08-04-exploiter-transfer-to-more-data-il.md`, where the all-days model
beat the exploiter but had changed on two axes at once.

- **The confound being resolved:** the all-days model saw ~3.3× the episodes
  **and** took ~3.3× the optimizer steps (127,748 vs the original's 38,562).
  "3 epochs" is not a fixed amount of training — a 3.3× bigger corpus makes an
  epoch 3.3× longer. So the exploiter's collapse (94% → 38.5%) could be caused
  by richer data OR simply by more gradient updates. Standing rule 4 ("compare
  at equal steps, not equal epochs") exists for exactly this.

- **Hypothesis:** the drop is caused by **data variety**, not training length.
  **Mechanism:** the exploiter wins by driving the opponent into states the
  training corpus under-represents (bench-heavy energy, damaged active, wall
  matchups); a wider corpus covers more of those states, whereas extra gradient
  steps on the *same* narrow corpus only sharpen the same blind spots.
  **Therefore** a model trained on the wide corpus for only 38,562 steps should
  still resist the exploiter.

- **Design & why this one:** controlled comparison holding **optimizer steps
  fixed** while varying corpus. This is the minimum change that separates the
  two candidate causes — a scaling sweep over steps would answer a different
  (sensitivity) question and cost several runs instead of one.

- **Independent variable:** the **training corpus** (small vs all-days), with
  steps pinned at 38,562 in both.

- **Arms:**

  | Name | Corpus | Episodes | Steps | Status |
  |---|---|---:|---:|---|
  | original imitation (`models/il_agent`) | 2026-07-26 only | 4,554 | 38,562 | already trained |
  | **all-days, equal steps** (`il_alldays_equalsteps_0804`) | all 10 days | **15,032** | **38,562** | **this run** |
  | all-days, full (`il_alldays_0804`) | all 10 days | 15,032 | 127,748 | already trained |

  ⚠️ **Methodological trap hit on the first launch — read before reproducing.**
  There are TWO copies of `train_il.py` with **opposite defaults** when
  `--hub-days` is omitted:

  | Copy | Line | Default |
  |---|---|---|
  | `main` | 420 | `hub_days = train_days` → **only 2026-07-26** (4,554 eps) |
  | `rewrite-kaggle-pokemon-tcg-prompt-c69997` | 359 | `hub_days = None` → **all 10 days** (15,032 eps) |

  The all-days model was trained with the *worktree* copy. The first launch of
  this arm used `main`'s copy and silently streamed the ORIGINAL corpus — it
  would have "shown" that training length was irrelevant, purely as an artifact.
  Caught from the run header (`4554 episodes, days 2026-07-26`), killed, and
  relaunched against the worktree copy, which reports `16 shards, 15032
  episodes`. The runner now hard-aborts if the header names the single train
  day. **The Hub repo holds 10 days**, not the 4 recorded in older notes.

  The third row is what we already have; the new run is the missing cell that
  makes the table readable. Note `--total-steps` drives the LR schedule too, so
  this is a *complete* 38,562-step run with proper warmup and decay — not the
  127k run stopped early. That matches how the original was trained.

- **Held fixed:** architecture (3,318,721 params, hidden 192 / 6 layers /
  6 heads), lr 3e-4, batch 64, warmup 200, grad-clip 1.0, seed 42, plain BC
  (`weight_arm=none`, `winner_only=False`).

- **Metric & protocol:** exploiter win rate, **100 mirrored pairs = 200 games**,
  seats alternated, T=1.0, Wilson 95% CIs, `PTCG_DEVICE=cpu`,
  `PTCG_FALLBACK_TRACK=1` — identical to the runs it is compared against, via
  `scripts/exploiter_transfer_experiment.sh`. Offline top-1 reported alongside,
  but see the decision rule: offline accuracy has already proven nearly blind to
  this effect.

- **Pre-registered decision:**
  - **Exploiter win rate stays low (~38%, CI overlapping the all-days-full
    result)** → **data variety** is the cause. Collecting/streaming more
    episodes is the lever; promote corpus growth.
  - **Exploiter win rate returns high (~92–97%, CI overlapping the control
    band)** → **training length** is the cause. Corpus size was incidental; the
    original was simply undertrained, and the cheap fix is to train the existing
    recipe longer.
  - **Intermediate (~55–75%, CIs excluding both)** → both contribute; report as
    partial and do not claim either lever alone.

- **Validity gate:** opponent fallback rate must be ~0. Re-run the
  `IL_MODEL_DIR` positive control if anything looks anomalous — that check is
  what proved the all-days result was a genuine weights effect.

- **Cost estimate:** 38,562 steps ≈ **45–50 min** (the 127,748-step run took
  2h45m at the same settings), MPS, plus ~90 s for the 200-game eval.
  **Chained, not overlapped:** a PPO job (pid 7833, `--max-seconds 3600`) holds
  the machine until ~16:03; this run starts after it exits, per the standing
  rule that MPS jobs never contend. Preflight at 15:13: load 12.16/12 cores,
  18% memory free, 66 GiB disk.

- **Prior work checked:**
  `notes/experiments/2026-08-04-exploiter-transfer-to-more-data-il.md` (the
  confounded result this resolves),
  `notes/experiments/2026-08-04-exploiter-generalization-to-pool.md`,
  `notes/il_v2_todo_from_exploiter_diagnostic.md` item 3, memory
  `exploit-transfers-across-seeds-not-across-data`.

- **Known confound, stated not controlled:** the all-days corpus differs from
  the original's in *composition* as well as size (different days, different
  meta). "Data variety" here means "this bigger corpus", not a clean
  size-only manipulation.

## Result (2026-08-04, training finished clean at 38,562/38,562)

The equal-steps model: offline top-1 **0.7414**, top-3 0.9484, ECE 0.0124,
eval loss 0.7008 (vs 0.381 majority).

Exploiter win rate, 200 mirrored games each, identical protocol:

| Model | Episodes | Steps | Exploiter win rate | 95% CI |
|---|---:|---:|---:|---|
| original imitation | 4,554 | 38,562 | 0.955 | [0.917, 0.976] |
| plain imitation, 3 seeds | 4,554 | 12,900 | 0.942 | band [0.920, 0.970] |
| **all-days, EQUAL steps** | **15,032** | **38,562** | **0.425** | **[0.359, 0.494]** |
| all-days, full | 15,032 | 127,748 | 0.385 | [0.320, 0.454] |

85W–115L. Opponent fallback **0/13,506** — the model played every decision.

- **Observed:** holding steps fixed at 38,562 and swapping ONLY the corpus
  (4,554 → 15,032 episodes) drops exploitability from **0.955 → 0.425**. Adding
  3.3× more steps on top of the bigger corpus (38,562 → 127,748) moves it only
  0.425 → 0.385, and those CIs overlap heavily.

- **Decision:** the pre-registered "stays low, CI overlapping all-days-full"
  branch fires → **DATA VARIETY is the cause, not training length.** The card's
  hypothesis is **confirmed**. Nearly the entire effect is bought by the corpus;
  the extra 89,186 steps contribute a difference indistinguishable from noise.

- **What we learned (transferable):** this is the payoff of insisting on
  equal-steps rather than equal-epochs. The original confounded comparison
  (3 epochs vs 3 epochs) moved data and compute together by the same 3.3×, and
  the intuitive reading — "we trained it longer, so it got better" — turns out
  to be **wrong**: compute was nearly irrelevant here, and a plausible-sounding
  attribution would have sent the next month of work at the wrong lever.
  Corollary for planning: **collect/stream more episodes, don't buy more MPS
  hours.** A 3.3× corpus at constant compute was worth ~53 points of
  exploitability; 3.3× compute at constant corpus was worth ~4.

  Second lesson, methodological: the first launch of this arm silently trained
  on the ORIGINAL corpus because two copies of `train_il.py` disagree on a
  default (see the trap table above). It would have produced a confident,
  exactly-backwards conclusion. **Read the run header that names your data
  before trusting any training result** — the gate now enforces it.

- **Still unanswered:** offline accuracy again barely moved across all of this
  (0.7534 original → 0.7414 equal-steps → 0.7583 all-days-full) while
  exploitability swung 0.955 → 0.425. Accuracy remains blind to the property we
  actually care about. And none of this shows the new models are *robust* — only
  that this one adversary fails against them; a fresh exploiter trained against
  them is still the open test (TODO item 6).
