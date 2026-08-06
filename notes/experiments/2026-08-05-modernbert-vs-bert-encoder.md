# BertModel vs ModernBERT-XXS as the option-scoring encoder — 2×2 factorial ablation

Design: **2×2 factorial ablation** (block type × geometry), not a two-point
comparison. Proposed by Rami 2026-08-05 after reading the Orbit Wars 2nd-place
writeup ("determine if ModernBERT-XXS is the exact model that author selected,
then ablate my BertModel against it"), coached into this card.

Why factorial and not the obvious A-vs-C comparison: swapping our encoder for
the author's config changes **three** things at once — the transformer block
(RoPE / GeGLU / bias-free / pre-norm vs BERT's learned-absolute / GELU-MLP /
biased / post-norm), the width (192→256), and the depth (6→7). A single
A-vs-C run that wins tells us nothing about *which* of the three paid. The
2×2 buys attribution for one extra pair of arms.

## Source check: is ModernBERT-XXS actually what the author used?

Yes, and precisely. From the writeup
(<https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars>,
author `simjeg`, 2nd place, 2026-07-07), verbatim: the transformer "follows
the XXS configuration proposed in the Ettin Encoder series: 7 layers, 4
attention heads and d=256", reported at **3.9M parameters** inside a 4.3M
total (290K 1D-CNN encoder + 3.9M ModernBERT + 130K heads).

Independent confirmation: instantiating `ModernBertConfig(hidden_size=256,
num_hidden_layers=7, num_attention_heads=4, intermediate_size=384)` — the
Ettin XXS config (`jhu-clsp/ettin-encoder-17m`, arXiv 2507.11412) minus its
50,368-row token table — gives **3,903,488 parameters**, i.e. the author's
"3.9M" to three significant figures. The config is the exact one, not a
paraphrase.

Three author modifications, all of which apply to us too:
1. **No token embedding table** — inputs are continuous embeddings. Identical
   to our setup (we feed `inputs_embeds`; `vocab_size=1` is a dummy).
2. **No positional encoding** — his bodies are an unordered set. *We cannot
   copy this*: our option token's position IS the action index the head must
   emit. We keep our learned `opt_pos_emb`/`slot_pos_emb` input embeddings in
   every arm; ModernBERT's RoPE rides on top (theta 160000 → near-identity
   rotation over 99 positions, so this is a small perturbation, not a
   competing positional scheme).
3. **Global attention only** — his max length is N=44. Ours is
   `1 + 50 state slots + 48 options = 99`, also far below any sliding window
   worth having, so we match him by setting `local_attention = 2·seq_len`.

⚠️ Scope of what the source supports: the author **asserted** this backbone,
he did not ablate it. His writeup lists "smaller or larger CNNs and
transformers" among things he tried and dropped — a *size* sweep, never a
block-type comparison against BERT. His 2nd place came from 10B steps of
from-scratch PPO self-play, not from the encoder. So "ModernBERT-XXS is what
2nd place used" is verified; "ModernBERT-XXS is why 2nd place won" is not
claimed by anyone, including him.

## Card

- **Hypothesis:** replacing our post-norm `transformers.BertModel` encoder
  with a ModernBERT block of matched width/depth/parameters raises held-out
  top-1 action match, because pre-norm residual streams and the GeGLU MLP
  train more stably at a fixed step budget than post-norm + GELU-MLP. Stated
  as a mechanism we can be wrong about: if the gain is real it should appear
  in arm B (block type alone, params matched to ±1.2%); if it appears only in
  arm C it was geometry or capacity, not the block.
- **Registered counter-mechanism (why this could *hurt*):** our option tokens
  are an unordered set whose index is an arbitrary action id. RoPE imposes a
  relative-distance prior over that index that is semantically meaningless
  here — the exact reason the Orbit Wars author deleted positional encoding
  from his copy. A negative Δ on arms B/C is a predicted outcome, not a bug.
- **Independent variables (two, crossed):** encoder block type
  {BertModel, ModernBERT} × geometry {192/6/6 ours, 256/7/4 Ettin-XXS}.
- **Arms** (encoder params measured, not estimated):

  | arm | block | h / L / heads | intermediate | encoder params | role |
  |---|---|---|---|---:|---|
  | **A** | BertModel | 192 / 6 / 6 | 768 | 2.689M | production baseline (incumbent recipe) |
  | **B** | ModernBERT | 192 / 6 / 6 | 512 | 2.657M | **block type alone** (−1.2% params vs A) |
  | **C** | ModernBERT | 256 / 7 / 4 | 384 | 3.903M | **the author's exact Ettin-XXS config** |
  | **D** | BertModel | 256 / 7 / 4 | 1024 | 5.555M | **geometry alone** — the control that stops a C win being read as "ModernBERT" when it was "wider+deeper" |

  A→B isolates block. A→D isolates geometry. C is the replication point;
  C−B is geometry within ModernBERT, C−D is block at XXS geometry.
  Intermediate 512 at d=192 is the param-matching value (ModernBERT's GeGLU
  is 3·d·i vs BERT's 8d², and it drops all biases), chosen so arm B is not
  secretly a capacity ablation.
- **Baseline:** arm A, retrained here under the identical protocol — *not*
  the shipped `models/il_agent` checkpoint. Re-running the baseline costs one
  arm and removes every recipe/corpus difference from the comparison.
- **Metric & protocol (staged, cheapest-first):**
  - *Stage 1 screen (this run):* 4 arms × 3 seeds {42, 43, 44}, **equal
    steps** (12,900 = 1 epoch over the 4,554-episode 2026-07-26 train day,
    hub source), lr 3e-4 cosine, batch 64, fresh init. Metric = top-1 action
    match on the held-out day 2026-07-27, **paired on identical eval rows**
    across arms, reported as mean ± across-seed spread and as a paired Δ.
    Offline accuracy is a screen here, not a selection metric — it is being
    used for what it is valid for (does this encoder fit the same rows
    better at the same step budget), and it does not decide anything alone.
  - *Stage 2 (only if Stage 1 clears the bar):* retrain the winning arm at
    the production recipe (full hub corpus, 3 epochs), run the fallback
    diagnostic (a silent `_safe_choice` regression would invalidate any
    battle result), then a local tournament against the anchored pool, then
    the ladder. No "better" claim gets written before the ladder read.
- **Pre-registered decision:** promote an arm to Stage 2 iff its mean paired
  top-1 Δ vs arm A **exceeds the across-seed standard deviation of arm A**
  and is positive for **≥2 of 3 seeds**. Drop the block-type hypothesis if
  arm B's |Δ| falls inside that band — a null there means ModernBERT's block
  buys us nothing at our scale regardless of what arm C does. If C wins but
  B and D are both null, the honest conclusion is "unexplained interaction",
  and Stage 2 waits for a fourth arm, not a submission.
- **Cost estimate:** 12 runs × 12,900 steps. Measured baseline throughput is
  10.86 steps/s at 192/6/6 batch 64 on MPS (`notes/phase6_projection.md`), so
  ≈20 min/run for arms A/B and more for the wider arms C/D → **projected
  4.5–6 h wall-clock, run serially so MPS is never contended**, nice'd.
  ~12 checkpoints × ~13 MB ≈ 160 MB disk (54 GiB free at preflight). No
  Kaggle submission slot consumed at Stage 1.
- **Prior work checked:**
  - Memory `model-size-flat-above-3m`: 3.32M vs 10.99M total params measured
    −0.11 pt, p=0.25, n=70,778 paired — the capacity axis is **flat upward**
    at our data scale. This is a real prediction about this experiment: arm D
    (5.56M encoder) should land on top of arm A. If D beats A materially,
    that prior result is in tension and gets re-examined.
  - `notes/phase1_decisions.md` §1.1: the BertModel choice was made for
    "well-tested block + `from_pretrained` for free", explicitly **not** on
    measured grounds — so there is no prior measurement defending arm A.
  - `notes/experiments/2026-08-04-il-full-corpus-ladder.md`: the production
    recipe and the 0.7583 offline-accuracy operating point.
  - Memory `feature-ablation-negative-result` and
    `energy-option-identity-negative-result`: this repo's last two
    architecture-adjacent ablations were both null. Base rate matters.
  - Memory `max-episodes-is-a-biased-prefix`: eval is paired on identical
    rows across arms for exactly this reason.
  - Ettin encoder series, arXiv 2507.11412; ModernBERT, arXiv 2412.13663.

## Result (Stage 1 screen, 2026-08-05)

**Grid complete: 12 of 12 cells.** The two missing cells (`c_mbert_xxs_s44`,
`d_bert_xxsgeom_s44`) were retrained 2026-08-05 20:1x–21:07 after the first
driver pass was killed mid-run under memory pressure (swap 11 GiB of 12 GiB;
no traceback, silent SIGKILL — not a code fault). Settings were held identical
across the retry, `--num-workers 4` included, so the seed-44 cells are
comparable to the rest. Numbers below supersede the earlier 10-cell table.

- **Observed** — paired top-1 on the held-out `eval` split, 75,563 rows from a
  400-episode prefix, every arm scored on one identical encoded cache
  (`reports/encoder_ablation.json`, `scripts/eval_encoder_ablation.py`):

  | arm | block | h/L/heads | seeds | top-1 mean | SD | per-seed Δ vs A | mean Δ | McNemar p |
  |---|---|---|---:|---:|---:|---|---:|---|
  | **A** | BertModel | 192/6/6 | 3 | .7312 | .0047 | — | — | — |
  | **B** | ModernBERT | 192/6/6 | 3 | .7414 | .0037 | +0.30, +0.93, +1.83 | **+1.02 pp** | 3.4e-3, 3.8e-20, 4.2e-69 |
  | **C** | ModernBERT | 256/7/4 | 3 | **.7475** | **.0023** | +1.35, +1.61, +1.94 | **+1.63 pp** | 2.1e-42, 1.5e-57, 1.4e-79 |
  | **D** | BertModel | 256/7/4 | 3 | .7270 | .0109 | +0.12, −1.89, +0.51 | −0.42 pp | 0.199, 9.3e-72, 2.4e-7 |

  Baseline across-seed SD = 0.468 pp. Pre-registered bar = mean paired Δ > that
  SD and ≥2/3 seeds positive. **B clears (3/3). C clears (3/3). D does not.**

  The third seed moved C **up** (+1.48 → +1.63 pp, and it is now the
  lowest-variance arm at SD 0.23 pp) and moved D up but still short
  (−0.89 → −0.42 pp). Both revisions run in the direction the "Open" note
  warned about, which is the argument for finishing a grid before quoting it.

- **Deployment cost** (single-threaded CPU, batch 1 — the evaluator's shape,
  `torch.set_num_threads(1)` before any torch work): arm C is **3.23 ms per
  decision median vs A's 2.97 ms** (+9%), and its p95 is *better* — 3.65 ms
  vs A's 17.28 ms. Checkpoints 18.2 MiB vs 12.7 MiB, both far inside the
  ~197.7 MiB envelope. The accuracy is not bought with latency we cannot pay.

- **Decision:** **adopted at Stage 1** — the block-type hypothesis is supported
  and Stage 2 is authorised by the card. Not yet promoted to production: Stage 2
  (production-recipe retrain → fallback diagnostic → local tournament → ladder)
  has not been run, and no "better" claim is made until the ladder read.

- **What we learned:**
  1. **The gain is the block, not the geometry.** Arm B isolates block type at
     matched params (−1.2%) and pays +1.02 pp on every seed. Arm D isolates
     geometry inside BertModel and does not clear. The pre-registered
     attribution test — "if it appears only in arm C it was geometry or
     capacity, not the block" — resolves in favour of the block.
  2. **The registered counter-mechanism did not fire.** RoPE over an
     arbitrary action index was predicted as a plausible *harm*; arms B and C
     are the two best. With theta 160000 over ≤99 positions the rotation is
     near-identity, which is the likely reason it costs nothing.
  3. **`model-size-flat-above-3m` survives.** Arm D adds ~2.9M encoder params
     over A and returns −0.42 pp. Capacity remains flat-to-negative upward at
     this data scale; C's win is not a capacity win.
  4. **But block and geometry interact, and that is the finding a two-point
     comparison would have missed.** Widening pays *inside* the ModernBERT
     block (C−B = +0.61 pp) and not inside BertModel (D−A = −0.42 pp). So an
     A-vs-C run alone would have credited "ModernBERT-XXS" with the whole
     +1.63 pp; the true decomposition is +1.02 block, ~0 geometry, remainder
     interaction. The two control arms cost half the grid and are the only
     reason that is knowable.
  5. **The stability mechanism shows up in the variance, not only the mean.**
     Arm D (post-norm at 7 layers) is the least stable arm across seeds
     (SD 1.09 pp; seed 43 at .7147), while arm C (pre-norm, same depth) is the
     most stable (SD 0.23 pp). That is the pre-norm story the hypothesis
     pre-registered, not a post-hoc reading — and it is why the across-seed
     spread belongs in the table next to the mean.
  6. **Calibration: a real but weaker concern than the 10-cell read.** Mean
     ECE across 3 seeds: A .0412, B .0448, C .0447, D .0462. The ModernBERT
     arms fit better and calibrate slightly worse, but C's per-seed range
     (.0397–.0478) overlaps A's (.0389–.0427), so the gap is inside seed
     noise. ⚠️ These ECE figures come from the **1,280-row in-training eval** —
     the very measurement flagged below as not the Stage 1 metric — so this
     point is provisional. Per `search-leaf-value-must-be-centered` it matters
     if this encoder ever feeds a search leaf; re-measure properly at Stage 2
     rather than assuming it washes out.

- **How this was nearly mis-called (methodological trap, worth reusing):** the
  in-training eval written to `train_metadata.json` is `--eval-batches 20` ×
  batch 64 = **1,280 rows**, not the Stage 1 metric. On those 1,280 rows arm A's
  across-seed SD reads 0.135 pp; on the real 75,563-row eval it is 0.468 pp —
  **3.5× larger**, because A's seed-44 cell is genuinely weak (.7258) and the
  small eval could not see it. Absolute accuracies were also ~1.8 pp low
  (prefix effect, cf. `max-episodes-is-a-biased-prefix`), shifting all arms
  together and leaving the paired deltas intact. Reading the grid off
  `train_metadata.json` is what made the arms look noise-separated. **The
  Stage 1 metric is `eval_encoder_ablation.py`; the in-training number is a
  training monitor and is not a substitute for it.**

- **Resolved:** the two missing cells were trained and are in the table above.
  C's figure is +1.63 pp on 3/3 seeds, not the provisional +1.48 pp on 2.

- **Open — Stage 2, and what is NOT yet claimed.** Everything above is offline
  action-match on one train day at 12,900 steps. It is a screen, and this repo
  has had three local-vs-ladder inversions. For calibration: +1.63 pp is ~3×
  the largest offline gain this architecture has ever produced (the full-corpus
  data jump was +0.5 pp), which is why it earns a Stage 2 — not evidence the
  ladder will agree. Stage 2 as pre-registered:
  1. Retrain arm C at the production recipe (all hub train days, 3 epochs,
     seed 42) — control is `models/il_agent_full_0804`, already trained, same
     recipe, architecturally identical to arm A.
  2. Fallback diagnostic (`run-fallback-diagnostic`) — a silent `_safe_choice`
     regression would invalidate any battle result.
  3. Local tournament vs the anchored pool, then the ladder read.
  **The ladder submission is not taken unattended**: it displaces an active
  slot and needs Rami's explicit go-ahead.

  Stage 2 step 1 was **not launched** at 21:5x: preflight found a concurrent
  session's `train_ppo_puffer.py --init` 2h50m into an MPS run, a third
  worktree running its own job, and swap at 10.7 GiB of 11.3 GiB. That is the
  same memory pressure that silently SIGKILLed a Stage 1 cell, and the
  standing rule is to never overlap MPS jobs. Launch the ~3 h retrain when
  the machine is clear.
