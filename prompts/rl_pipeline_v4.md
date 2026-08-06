# RL Pipeline v4 — IL → Offline RL → Anchored Self-Play, with a kill-gate

Revision of `prompts/rl_pipeline_v2.md`. Anything v2 settles with a measurement
stays settled and is cited, not re-derived. Phase names follow
[[Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers]]
(Grigsby et al., RLC 2025, arXiv:2504.04395 — "the Metamon paper"): **IL →
Offline RL → self-play fine-tuning**.

Written 2026-08-05. Ladder closes 2026-08-16.

---

## §0 — What the ladder actually says, and the kill-gate

This section is first because it is the only section whose conclusion could
make the rest of the document moot.

Every settled submission, sorted by whether a learned model is in the acting
path (`reports/submission_ledger.jsonl`, read 2026-08-06):

| Family | Settled scores | n | Range | Median |
|---|---|---:|---|---:|
| **Learned** (BC / offline RL / PPO / MCTS-with-BC-prior) | 418.0, 400.0, 397.3, 395.0, 383.2, 320.4, 291.4, 275.1, 254.9, 190.3 | 10 | 190–418 | ~352 |
| **Hand-written heuristic / search**, no ML in the acting path | 804.0, 701.6, 699.0, 692.7, 683.2, 677.1, 666.1, 602.6 | 8 | 602–804 | ~688 |

*(Re-read 2026-08-06 06:19; `55270787` 311.3 → 383.2, `55253900` 267.4 → 254.9.)*

**`55279487` is excluded because 2–3 readings do not settle a score.** It is a
learned agent that has read 493.9 → 547.7 → **460.5**. An earlier revision of
this section warned that at 547.7 it was closing on the heuristic band and that
§0's premise might be withdrawn; at 460.5 the gap to the lowest heuristic score
(602.6) is **142 points, outside the ~±100 band**, and that warning is retracted.
It was an artifact of reading a trajectory out of a mid-band swing — see §9.7,
where the same mistake was made twice in opposite directions.

**Two further learned reads have landed since, both low**, and neither depends on
the gate ref: `55284059` (`selfplay_g3_final`) at 395.3 and `55270787` at 383.2.
That makes **three** self-play/BC submissions below the plain-BC 418.0 line. The
family separation in the table above is holding, not eroding.

**The two families do not overlap.** No learned agent has ever outscored any
heuristic agent on this ladder, across 18 submissions. The gap between medians
is ~377 points against a documented same-build noise band of ~±100. The
best-ever build (804.0, `55162376`) is behaviorally a pure `AdvancedPolicy`
heuristic — `USE_SEARCH`/`USE_BC_PRIOR` are present but dead code
(`notes/phase0_gate0_report.md`), corroborated by
[[search-bc-prior-no-measurable-gain]] (900 games each, Δ −1.1 pts).

v2 and this document both plan eleven days of work on the weaker family. That
is worth doing only if the learned line can plausibly close ~300 points. So:

### The kill-gate (evaluate 2026-08-08)

`55253900` (`selfplay_g1_ref430k`) is the first properly anchored self-play
candidate — KL leash to a frozen teacher, PFSP league, promotion ratchet, all
of it. It settles ~08-08. Readings so far: 248.6 → 267.4 → 267.4 → **254.9**
— flat-to-down across five reads, ~295 below the PASS line and ~195 below the
FAIL line. Unchanged as of the 2026-08-06 06:19 read; no new reading has landed
since 08-05 19:50.

- **Gate PASS** (settled ≥ 550): the learned line is closing. Execute this
  document as written.
- **Gate FAIL** (settled < 450): four independent confirmations
  (`ppo_u120832` 275.1, `s2_e1_s43` 395.0, `il_alldays_0804` 418.0, this).
  **Phases 2–3 stop.** The remaining days go to §0.1 instead. Write up the
  negative result — "offline-RL-then-self-play on 3.3M params and ~10⁴
  episodes does not reach a hand-written heuristic in this domain" is a real
  finding, and it is the honest one.
- **Ambiguous (450–550)**: one more read on 08-10, then decide. Do not run a
  third generation on an ambiguous gate.

### §0.1 — What the eleven days do if the gate fails

Ranked by expected ladder points per day, all reusing built infrastructure:

1. **Harden the 804.0 heuristic.** It is the only thing we have that scores.
   `agent_core_improved.py` is 38K of hand-written policy; the local pool and
   `benchmark_agents.py` can measure changes to it directly. Its own roll
   spread (666–828) is the noise floor to beat.
2. **Search on top of the heuristic, not on top of BC.** `search_api.py`
   (built, measured: `search_step` ~0.1 ms, `reports/search_api_phase2.md`)
   with the *heuristic* as the rollout/prior policy rather than the BC net.
   The BC prior is what failed; the search machinery is not implicated.
3. **Endgame the active set** (§Ladder) — worth ~370 points today for zero
   modelling work, and worth more than any model change we have evidence for.

Nothing in §0.1 requires this document. It is here so the failure branch is
specified before the gate is read, not negotiated after.

---

## §1 — What changed from v2

1. **§0 kill-gate added.** v2 assumed the learned line would work; the ladder
   has since produced 10 learned submissions and none reached the heuristic
   band.
2. **Reward: win = 1, everything else = 0.** v2 mapped draws to 0.5. Draws are
   ~1% of decisive-gate games (`promotion_log.jsonl`: 99–100 decisive of 100)
   so the simplification costs nothing measurable. **Standing rule preserved:
   a loss is never −1** — negative terminal reward under γ < 1 pays the agent
   to delay losing.
3. **The anchor is the promoted-best, not a fixed IL prior**, and it is
   continual (§Anchor). Built: `pokemon_tcg/pufferl_kl.py`,
   `pokemon_tcg/promotion.py`.
4. **The 70% promotion ratchet is measured, and it does not work as intended**
   — 1 promotion in 170 gates across three full runs (§Anchor). v4 changes it.
5. **KL units answered with repo measurements** (§KL scale), and the
   0.01/0.1/1.0/10 ladder in the v4 request is **mis-scaled for this domain by
   ~an order of magnitude**. The real operating band is 0.001–0.3 nats.
6. **Entropy collapse identified as an already-tripped stop condition** —
   `exp(H) ≈ 1.0–1.19`, the policy is effectively deterministic (§KL scale).
7. **Explicit planned-value hyperparameter table** (§Hyperparameters), one row
   per knob, each marked default / convention / measured / guess.
8. **League: PFSP against weak-opponent farming** (§League), plus a verdict on
   whether the local pool should gate submissions (it should not).
9. **Deck randomization** as a knob inside Phase 3, not a phase of its own
   (§Decks). 33 pool decks probed, 33 cg-legal.
10. **Data: the corpus streams from Hugging Face**; local raw split dirs are
    stubs and must never be pointed at (§Data). Retrain cadence specified.
11. **Ladder is a standing daily item with slot arithmetic** (§Ladder), not a
    gate clause.
12. **R-NaD verdict** delivered (§Research): do not replace PPO, but adopt one
    specific piece of it.

Carried forward from v2 unchanged and not re-argued: the literal 15M/50M/200M
grid stays rejected in favour of Option B (3.32M / ~10–12M / 50M-probe); the
architecture gap (per-decision set transformer vs the paper's causal
transformer over battle history) stands as registered-not-built; weighted BC is
situated as the one-step corner of offline RL (AWR / MARWIL / CRR lineage).

---

## §2 — Phase map

| Phase | Paper's name | Status | Gate |
|---|---|---|---|
| 1 | **IL** | Complete. `il_alldays_0804`, 3.32M params, full Hub corpus, 127,748 steps, top-1 .7583 | Ladder read 418.0 (settled) |
| 2 | **Offline RL** (weighted BC on the human corpus) | Arms E0/E1/E2 trained; E1 → 395.0, E0-v2 → 320.4 | Rung-2 + a ladder read above the Phase-1 number. **Not met** |
| 3 | **Self-play fine-tuning** (our on-policy PPO adaptation) | g1/g2/g3 run, ~3.5M steps total; `selfplay_g1_ref430k` → 254.9, `selfplay_g3_final` → 395.3 (1 read) | §0 kill-gate, 08-08 |

**The accepted deviation, stated once and not re-litigated.** The Metamon paper
makes Phase 3 *offline*: self-play battles are appended to an ever-growing
corpus and offline RL retrains over the union, so the human prior can never be
forgotten because every retrain still sees it. We run **on-policy PPO** instead.
The original reason was disk; the honest current reason is that storing and
re-training over generations of self-play episodes is not something this laptop
workflow supports — the human corpus lives on Hugging Face and streams, and the
laptop is not a corpus store.

**What the trade costs:** PPO only ever sees states the current policy visits,
so *forgetting the prior is a live failure mode*, not a hypothetical one.
**How the cost is watched:** the KL anchor (§Anchor) is the guard, and
`kl_to_prior` — globally and per SelectContext, every update, in
`train_metrics.jsonl` — is how the guard is checked rather than assumed. The
measured answer for g1 is in §KL scale: the anchor held. The run's problem was
the opposite of forgetting.

---

## §3 — Anchor: continual KL to the promoted best, with the ratchet

### 3.1 What is built

`pokemon_tcg/pufferl_kl.py::PuffeRLPriorKL` is a verbatim copy of PufferLib
3.0's `PuffeRL.train`, with additions fenced by `>>> KL ANCHOR` markers:

```
loss = pg_loss + vf_coef*v_loss - ent_coef*entropy + kl_coef * KL(π_θ ‖ π_ref)
```

- **Continual, not a warm-up.** The term is on for every minibatch of every
  update, before and after every promotion. Nothing decays it.
- **The reference is the promoted best.** `retarget_prior()` copies the live
  weights into the frozen reference in place (`load_state_dict` — `requires_grad`
  stays False, the optimizer never held those tensors), and the pull continues
  against the new anchor.
- **Promotion** (`promotion.py::evaluate_gate`): every `--promote-every` updates
  (default 20), live π_θ plays the frozen π_ref as **mirrored pairs** — two
  games with seat order swapped, cancelling first-player advantage — both sides
  sampling at temperature 1.0, same deck. Promote iff live wins strictly more
  than `--promote-threshold` (0.70) of **decisive** games. `MIN_DECISIVE = 20`
  guards against a gate that mostly draws.

**Both π_θ and π_ref apply the identical action mask before softmax.** This is a
correctness requirement, and it has a test, not a footnote: `masked_kl()` requires
both logit tensors to carry the same legal-action mask, and `train()` asserts it
once per prior —

```python
assert torch.equal(logits.detach() < NEG/10, prior_logits < NEG/10), \
    "KL anchor mask mismatch: current and prior policies disagree on which options are illegal"
```

Masking uses a finite `NEG = -1e9`, never `-inf` (`torch.Categorical.entropy`
NaNs on `-inf`). Illegal slots contribute exactly 0 to the KL sum: `p`
underflows to 0 through the softmax, and `logp - logq` stays finite because both
sides are offset by the same `NEG`. So no `torch.where` gating is needed.

**Anchor vs PPO's own clipping — the one-sentence distinction.** PPO's clip is
against `π_old`, the policy that collected the *current* rollout, and it is
refreshed every update; it bounds the size of each individual step. The KL
anchor is against a *frozen reference held across many updates*; it bounds total
drift over the run. They do different jobs and neither substitutes for the other.

**Value/win-prob anchor:** specified but **not shipped**. v2 called for two
terms; the condition for adding the second was "if critic instability is
actually observed." It was: g1's `explained_variance` reads −0.51 at update 1
and the critic was re-initialized from scratch. But the fix for a critic that
explains negative variance is a better critic init (§Hyperparameters), not a
pull toward an equally-bad frozen one. Revisit only if the critic is warm-started
and *still* unstable.

### 3.2 The ratchet does not fire — measured

Three complete runs, gates every 20 updates:

| Run | Steps | Gates | Promotions | Mean win-rate | SD | Max |
|---|---:|---:|---:|---:|---:|---:|
| `selfplay_g1` | 962,560 | 47 | **1** (step 430,080, wr 0.73) | 0.551 | 0.071 | 0.730 |
| `selfplay_g2` | 1,290,240 | 63 | **0** | 0.537 | 0.061 | 0.670 |
| `selfplay_g3` | 1,228,800 | 60 | **0** | 0.568 | 0.050 | 0.660 |

**One promotion in 170 gates across 3.48M steps.** Consequences:

1. **The "continual pull toward previous-best" degenerates into "static pull to
   the IL prior"** — exactly the v2 design the ratchet was meant to replace. For
   g2 and g3 the reference *never moved once*.
2. **The one promotion that did fire is the one that got submitted**, and it
   reads 267.4 against its own teacher's 418.0.
3. **The 70% bar is not the noise-proof threshold it was chosen to be.** At
   n = 100 decisive games and a true win-rate equal to each run's mean, the
   probability that *at least one* gate in a run reads > 0.70 by chance is
   **4.5% (g1), 2.3% (g2), 15.8% (g3)**. That is a family-wise error rate across
   repeated looks, not a per-gate one — the gate is a significance test re-run
   47–63 times per run, and nothing corrects for the multiplicity.

**Diagnosis: the anchor and the ratchet are fighting each other.** The anchor's
job is to prevent drift; the ratchet requires drift large enough to win 70% of
mirrored games. At β = 0.05 the anchor wins, the policy stays within ~0.15 nats
of the reference, and 0.15 nats is not enough behavioural change to clear 70%.

### 3.3 What v4 changes

- **Drop the win-rate gate; retarget on a fixed cadence.** This is R-NaD's
  *update* step (§Research): π_{n+1,reg} ← π_{n,fix}, applied unconditionally
  once the inner dynamics have run, with no win-rate condition anywhere in it.
  Concretely: retarget every 200k steps regardless of the gate. Keep running
  the gate and keep logging it — it becomes a *diagnostic* of how far the policy
  moved between retargets, which is what it actually measures, rather than a
  control-flow condition it is too noisy to be.
- **If the gate is kept as a gate anyway**, it needs `--promote-pairs 250`
  (n = 500 decisive games, σ ≈ 0.02) and a rule that promotion requires two
  *consecutive* gates over threshold. That is ~5× the current eval cost per
  gate; the fixed-cadence option is cheaper and better-founded.
- **Log KL against two references separately**: the current promoted best *and*
  the original IL prior. Today only the former is logged, so "has it forgotten
  the human prior" is literally unanswerable from the logs after the first
  retarget.
  **BUILT 2026-08-05** — `--il-prior` on `train_ppo_puffer.py`, logged as
  `kl_to_il_prior` globally and `kl_il` per SelectContext. Diagnostic only:
  detached, never multiplied by β, never added to `loss`. The second forward
  pass is skipped while both references hold the same policy (generation 1),
  and whether they do is decided from the weights via
  `kl_math.same_policy_weights`, not from the caller's intent — `run_selfplay_g3.sh`
  anchors to a promoted teacher while the IL reference stays the human prior,
  and assuming otherwise would log the anchor's KL under both names.
  Verified on a forced-retarget smoke run: pre-retarget the two series coincide
  exactly (the control), post-retarget `kl_to_prior` reads 9e-05 while
  `kl_to_il_prior` reads 0.0225 — 250× larger, and previously invisible.
  Chart `reports/figures/kl_dual_reference.png`; note
  `notes/experiments/2026-08-05-v4-preflight-dual-kl-and-prereg.md`.
  Calibration point from the same runs: two independently-trained BC
  checkpoints (`il_agent` vs `il_alldays_0804`) sit ~0.88 nats apart, i.e.
  nearly a full "anchor stopped binding" (§4.2) apart from each other. The
  `> 0.6` stop rule is about `kl_to_prior`, the term in the loss;
  `kl_to_il_prior` does not inherit it.

---

## §4 — KL scale: what the logged number means

### 4.1 Units, precisely

`masked_kl` returns `(p·(log p − log q)).sum(-1).mean()` — **nats per decision**,
averaged over the minibatch. The driver then accumulates
`losses['kl_to_prior'] += kl.item()/total_minibatches`, so the logged scalar is
**mean nats per decision, averaged over every minibatch in one update (epoch)**.
It is written to `<out>/train_metrics.jsonl` every update, alongside a
`per_context` dict giving `{kl, entropy, n}` **per SelectContext** — the v4
request's ask here is already satisfied; the gap was that nothing read it.

The reference is whatever `kl_prior_policy` currently holds — i.e. the **current
promoted best**, not the initial IL prior. After a retarget the number is against
the new anchor and is not comparable to earlier values. §3.3 fixes this by
logging both.

### 4.2 What the numbers mean here — measured, not assumed

`selfplay_g1`, β = 0.05, 941 logged updates:

| Step | KL to ref | Entropy | clipfrac |
|---:|---:|---:|---:|
| 1,024 | 0.010 | 0.340 | 0.0863 |
| 80,896 | 0.207 | 0.200 | 0.0238 |
| 240,640 | 0.263 | 0.220 | 0.0130 |
| 400,384 | 0.246 | 0.246 | 0.0046 |
| 480,256 | 0.134 | 0.216 | 0.0137 | ← post-retarget (ref moved at 430,080) |
| 719,872 | 0.146 | 0.108 | 0.0078 |
| 959,488 | 0.166 | 0.157 | 0.0049 |

**The scale is set by the policy's own entropy, and that entropy is tiny.**
Final-update per-context readings:

| ctx | n | KL | entropy | exp(H) |
|---:|---:|---:|---:|---:|
| 0 | 1683 | 0.2272 | 0.1759 | **1.19** |
| 7 | 494 | 0.1405 | 0.1609 | 1.17 |
| 22 | 187 | 0.0006 | 0.0044 | 1.00 |
| 21 | 185 | 0.2677 | 0.1323 | 1.14 |
| 8 | 165 | 0.0431 | 0.0053 | 1.01 |
| 4 | 47 | 0.2249 | 0.0923 | 1.10 |

`exp(H)` is the effective number of options the policy is actually choosing
among. **It is 1.0–1.19 everywhere.** The policy is effectively deterministic.

So the calibration ladder, for *this* domain:

- **KL ≈ 0.001–0.01** — indistinguishable from the reference. Either training
  has barely started, or the context is forced (ctx 22, 1, 41, 43: KL ~1e-4,
  entropy ~0 — these are the forced/no-op decisions, ~6.7% of corpus mass per
  [[bc-corpus-pass-mass-measured]]).
- **KL ≈ 0.1–0.3** — the entire operating range of three full runs. This is what
  a working anchor at β = 0.05 looks like. Note KL ≈ 0.23 in ctx 0 is **larger
  than that context's own entropy (0.18)**: the policy has moved to a different
  near-deterministic action on a meaningful fraction of decisions, which is a
  substantial behavioural change even though the number looks small.
- **KL ≈ 1.0** — never observed in 3.48M steps. It would be ~5× the entropy
  scale of the whole distribution. If you see it, the anchor has stopped binding.
- **KL ≈ 10** — not reachable. With `exp(H) ≈ 1.2` and typically <10 legal
  options, the distribution does not have 10 nats to give.

**This contradicts the v4 request.** The proposed 0.01 / 0.1 / 1.0 / 10 ladder
is mis-scaled by roughly an order of magnitude at the top end. Anchoring
intuition on RLHF-per-token KL numbers does not transfer: this action space is
smaller and this policy is far sharper.

**Stop rule, in numbers:**
- Global `kl_to_prior` > **0.6** sustained over 5 consecutive updates → the
  anchor has stopped binding; halt, report, raise β.
- Any context with `n ≥ 100` and `entropy < 0.05` → that context has collapsed
  to a single action; halt and inspect.
- **Already tripped, and this is the real finding:** g1's global entropy fell
  0.34 → 0.11–0.18 and `exp(H)` sits at ~1.1. v2 named entropy collapse a stop
  signal; it fired and nobody read the log. **Temperature-1.0 rollout sampling
  is not exploring** — the BC prior is already sharply peaked, so self-play
  produces near-duplicate games, which is precisely the failure mode v1 and v2
  warned about for *argmax* rollouts. It is happening at temperature 1.0 anyway.

---

## §5 — Hyperparameters: planned values

Starting values actually planned for the next run. `(D)` = PufferLib 3.0 default
kept, `(C)` = published self-play convention, `(M)` = derived from a measurement
in this repo, `(G)` = guess.

| Knob | Planned | Why | Swept later? |
|---|---|---|---|
| learning rate | **1.5e-4** (was 3e-5) | **(M)** clipfrac fell to 0.005–0.014, ~1/20 of the 0.1–0.2 healthy band — the update is barely moving. Same gen-2 diagnosis, now confirmed across g1/g2/g3. `(C)` for imperfect-info PPO is 2.5e-4 | yes, priority 1 |
| clip ε | **0.2** | **(D)** — never set in `train_ppo_puffer.py`, inherited from PufferLib. Make it explicit | yes |
| γ | **1.0** | **(C)+(M)** terminal-only reward; γ=1.0 is the convention in imperfect-info self-play (Leduc, PSRO best-response oracles). Already our default. Note gen 2 tried 0.997 and did not promote | no — settled |
| GAE λ | **0.95** | **(C)** standard; games are ~68 decisions/seat so λ=0.95 has an effective horizon of ~20 decisions, well inside a game | yes, priority 4 |
| value coef | **0.5** | **(D)** | no |
| entropy coef | **0.01** (was 0.001) | **(M)** `exp(H) ≈ 1.1` — the policy is deterministic and 0.001 is not holding it open. `(C)` is 0.01. Secondary to the anchor in principle, but the anchor is not the thing that failed here | yes, priority 3 |
| **KL-anchor β** | **0.02** (was 0.05) | **(M)** at 0.05 the policy sat at KL 0.15 and never cleared the ratchet. Lower β buys the drift the ratchet needs. Risk: forgetting — watched by the stop rule in §4.2 | yes, priority 2 |
| update epochs | **3** | **(D)** repo default | yes |
| minibatch size | **512** | **(D)** repo default | yes |
| rollout buffer | **1024** (8 envs × 128 bptt) | **(M)** `batch_size: "auto"` from envs × horizon | yes |
| rollout temperature | **1.0** | **(M)** — and §4.2 shows this does not explore. The fix is the entropy coef, not temperature > 1, which would distort the on-policy distribution PPO optimizes | no |
| parallel envs / workers | **8 / 8** | **(M)** hard constraint: `assert num_envs == num_workers` — the cg engine is a per-process singleton, one env per worker always | no |
| promotion cadence | **every 200k steps, unconditional** | **(M)+(§Research)** replaces the 70% gate; see §3.3 | no |
| head-to-head games behind the test | **100 (50 mirrored pairs), diagnostic only** | **(M)** σ ≈ 0.05 at n=100 is too noisy to gate on (§3.2). Kept as a logged diagnostic | n/a |
| optimizer | **adam**, not muon | **(M)** fine-tuning a BC prior | no |
| max grad norm | **1.0** | **(D)** | no |
| total timesteps | **1.0M** | **(M)** ~61 agent-steps/s measured → ~4.5 h/run; chain runs, never overlap MPS | no |

**Typical self-play PPO vs single-agent PPO — where the defaults diverge:**

- **γ.** Single-agent PPO defaults to 0.99. Two-player zero-sum with a
  terminal-only reward conventionally uses **γ = 1.0** — there is nothing to
  discount toward, and γ < 1 actively distorts the objective (it is the same
  mechanism that makes a −1 loss reward pay for stalling).
- **Entropy.** Single-agent PPO treats entropy as the exploration knob. In
  self-play it is also a *diversity* knob: entropy collapse means the league
  plays the same game repeatedly, so the opponent distribution degenerates even
  though the opponent *set* is diverse. Our measurement is a direct instance.
- **Non-stationarity.** The opponent changes as the learner changes, so the
  value function chases a moving target. Standard mitigations: keep the critic
  warm across generations rather than re-initializing (g1 re-initialized and
  read `explained_variance` −0.51), and prefer smaller, more frequent updates.
- **What self-play adds that single-agent PPO has no analogue for**: opponent
  sampling (§League), the anchor (§Anchor), and promotion cadence. These matter
  more than any of the classic PPO knobs, which is why β sits at priority 2.

### Protein sweep space — **later, not now**

Do not run this before the §0 gate. Each trial costs a real evaluation, and
[[ppo-hyperparameter-sweep-negative-result]] already recorded a 19-run Protein
sweep dropped on pre-registered validation (winner's curse). Driver exists:
`scripts/sweep_ppo_protein.py`.

| Priority | Knob | Range |
|---|---|---|
| 1 | learning rate | 1e-5 – 3e-4, log |
| 2 | KL-anchor β | 0.005 – 0.2, log |
| 3 | entropy coef | 0.003 – 0.03, log |
| 4 | GAE λ | 0.90 – 0.98 |
| 5 | update epochs × minibatch | 1–4 × {256, 512, 1024} |
| 6 | retarget cadence | 100k – 400k steps |

Fixed, not swept: masking (structural), reward mapping, seed policy,
`num_envs == num_workers`.

---

## §6 — League: opponent sampling and the weak-opponent bias

### 6.1 The hypothesis, and whether it happened

> A learner rewarded per win can inflate its win-rate by farming weak league
> agents while learning nothing — or getting worse — against strong ones.

**Partly confirmed, and PFSP is already built.** `models/selfplay_g1/pool_weights.json`
shows the sampler is not uniform:

```json
{"kiyotah_dragapult": 0.2444, "plamen06_steel": 0.2486, "mechi22_alakazam": 0.0506}
```

`mechi22_alakazam` — the one we beat easily — is downweighted ~5× relative to
the two strong opponents. That is the right shape.

But the mix is `--mix 0.625,0.375,0` (mirror / league / **pool = 0**), so in the
g-runs the public pool weight was **zero**: the learner trained against itself
and frozen copies of itself, and the PFSP weights over public agents were
computed but barely exercised. The farming risk did not materialize because the
weak agents were hardly sampled — but neither did the *benefit*, and the
per-opponent win-rate log (`pool_wr` / `pool_games` in `train_metrics.jsonl`)
shows `mirror` dominating everything.

### 6.2 The scheme for v4

- **Mix: 0.4 mirror / 0.3 frozen league / 0.3 public pool.** Non-zero pool
  weight is the point; the 0.625/0.375/0 mix made this a mirror-only run.
- **PFSP weighting over the pool**, keep the existing shape:
  `w(o) ∝ p_o(1 − p_o)` style frontier weighting — overweight opponents we win
  ~50% against, underweight both those we beat >90% (nothing left to learn) and
  those we lose >90% to (no gradient signal). Recompute weights each retarget.
- **Farming detection, named explicitly**: `pool_wr` and `pool_games` per
  opponent per update, already logged. Farming looks like *aggregate win-rate
  rising while the win-rate against the top-2 weighted opponents is flat or
  falling*. Chart it per run; the aggregate alone cannot show it.

### 6.3 Verdict: should a local-pool benchmark gate Kaggle submissions in Phase 3?

**No.** Separate the two roles and gate on neither locally:

| | Training league | Evaluation pool |
|---|---|---|
| Wants | diversity, including weak agents (gradient signal across the strength range) | ladder-predictiveness only |
| Composition | all 33 legal pool decks/agents + frozen self | the ladder-anchored subset |
| Role | shapes the gradient | ranks candidates for submission |

The tempting argument for gating is the anchored 8-agent pool's Spearman
ρ = +0.929 ([[anchored-pool-rho-0.93]]). Three reasons that is not enough:

1. **The ρ is in-sample.** The pool has never been tested on a submission it did
   not help select.
2. **n = 8, and most of its anchors are self-reported**, not verified
   ([[ladder-anchors-mostly-alleged]] — `agent_core_improved` is 685.3, not the
   804.0 it was anchored at).
3. **Local ordering has now inverted against the ladder four times** — 08-02,
   08-03, the IL-prior MCTS read, and `selfplay_g1`, which met its local gate
   (beat its teacher 73–27; 62.5% [55.6, 68.9] over 200 games) and reads 267.4
   against that teacher's 418.0.

**And gating is nearly free to skip**, which is the decisive point: §Ladder shows
43 readable slots remain against a queue that will not produce 43 candidates.
Slots are not scarce; settled reads are. Gating trades an abundant resource for a
scarce one.

**What to do instead** — turn ρ into a real number at zero cost: **pre-register**
the pool's predicted ladder ordering for the next five submissions *before*
submitting them, record it in the ledger's `expects` field, then measure
out-of-sample rank correlation. If it holds out-of-sample past ~08-12, gating
becomes defensible for a future competition. It does not become defensible in
time to be used in this one.

**BUILT 2026-08-05** — `scripts/prereg_pool_prediction.py`
(`lock` / `show` / `expects` / `bind` / `score`), file
`reports/pool_prereg.jsonl`, chart
`reports/figures/pool_prereg_calibration.png`.

The registered claim is the **ordinal** prediction, not a score. A score map is
frozen alongside it — `ladder = -554.4 + 0.8076 × local_glicko`, fit on n = 19
overlapping agents on the compounded scale, in-sample **ρ = +0.718**
(permutation p = 0.0007) — but its residual SD is **198.7**, about twice the
~±100 same-build noise band, so the 95% band is ±397 and a point prediction
claims almost nothing. Spearman is ordinal anyway, and the ordering is the part
of the claim the mostly-alleged anchors ([[ladder-anchors-mostly-alleged]])
cannot move. The coefficients are frozen in source and the ratings they were fit
on are snapshotted to `reports/pool_prereg_source_glicko.json`, because
`reports/glicko_ratings.json` keeps compounding and would not be reproducible
from the commit — refitting after an outcome lands is the exact failure the
arrangement exists to prevent.

Slate locked: `proto` (1808.0 → 905.7, never submitted), `mega_lucario_restore`
(1748.7 → 857.9), `rule_baseline` (1601.7 → 739.1, never submitted). Two of the
five slots stay unlocked because the post-gate candidates do not exist yet;
freezing the *coefficients* today is what makes their predictions non-tunable
when they do. `score` refuses to report an out-of-sample ρ below 4 settled
candidates.

`il_agent_v2` is excluded as contaminated — it read back before the slate was
locked. Recorded as an unregistered observation only: the frozen map puts it at
**831.0**, its submitter predicted **380–460**, and it is reading **547.7** on 2
readings and rising (493.9 → 547.7). It currently sits *between* the two, above
the human prediction and below the pool's. Both cannot be right, and whichever
way it settles is informative about the map's direction of error — but it is not
part of the out-of-sample ρ and must not be counted toward it.

---

## §7 — Decks

### 7.1 The meta-deck list — built

`src/pokemon_tcg/deck_pool.py` + `scripts/probe_deck_legality.py`, with
`tests/test_deck_pool.py`. Probe result (`reports/deck_legality.json`):
**33 decks probed, 33 legal, 0 failed**, with `play_out: true`.

The standing trap is handled: public decks *can* be cg-illegal (>1 ACE SPEC →
INVALID every game, [[public-deck-cg-legality-ace-spec]]), so legality is probed
at `battle_start` before a deck enters the list. The current pool happens to be
clean, but the probe stays in the pipeline — new decks are not assumed legal.

Mining decks from the episode corpus (reconstructing an opponent's list from
episodes where they played it out) is **registered, not built**. 33 legal decks
is already more than the sampler exercised; corpus mining is only worth building
if held-out-deck generalization (§7.3) turns out to be the binding constraint.

### 7.2 Placement: a knob inside Phase 3, not a phase of its own

**Recommendation: a knob.** The plumbing already exists — `--deck-pool`,
`--deck-pool-k`, `--deck-pool-seed`, `--deck-pool-pin`, `--mirror-deck`.

Reasoning: a separate phase would need its own init, its own gate, and its own
ladder read — three days of calendar for an axis we can turn on with a flag.
More importantly, a separate deck-robustness phase *followed by* Phase 3 would
have its robustness trained back out by a Phase 3 that fights one deck. Deck
diversity has to be present while the policy is being shaped, which means inside
the opponent sampler.

Small mutations (swap one card) are **not** recommended for v4: every mutation
needs its own legality probe, and 33 legal base decks is enough coverage to
measure generalization at all. Register for later.

### 7.3 Measuring generalization

Hold out **8 of the 33 decks** from the training sampler entirely. Report
win-rate on trained decks vs held-out decks separately. The gap between them
*is* the generalization number, and it is the only thing here that measures
"good against deck combinations it never trained on." A run that improves on
trained decks while the held-out gap widens has overfit the league.

### 7.4 The pilot-deck question — open, do not assume

Local evidence said the deck axis was worth 544 Glicko points and the checkpoint
axis was flat ([[deck-matters-more-than-checkpoint]]). **The ladder disputed the
deck half on 2026-08-05.** Same weights (`il_alldays_0804`), two decks:

| Deck | Submission | Settled |
|---|---|---|
| Mega Lucario ex | `55248985` | **418.0** |
| Marnie's Grimmsnarl ex | `55270787` | **311.3** |

Grimmsnarl scored *lower*, opposite to the local prediction — though the 107-point
gap sits inside the ~±100 same-build band, so this falsifies the confident local
claim without establishing the reverse. **Treat pilot-deck choice as an open
experiment under the `deck-selection` skill, gated on settled reads, n ≥ 2 per
arm.** Do not bake either deck into Phase 3.

---

## §8 — Data

- **The corpus streams from Hugging Face.** `Rami/ptcg-episodes` is the **only
  copy**. Train with `--data-source auto --num-workers 4`.
- **Local raw split dirs are stubs** — 24/12 files, and `train_combined` is 9,820
  dangling symlinks. `iter_decisions` returns near-nothing *silently*
  ([[local-raw-splits-are-stubs]]). **Never point a trainer at them.**
- **Ratings source**: the [Kaggle episodes index dataset](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index)
  carries per-episode avg/median score — the input to any skill-weighted arm.
  Note [[skill-filter-negative-result]]: min_score filtering *hurt* BC, and
  pooled thresholds are day filters in disguise. No skill-gate arm is planned.
- **The all-days IL retrain already happened**: `il_alldays_0804`, no skill-gate
  arm, 3 epochs / 127,748 steps over the full Hub corpus, top-1 .7583 (vs .7534
  and a .381 majority line). Settled ladder read **418.0** — the best any
  learned agent has scored, and still ~270 below the heuristic band.

**Retrain cadence.** As new episode days land on the Hub:

- **Trigger**: a retrain when the Hub corpus grows ≥ 25% in episodes since the
  last IL checkpoint, or when ≥ 3 new days land — whichever comes first.
- **What a refresh invalidates downstream**: everything warm-started from it.
  A new IL prior invalidates (a) the Phase-2 arms' warm start, (b) the Phase-3
  actor init, and (c) **every KL-anchor number logged against the old prior** —
  KL to a different reference is a different quantity and the two series must not
  be plotted on one axis. It does *not* invalidate the encoder, provided
  `encode_observation()` is unchanged; any encoder change forks the pipeline
  version and invalidates cross-stage comparisons outright.
- **Given the 08-16 close, expect zero further retrains.** A retrain costs ~4 h
  plus a Phase-2/3 redo, and the §0 gate governs whether that spend is justified
  at all. Documented for completeness, not scheduled.

---

## §9 — Ladder: a standing daily task

### 9.1 State as of 2026-08-06 06:19 UTC

Rank **5121 / 6397**, team score **460.5**, top-8 cutoff 1130.1 (gap +669.6).
Active set: `55284059` (395.3) + `55279487` (460.5). Best-ever **804.0** is
still displaced.

*(08-05 20:57: rank 4235/6361, team 547.7.)*

**Both active slots hold learned experiments.** The §9.4 slot-1 restore was not
executed; a concurrent session spent the slot on `il_agent_v2` (`55279487`)
instead. That is the two-experiments-back-to-back state §9.2 names as the way
the team fell 804.0 → 395.0 on 08-03 — the floor is currently whatever these two
learned agents settle at, with no heuristic build underneath them.

*(Earlier reading, for the record: rank 5910/6336, team 311.3, active
`55270787` + `55253900`.)*

### 9.2 The scoring mechanics that drive the strategy

- `max_daily_submissions = 5`, UTC day. **Used on 08-06: 1 (`55284059` 00:29)
  → 4 remain.** (08-05 used 3: `55253900`, `55270787`, `55279487`.)
- Every submission starts at the **μ₀ = 600.0 prior**. A fresh 600.0 is not a
  score. It converges over ~3 days.
- **Active set = the latest 2 by recency**; a new submission displaces the older.
- **Team score = MAX over the active set** (verified 2026-08-05 against
  `reports/leaderboard_history.jsonl`: 418.0 vs 291.4 → 418.0; 311.3 vs 267.4 →
  311.3; 395.0 vs 275.1 → 395.0; no case fits a mean).
- **Displaced submissions keep playing and keep scoring** — verified on
  `55248781`, which moved 439.8 → 276.6 → 291.4 across three readings *after*
  being displaced. Displacement costs the team score, not the measurement.

**The corollary that governs everything below:** because the team score is the
max of the latest 2, **an experiment is free as long as the other active slot
holds a strong build.** Alternating good/experiment never drops the floor;
two experiments back-to-back is exactly how the team fell 804.0 → 395.0 on
08-03, ~4,500 ranks.

### 9.3 Slot arithmetic

| Quantity | Count |
|---|---:|
| Remaining today (08-06, after 1 used) | **4** |
| 08-07 → 08-16, 10 days × 5 | 50 |
| **Total remaining** | **54** |
| **Readable** (submitted ≤ 08-13, settles in time to inform another decision) | **39** |
| Final-positioning (08-14 → 08-16) | 15 |

**A correction to the v4 request's premise, in our favour.** 08-13 is the last
day a submission can be read back *in time to inform another submission*. It is
**not** the last day one counts: games continue to ~08-31, so anything submitted
through 08-16 settles fully. The 08-14 → 08-16 slots are not throwaway — **they
are the final active set, and the final active set is the only thing that
determines placement.**

**Slots are not the binding constraint. Settled reads are.** 43 readable slots
against a queue that cannot produce 43 distinct candidates is why §6.3 refuses to
spend reads on local gating.

### 9.4 Today's queue (2026-08-05, 3 slots)

~~Resubmit the mega_lucario bundle byte-identical.~~ **WITHDRAWN 2026-08-06,
Rami's call. Do not propose it again.**

The rolls of that byte-identical bundle decay monotonically:

| 55162376 | 55191752 | 55219194 | 55228113 | 55224682 |
|---:|---:|---:|---:|---:|
| 827.8 → 804.0 | 699.0 | 692.7 | 683.2 | 666.1 |

~140 points of decline across identical bytes. Extrapolating the trend, a fresh
roll lands near 600 — not the ~688 median this table used to quote, which
averaged over a *declining* series and so overstated what the next roll would
get. The field is also growing (6,206 → 6,397 teams over the same window), so
the decay is most likely the field strengthening around a fixed agent rather
than anything about the agent. That mechanism applies to **every** build we
hold, which is the part worth carrying into §9.5: a bundle banked early keeps
sliding, so the endgame is about *when* the final active set is placed, not
only what is in it.

Secondary benefit of the restore: matchmaking pairs submissions of *similar
rating*, so sitting at 311 means playing 311-rated opponents and learning less
about the field.

### 9.5 Endgame

Land the best ladder-verified bundle in **both** active slots by ~08-14 and
submit nothing experimental after it. Anything submitted after that displaces
the final active set with something unsettled.

### 9.6 The ritual (unchanged, per submission)

Refresh the ledger first (concurrent sessions submit too,
[[refresh-ledger-before-submitting]]) → forced-CPU latency rehearsal → build,
read the printed tarball MiB → submit with a detailed message (what changed,
from what baseline, what it displaces, a falsifiable prediction) → `ledger log`
the detail the 500-char field could not hold → **read back after settling and
record the number next to the local claim.** One control per submission. After
every phase of this pipeline the trained model goes to the ladder and its Glicko
is recorded here before the next phase starts.

### 9.7 Running results

| Ref | Date | What | Settled | Local claim at submit | Verdict |
|---|---|---|---:|---|---|
| 55162376 | 08-01 | agent_core_improved (heuristic) | **804.0** | — | best-ever |
| 55169814 | 08-01 | improved_prob_main | 701.6 | "top-ranked, Glicko 1720.2" | **inverted** |
| 55196434 | 08-02 | s2_e1_s43 (Phase 2, E1) | 395.0 | beats PRIOR 62.5% | inverted |
| 55215267 | 08-03 | ppo_u120832 (Phase 3 gen 1) | 275.1 | beats E1 87.5%, non-overlapping | **inverted** |
| 55248781 | 08-04 | IL-prior MCTS | 291.4 | beats il_agent 67.2% | **inverted** |
| 55248985 | 08-04 | il_alldays_0804 (Lucario) | 418.0 | not stronger locally | consistent |
| 55253900 | 08-05 | selfplay_g1_ref430k | **254.9** *(5 reads: 248.6 → 267.4 → 267.4 → 254.9)* | beats teacher 73–27; 62.5% [55.6,68.9] | **§0 kill-gate, 08-08 — tracking FAIL** |
| 55270787 | 08-05 | il_alldays_0804 (Grimmsnarl) | **383.2** *(6 reads: 361.8 → 314.1 → 353.9 → 366.5 → 383.2)* | deck axis = 544 Glicko pts | **falsified** |
| 55279487 | 08-05 | il_agent_v2 (BC, 9-day corpus, Grimmsnarl) | **460.5** *(3 reads: 493.9 → 547.7 → 460.5)* | beats current il_agent vs every shared opponent, agg 63% vs 32%, H2H 15–1; predicted 380–460 | **inside its own prediction** |
| 55284059 | 08-06 | selfplay_g3_final (Mega Lucario) | 395.3 *(1 read)* | 63.0% ±5.8 vs BC init 45.9% ±5.9 over 270 games; predicted 250–420 | **inside its own prediction; 3rd self-play sub below the BC line** |

Read 2026-08-06 06:19 UTC. Team **460.5**, rank **5121 / 6397**.

**Two directional calls made in this doc about `55279487`, both wrong.** The
19:50 entry said it was "still falling through the μ₀ = 600 prior"; it then rose.
The 20:57 entry said it was "rising away from its own prediction"; it then fell.
Three readings: 493.9 → 547.7 → 460.5, a total swing of 87 points — comfortably
inside the documented ~±100 same-build band. **The lesson is not about this
submission.** It is that narrating a trajectory from 1–3 readings inside that
band produces confident statements with no information in them, twice in a row
here. Quote the band; do not draw the arrow.

Where it actually landed: **460.5, inside the submitter's pre-registered 380–460
band** (at the top edge). The pre-registration was good and the commentary on it
was not.

**Retraction: §0's premise is not under pressure after all.** At 547.7 the gap
to the lowest heuristic score (602.6) was 55 points and this doc said the
learned/heuristic separation was "no longer clean." At 460.5 the gap is back to
**142 points, outside the noise band**. That warning was an artifact of the same
mid-band reading. §0's family separation stands.

**A third self-play submission reads low.** `55284059` is `selfplay_g3_final`,
the gen-3 anchored self-play run, at 395.3 on one reading — below the plain-BC
line (418.0), like `selfplay_g1_ref430k` (254.9) before it. Its submit message
is worth copying as practice: it pre-registered 250–420 *and* stated the local
pool's own failure on this exact family (ρ = −0.50, n=3) before reading back.
This is evidence toward the §0 FAIL branch that does not depend on the gate ref.

**The kill-gate ref did not move.** `55253900` is unchanged at 254.9 on 5
readings, no new reading since 08-05 19:50. Still unresolved, still due 08-08.

**Both active slots are learned experiments again** (`55284059` 395.3 +
`55279487` 460.5), so the team floor is once more whatever two experiments
settle at, with no heuristic build underneath. Best-ever 804.0 stays displaced.

Earlier reading, 2026-08-05 19:50 UTC:

- **`55253900`, the kill-gate ref, has moved down, not up** — 267.4 → 254.9 on
  its 5th reading. Three days from the 08-08 verdict it sits ~295 points below
  the 550 PASS line and ~195 below the 450 FAIL line. Nothing is settled until
  08-08, but the trajectory is not ambiguous-band behaviour.
- **`55270787` rose 311.3 → 353.9**, which does not change §7.4's verdict: the
  Lucario/Grimmsnarl gap is now 418.0 vs 353.9 = 64 points, further *inside* the
  ~±100 same-build band than the 107 it was. The confident local deck claim stays
  falsified; the reverse still is not established.
- **`55279487` is the highest number any learned agent has posted (493.9), and it
  should not yet be treated as a number at all.** It has exactly one reading,
  taken four minutes after submission. Every submission starts at the μ₀ = 600
  prior and descends: its own cousin `55248985` read 600.0 → 418.0 across nine
  readings. 493.9 four minutes in is consistent with a submission still falling
  through the prior, and it is *above* the submitter's own pre-registered
  380–460 prediction — which is the reading that would need to survive, not the
  first one. Re-read 08-06/08-07 before anyone cites it.
- **§0's family separation still holds.** Lowest heuristic score is 602.6;
  highest learned score with more than one reading is 418.0. No overlap, across
  now 19 submissions.

---

## §10 — Research findings

### 10.1 R-NaD / DeepNash — verdict

[[Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning]]
(Perolat et al., arXiv:2206.15378). R-NaD is three steps iterated:

1. **Reward transformation** against a regularization policy π_reg:
   `r'ⁱ = rⁱ − η·log(πⁱ(aⁱ)/π_regⁱ(aⁱ)) + η·log(π⁻ⁱ(a⁻ⁱ)/π_reg⁻ⁱ(a⁻ⁱ))`
2. **Dynamics**: run replicator dynamics (≡ Follow the Regularized Leader) on
   the transformed game to its fixed point π_fix.
3. **Update**: `π_{n+1,reg} ← π_{n,fix}`. Repeat.

Convergence to the transformed game's unique fixed point is guaranteed by a
Lyapunov function, and the sequence of fixed points converges to a Nash
equilibrium of the original game.

**Verdict: do not replace PPO with R-NaD.** Three reasons.

1. R-NaD's guarantees rest on running the inner dynamics **to convergence** each
   outer iteration. We cannot — one inner loop at our throughput (~61
   agent-steps/s) is the whole calendar.
2. It is designed to learn to Nash **from scratch**, not to fine-tune a human
   prior. Our entire premise is that the human prior is worth keeping.
3. DeepNash's validation is at a compute scale ~4–5 orders of magnitude above a
   laptop. Per [[feedback-research-relevance-tight-methodological-fit]], the
   methodological fit is what matters, and the fit here is the *idea*, not the
   algorithm.

**But adopt one specific piece**, and it is the fix for §3.2: R-NaD's update step
retargets π_reg to the converged policy **unconditionally** — there is no
win-rate gate anywhere in R-NaD. Our 70% gate is a bolt-on that fires once per
3.5M steps and turns the continual ratchet into a static anchor. Replacing it
with a fixed-cadence unconditional retarget is R-NaD's actual mechanism, is
cheaper, and is better-founded than a threshold we chose by intuition.

One structural note worth recording: R-NaD's transformation is **two-sided and
antisymmetric** — minus the learner's log-ratio, *plus* the opponent's — which
preserves the zero-sum structure of the transformed game. Our KL penalty is
one-sided, on the learner's loss only. In a true mirror self-play setting that
is a real approximation error; with frozen league and pool opponents the game is
not symmetric anyway, so it is the lesser issue.

### 10.2 Behavior-regularized RL — the vocabulary

The umbrella term the literature uses for our anchor is **KL-regularized** or
**behavior-regularized** policy optimization: constrain π_θ to stay near a
reference π_ref via a divergence penalty. Adjacent, in decreasing closeness:
piKL (Jacob et al., arXiv:2112.07544 — already cited by our MCTS work) does the
same thing at *inference* time; TRPO/PPO's trust region does it against π_old
rather than a frozen reference; RLHF's KL-to-SFT term is the same equation with
different nouns. Using "behavior-regularized" in the writeup puts the doc in the
right search neighbourhood.

### 10.3 Self-play PPO conventions

γ = 1.0 for terminal-only imperfect-information games (Leduc, PSRO best-response
oracles); GAE λ = 0.95 standard, 0.995 in small board games; entropy coefficient
0.01 typical, swept 0–0.05; LR ~2.5e-4. Our γ = 1.0 and λ = 0.95 are already on
convention; our LR (3e-5) and entropy coefficient (0.001) are both roughly an
order of magnitude *below* it, which §5 corrects and §4.2 shows the symptoms of.

### 10.4 PufferLib

Installed and pinned: **3.0** in the py3.12 side venv `.venv-ppo` (4.x could not
share the py3.13 main venv; 3.0 pins numpy<2). Latest release is **4.0**
(2026-04-07): a native backend of ~1500 lines Python + ~5000 lines CUDA C, with
a PyTorch backend retained for prototyping and fallback.

**Recommendation: stay on 3.0 through 08-16.** The CUDA-C native backend is
irrelevant on MPS, the PyTorch fallback path is the one we would land on anyway,
and `pufferl_kl.py` is a **verbatim copy of 3.0's `PuffeRL.train`** that would
need re-diffing against upstream on any upgrade. The Protein results we rely on
are 3.0-era. Upgrading buys nothing measurable before the deadline and risks the
one integration that works. Re-evaluate after the competition.

### 10.5 ByteRL / OSFP and the Orbit Wars framing

[[Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play]]
(arXiv:2303.04096) is the closest published domain match — a two-player card
game with deck-building, solved end-to-end with policy learning plus optimistic
smooth fictitious play. OSFP is the source of the double-oracle / promotion
vocabulary: maintain a population, best-respond against a mixture over it, add
the best response to the population. Our league + promotion loop is a
single-oracle approximation of this.

**A sourcing caveat, stated plainly.** I could not verify the ">70% of
evaluation games promotes the frozen best" rule from the Orbit Wars 1st-place
writeup — the Kaggle writeup page is JS-gated and returned no body text. What is
independently confirmable about IsaiahPressman's approach is the **frozen-teacher
KL** ("a frozen teacher model performs inference on all states, with a KL loss
term for the current model's policy from the teacher's, to stabilize behavior
and prevent strategic cycles") and, from the same lineage, IMPALA with UPGO and
TD-λ terms, plus **reward shaping for the first 20M steps**. Note that last item
diverges from our terminal-only rule — flagged, not adopted; our own shaping
follow-up (S3-E2) stays registered and unbuilt. Treat the specific 70% threshold
as **your recollection, unverified from source** — which matters less than it
would have, since §3.2 replaces the gate on measured grounds regardless.

---

## §11 — Constraints (self-contained restatement)

- `resolve_device()` only; **no CUDA branch exists**. Training on MPS; the
  evaluator is CPU-only (~1.6 vCPU, ~197.7 MiB). `torch.set_num_threads(1)`
  before heavy torch work.
- Everything under `uv run`. Paths from `pokemon_tcg.config`. Seed 42.
- **Action masking is structural** (Pattern-B option-scoring head) and part of
  the objective — in the rollout, in the loss, and on **both** sides of the KL
  ([[A Closer Look at Invalid Action Masking in Policy Gradient Algorithms]],
  Huang & Ontañón, arXiv:2006.14171). Mask with finite `-1e9`, never `-inf`.
- **No opponent-private information reaches the encoder.** Rollouts feed the
  acting agent's `obs_dict` as the ladder serves it, never the engine's
  omniscient state.
- **Disk**: ~98 GB free as of 08-03, but the HF dataset is the only copy of the
  corpus — **never delete or overwrite raw episode data without asking.** No
  phase writes an episode corpus; checkpoints, TB logs, figures, and a few
  transcripts only.
- **Critic is train-time only**; the shipped bundle contains the actor alone.
- Every comparison: **≥3 seeds, equal steps** (never equal epochs), an RD or σ
  on every number, a chart in `reports/figures/`, the control beside the claim.
- **Long runs do not stop for approval** — the 1-hour gate is retired. Report the
  projection, chain runs so MPS is never contended, nice everything, cap workers,
  keep the laptop responsive.
- **Plain checkpoint names** (method-data-seed words); roles via
  `model_registry.py`. No opaque codes.
- `num_envs == num_workers`, always — the cg engine is a per-process singleton.

---

## §12 — Work items, in order

**Standing daily item (every day until 08-16), before anything else:**
`submission_ledger.py refresh` → `check_leaderboard.py` → update §9.7 → decide
the day's slots under the alternation rule (§9.2). A slot that expires unused is
discarded information; a slot spent on a second consecutive experiment is worse
than unused.

0. ~~Restore the active set with a mega_lucario re-roll.~~ **WITHDRAWN
   2026-08-06 (§9.4). Do not re-propose it.** The re-roll's own history is a
   monotone decline, so the "free ~+375" this item claimed was an artifact of
   quoting a median over a falling series.
1. **[08-06 → 08-08] Read the kill-gate.** `55253900` and `55270787` settle.
   Record both in §9.7. **Everything below is conditional on §0 PASS.**
2. **[on PASS] Fix the three measured defects**, in one run, before any sweep:
   LR 3e-5 → 1.5e-4, entropy coef 0.001 → 0.01, β 0.05 → 0.02; retarget on a
   fixed 200k-step cadence instead of the 70% gate; log KL against **both** the
   promoted best and the original IL prior. Acceptance: clipfrac back into
   0.05–0.20, `exp(H)` > 1.5, and the reference actually moving.
   **Logging half DONE 2026-08-05** (§3.3) — correct under either gate branch,
   so it shipped ahead of the gate. The three hyperparameter changes and the
   fixed-cadence retarget remain conditional on §0 PASS and are **not** applied.
   When gen ≥ 2 runs, pass `--il-prior models/il_alldays_0804` explicitly: the
   default follows `--kl-prior`, which from gen 2 on is a promoted teacher, not
   the human prior.
3. **[on PASS] Turn the league on.** `--mix 0.4,0.3,0.3` — the g-runs trained at
   pool weight **zero**. PFSP weights recomputed each retarget; chart per-opponent
   win-rate against the top-2 weighted opponents alongside the aggregate.
4. **[on PASS] Deck randomization as a Phase-3 knob** — `--deck-pool` over the
   33 legal decks, 8 held out. Report trained-deck vs held-out win-rate separately.
5. **[on PASS] Ladder the result**, then §9.5 endgame from ~08-14.
6. **[on FAIL] Execute §0.1** and write up the negative result.
7. **[Parallel, free] Pre-register** the local pool's predicted ordering for the
   next five submissions in the ledger's `expects` field (§6.3). Costs nothing,
   and is the only thing that can turn ρ = +0.929 into a defensible gate.
   **DONE 2026-08-05** — machinery + 3 of 5 candidates locked; the remaining 2
   are post-gate and lock when they exist. See §6.3.

**Stop conditions** (any → halt and report; none require approval):

- Opponent-private field reaching the encoder from the rollout path.
- `kl_to_prior` > 0.6 sustained over 5 updates (anchor stopped binding).
- Any SelectContext with n ≥ 100 and entropy < 0.05 (**already tripped in g1** —
  fix in work item 2 before restarting).
- Two consecutive retargets with the gate diagnostic below 0.55 (the policy has
  stopped improving against its own reference).
- Any design that wants to write an episode corpus to disk.
- Bundle over the ~197.7 MiB envelope.
- A trainer pointed at the local raw split dirs (silent near-empty iteration).
