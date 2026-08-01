# Phase 0 Discovery Report — Pokémon TCG BC Agent

Date: 2026-07-31. All numbers below are measured on this machine against the
actual data on disk (`data/episodes/splits/train-2026-07-26/`, 4,554 episodes;
`data/episodes/splits/eval-2026-07-27/`, 4,430 episodes) — not estimated.
Scripts used to produce these numbers are in the scratchpad and reproducible;
ask if you want them moved into the repo as permanent analysis scripts.

**Headline: significant Phase 2 work already exists in this repo** — before
any of this discovery was done. `src/pokemon_tcg/il_dataset.py`,
`il_model.py`, `scripts/train_il.py`, and a trained checkpoint at
`models/il_agent/` are all present and untracked in git. I audited them as
part of this discovery pass rather than ignoring them. Verdict up front: the
core design choices (Pattern-B option scoring, no opponent-hand leakage, no
history) are actually right and match what I would have recommended. But
there are real gaps — one crash risk, one missing device-abstraction, one
undocumented-but-benign data quirk I had to reverse-engineer from the
environment's own source. Details below, organized by your gate structure.

---

## 0.1 What does the agent actually see

`obs_dict` passed to `agent()` has exactly these top-level keys: `select`,
`current`, `logs`, `remainingOverageTime`, `search_begin_input`, `step`.
**It does NOT include `visualize`** — I confirmed this by direct inspection:
`visualize` is a sibling field on the raw replay's per-agent step dict
(`step[agent_idx]["visualize"]`), one level *above* `observation`, and it is
absent from the `observation` dict itself. `visualize` is spectator-only
rendering data added post-hoc by the environment's `finish()` hook (see
`kaggle_environments/envs/cabt/cabt.py:88-107`) — it is never constructed
until the episode ends and is never passed to a live agent. `il_dataset.py`
already reads only `obs_dict.get("select")` / `obs_dict.get("current")`, so
this leakage vector is closed by construction, and I verified it's closed
empirically too (below).

**Field classification** (verified against a real decision-point obs, not
inferred from the dataclass docstrings alone):

| Field | Visibility | Evidence |
|---|---|---|
| `current.players[i].active/bench` | PUBLIC (both sides) | in-play Pokémon are visible to both players by game rules |
| `current.players[i].discard` | PUBLIC (both sides) | discard piles are public |
| `current.players[me].hand` | OWN-PRIVATE | full `Card` list with real IDs |
| `current.players[opp].hand` | — | **empirically `None`** at every decision point checked; never populated |
| `current.players[i].deckCount` / `handCount` | PUBLIC (count only) | no `deck` key present in `current.players[i]` at all — deck order is never exposed for either side, own or opponent |
| `current.players[i].prize` | PUBLIC (facedown = `None` unless revealed) | matches dataclass docstring |
| `select.deck` | OWN-PRIVATE, deck-step only | populated only when literally selecting from your own deck |
| `logs` | PUBLIC | events since your last decision (`LogType` enum: draw, move, attach, attack, HP change, etc.) |

**Q2 (BLOCKING) verdict: no leakage.** I loaded real decision-point
observations from 8+ episodes and confirmed `players[opp].hand` is always
`None`, no `deck` key ever appears on either player, and `il_dataset.py`'s
encoder (`_build_state_slots`) only ever reads `me.hand`, never touches the
opponent's hand/deck. What would change my mind: finding a single decision
point where `players[opp].hand` is non-null — I did not find one in every
sample checked, but I did not exhaustively scan all 8,984 episodes for this
specific field, so treat it as strong evidence, not formal proof for every
row.

**`logs`**: usable as short-horizon history (draws, attaches, attacks,
switches since your last turn). Not currently used by the v1 encoder — see
Phase-1 history discussion below (out of scope for this report to decide,
flagging for your review).

**`select.context` (SelectContext, 0–48)**: full frequency table below (0.5).
48 distinct values observed on train day, matching the documented enum range
exactly (no unseen future IDs yet, as `cg.api`'s comment warns may happen).

**`select.option` / OptionType / minCount / maxCount**: confirmed against
real data. `minCount`/`maxCount` are enforced server-side — the engine's own
`search_step` raises `"Must be minCount <= len(select) <= maxCount"` on
violation (verified by reading `cg.api.search_step`'s error table), so any
in-bounds selection your policy emits is guaranteed legal by construction if
it respects those bounds.

**`obs["select"] is None` ⟺ deck-selection step**: confirmed — every
episode's step 0 has `select: null, current: null`, and this is the only
point in the recorded data where that occurs.

**`current["result"]` — correction to the assumed encoding.** I checked this
empirically and it does **not** behave as `<0 in progress / 0 win / 1 loss /
2 draw` observed per-agent. Across every episode I inspected (8 in detail,
spot-checked more via the full-day scan), **`current.result` is always `-1`**
in the recorded per-agent observation stream — including at the final
recorded tick of a completed match — and no `LogType.RESULT` (23) log entry
ever appears either. The real outcome is only available via the
**episode-level `rewards`/`statuses` fields** (top of the replay JSON), e.g.
`rewards: [-1, 1]`. **Do not build anything — loss filtering, reward
shaping, terminal detection — on `current.result`; it is dead in this
dataset's replay format.** This is a stronger and different finding than
what you flagged ("not symmetric ±1") — it's not populated at all, ever, in
what an agent actually receives.

---

## 0.2 Do I have the simulator?

**Yes, confirmed by running it, not just importing it.** `kaggle_environments.make("cabt")`
instantiates cleanly; I found and read the actual interpreter source at
`.venv/lib/python3.13/site-packages/kaggle_environments/envs/cabt/cabt.py`
(210 lines) rather than treating it as a black box — this let me resolve a
data-format question below that would otherwise have been unresolvable.

- Instantiate: `from kaggle_environments import make; env = make("cabt")`
- Plug in an agent: `env.run([agent_a, agent_b])` where each is `obs_dict -> list[int]`
- **Measured**: `rule_baseline` (existing repo agent) self-play, 20 matches:
  **3.98 matches/sec** (0.25s/match), 25–176 steps/match (mean 137 steps),
  0/20 errors, all `DONE`.
- `scripts/benchmark_agents.py` already provides a working round-robin harness
  (see 0.3).

---

## 0.3 What already exists in this repo

This is the important one — you should know this before deciding anything
in Phase 1, because a chunk of it is already built.

| Component | Path | State |
|---|---|---|
| Rule-based baseline | `agents/mega_lucario/agent_core.py` | working, entered in benchmark |
| Search-improved baseline | `agents/mega_lucario/agent_core_improved.py` | uses `cg.api.search_begin/search_step` for internal lookahead |
| Prototype search agent | `scripts/_proto_agent.py` | same search API |
| **IL agent (BC policy)** | `agents/il_agent/agent_core.py` + `main.py` | **already implemented and wired into the benchmark harness** |
| IL dataset/encoder | `src/pokemon_tcg/il_dataset.py` (416 lines) | streaming `IterableDataset`, versioned constants, Pattern-B encoding |
| IL model | `src/pokemon_tcg/il_model.py` (169 lines) | `transformers.BertModel` fed `inputs_embeds`, option-scoring head |
| Training script | `scripts/train_il.py` (169 lines) | AdamW, no LR schedule, device picked inline (not via a shared helper) |
| **Trained checkpoint** | `models/il_agent/model.safetensors` | **4.7 MB**, hidden=128/layers=4/heads=4, **no accompanying metrics, run log, git SHA, or record of what it was trained on** |
| Benchmark harness | `scripts/benchmark_agents.py` | round-robin, mirrored-seat pairs (cancels first-player bias), writes `reports/agent_benchmark.json` |

**Contract a new agent must satisfy to enter the benchmark**: a Python module
with a top-level callable `agent(obs_dict: dict) -> list[int]`, and either
(a) a `submissions/<name>/main.py` that pre-wires `my_deck`, or (b) a bare
module with a `my_deck` attribute injected from a sibling `deck.csv` (60
newline-separated card IDs). `agents/il_agent/` already follows this pattern
correctly.

**Things I'd flag as gaps in what's already built, not blockers**:
- `train_il.py` picks device as `cuda > mps > cpu` inline in the script
  itself — this is exactly the pattern your 2.7 spec explicitly rules out
  ("no cuda branch is dead code that will rot," "the ONLY place that decides
  a device"). There is no `utils/device.py` yet.
- No checkpoint device-normalization (`state_dict` isn't explicitly moved to
  CPU before `save_pretrained`), no `map_location` on load, no forced-CPU
  smoke test on record.
- The existing checkpoint has **no metadata** — I can't tell you what it was
  trained on, for how long, or what its offline accuracy was. Treat it as
  disposable scaffolding, not a result.
- **Crash risk, confirmed by direct test**: `agents/il_agent/agent_core.py`'s
  `agent()` has no `try/except` around the model forward pass. I fed the
  model's own `card_emb` embedding an out-of-vocab ID directly and it raises
  `IndexError: index out of range in self` uncaught. `CARD_VOCAB_SIZE=1269`
  is a frozen constant based on an "observed 1..1267" scan — any card ID at
  or above 1269 seen at inference (a real risk given the spec's own warning
  that new IDs may be appended during the competition) will crash the
  process, which — per your own Q16/2.6 — means an instant loss. This is a
  concrete, fixable gap, not a hypothetical one.
- `src/pokemon_tcg/config.py`'s `EPISODES_TRAIN_DIR`/`EPISODES_EVAL_DIR`
  point at `data/episodes/{train,eval}/`, which don't exist — the real path
  (`data/episodes/splits/{train-2026-07-26,eval-2026-07-27}/`) is only
  correctly resolved via `il_dataset.resolve_split_dir()`. Minor, but will
  mislead anyone who trusts `config.py`.

---

## 0.4 Hardware probe

| | |
|---|---|
| Chip | Apple M4 Pro, 12 cores (8 performance + 4 efficiency) |
| RAM | 24 GB total (25,769,803,776 bytes) |
| RAM headroom right now | only ~57 MB "free" pages + ~5.6 GB "inactive" (reclaimable) — real slack is ~5–6 GB under current load, not 24 GB |
| macOS | 26.5.2 (BuildVersion 25F84) |
| Disk | 460 GB volume, **only 47 GB free (90% full)** |
| torch | 2.13.0, MPS available and built |

**Disk is a real constraint you should plan around**: the two splits already
consume 40 GB (20 GB + 20 GB), and only 47 GB is free. Phase 2.1's memmap
shards will need meaningful additional headroom on top of that. Budget this
explicitly before running preprocessing — either preprocess into a
compact/quantized shard format, encode the eval split lazily instead of
materializing it, or free space first.

**5M-parameter set-transformer micro-benchmark** (`nn.TransformerEncoder`,
d_model=352, 3 layers, 4 heads, batch=512, seq_len=64 — sized to hit ~5M
params to match your ask):

| Config | steps/sec | vs CPU fp32 |
|---|---|---|
| CPU fp32 | 0.88 | 1.0x |
| MPS fp32 | 2.81 | 3.2x |
| MPS bf16 (dtype set directly, no autocast) | 3.03 | 3.4x (only **1.08x over MPS fp32**) |

Per your own Phase-3 rule ("if bf16 gives less than ~1.3x on MPS, use fp32
and stop optimizing"): **use fp32, don't chase bf16 here.** bf16 did not
raise any error, it just isn't worth the complexity at this model size.

---

## 0.5 Majority-class baseline

Computed over **every** single-choice decision in the **full** train day
(718,122 rows) and the full eval day (720,552 rows) — not a sample.

**Global majority baseline: 38.1% on both days** (majority label = 0, i.e.
"first legal option," which is usually `END` or the first `CARD` option).
This is good news relative to your worst-case fear (80% baseline) — there is
real signal here. But it is extremely uneven across `SelectContext`:

| Context (id) | n (train) | majority share (train) | majority share (eval) |
|---|---:|---:|---:|
| MAIN (0) | 383,151 | **27.7%** | 28.0% |
| TO_HAND (7) | 105,537 | 48.8% | 50.7% |
| ATTACH_FROM (21) | 29,276 | 46.8% | 46.4% |
| DAMAGE_COUNTER (13) | 24,503 | **26.8%** | 26.1% |
| REMOVE_DAMAGE_COUNTER (16) | 24,409 | 65.1% | 62.5% |
| REMOVE_DAMAGE_COUNTER_COUNT (40) | 22,737 | 83.5% | 85.5% |
| ACTIVATE (43, yes/no) | 22,593 | 95.3% | 97.0% |
| TO_ACTIVE (4) | 21,360 | 39.6% | 38.7% |
| SWITCH (3) | 16,617 | **37.9%** | 37.7% |
| DAMAGE (15) | 16,106 | **33.4%** | 35.0% |
| SETUP_ACTIVE_POKEMON (1) | 9,108 | 79.8% | 79.3% |
| DISCARD_ENERGY (30) | 9,231 | 90.9% | 84.7% |
| EVOLVE (37) | 5,856 | 95.9% | 96.6% |
| DAMAGE_COUNTER_ANY (14) | 5,412 | **32.3%** | 32.2% |
| IS_FIRST (41, yes/no) | 4,554 | 99.6% | 99.9% |
| SETUP_BENCH_POKEMON (2) | 2,652 | 100.0% | 100.0% |

Both days agree closely context-by-context — a good sign for Q21 (no gross
metagame shift between days at the decision-difficulty level).

**Read**: MAIN alone is 53% of all labeled rows and sits at ~28% majority
share — the model's capacity should mostly go here. A handful of contexts
(IS_FIRST, SETUP_BENCH, ACTIVATE, EVOLVE) are 95–100% deterministic and
contribute ~zero learning signal while diluting a shared cross-entropy loss
— recommend either routing them to trivial heuristics or explicitly
down-weighting them, not training a shared head to memorize them.

**Additionally**: 205,373 of 1,509,575 total decisions (13.6%, both days
similar) have **exactly one legal option** — 100% trivial by construction.
Recommend dropping these from the loss entirely, per your own Q7 reasoning —
they inflate every aggregate accuracy number for free.

---

## 0.6 Label quality — whose behavior are we cloning?

- **`manifest.csv` (episode-level `avg_score`/`min_score`/`sum_score`)
  exists ONLY for the eval day.** I verified this directly: all 4,430
  manifest rows match the eval split's episode IDs exactly; zero overlap
  with train. **There is currently no skill/rating signal available for the
  training day at all.** eval `avg_score` ranges 1075–1223 (mean 1122, std
  34); `min_score` (weaker of the two players in a match) ranges 984–1216.
  This is a genuine gap for Q1. Two ways to close it, neither requiring new
  downloads of episode data: (a) if this manifest was originally produced by
  a since-removed download script hitting the Kaggle API, regenerating a
  train-day equivalent would need a network call — **I did not do this,
  it's a stop condition, ask first**; (b) proxy train-day quality via
  team-name overlap — 150+ of the ~172 train-day team names also appear on
  eval day with a measured score there (e.g. "Dries @ Tufa Labs", "Dominic
  Peel", "James Cox" are heavy contributors on both days), so their eval
  score is a same-agent, different-day stand-in. Imperfect (assumes
  day-to-day skill stability) but free.
- **Terminal quality (Q3): clean.** 9,106/9,108 train agent-statuses and
  8,859/8,860 eval agent-statuses are `DONE`; only 2 and 1 `TIMEOUT`
  respectively (~0.02%). No `ERROR`/`INVALID` statuses observed at all.
  Draws are rare (4/4,554 train, 1/4,430 eval, ~0.1%). **Recommend no
  filtering needed for abnormal termination** — there's essentially nothing
  to filter.
- **Agent/team concentration (Q4)**: 172 distinct teams contribute to train
  day (top team: 462/9,108 agent-slots = 5.1%; top-10 = 32.6%); 136 distinct
  teams on eval day (top team 8.1%; top-10 = 41.5%). Not dominated by a
  single contributor, but a meaningful top-10 concentration — worth capping
  or downweighting per-team row contribution if any one team's strategy
  turns out to be degenerate.
- **Deck diversity (relevant to Phase 1.4, surfacing now since I measured
  it)**: own-side card IDs observed across a 400-episode/800-perspective
  sample = **148 distinct card IDs**, far more than one 60-card deck's
  worth (~20–25 unique cards). **The corpus spans many different decks, not
  one frozen list.** This directly bears on your Phase 1.4 question: if you
  filter training data down to only the frozen deck, you will discard most
  of the corpus; if you keep all decks, the model needs to know which deck
  it's piloting or it will average across incompatible strategies. I'm
  surfacing the evidence, not making the call — that's explicitly a Phase 1
  decision you asked me to argue, not decide in Phase 0.

---

## MAJOR FINDING (not on your list, genuinely blocking Phase 2 until understood): the action-pairing offset, resolved

`il_dataset.py`'s docstring claims the recorded `action` for a given
decision is logged "one tick late" (response to `decisions[i]`'s select
appears in `decisions[i+1]`'s action field), "verified empirically, 0/3347
mismatches... across a 20-episode sample." I re-ran this at 300-episode
scale and got a very different-looking number: **of 91,682 nominal
`maxCount==1` decisions, only 52.1% produce a valid length-1 label; 46.6%
pair to an *empty* action despite `minCount >= 1`**, which should be
illegal per the engine's own validation. That's not a small discrepancy — I
did not want to hand you a report with a labeling pipeline this shaky
underneath it, so I ran it down.

**Root cause, confirmed by reading the actual environment interpreter
source** (`kaggle_environments/envs/cabt/cabt.py`, not guessed): the
interpreter only updates `state[index].observation` for whichever player is
currently `ACTIVE` (`select_player = Battle.obs["current"]["yourIndex"]`);
the other player's observation object is left completely unchanged —
**including its `select` field, which is never cleared to `None`.** kaggle-
environments' core still invokes every agent's `agent()` callable on every
tick regardless of whose turn it is, so the inactive player's function gets
called repeatedly against a **stale, unchanged observation**, and its
returned action is discarded by the interpreter and logged as empty. So a
large fraction of what looks like "a fresh decision with select≠None" in the
per-agent stream is actually a *repeat echo* of an already-answered
decision, not a new one.

I confirmed this precisely: cross-tabulating "is this decision's
`(turn, turnActionCount, context, minCount, n_options)` signature identical
to the immediately preceding one" against "is the paired action empty,"
**the correlation is essentially perfect** (150-episode sample):

| signature repeats prev? | paired action empty? | count | share |
|---|---|---:|---:|
| no (fresh decision) | no | 24,449 | 52.0% |
| no (fresh decision) | yes | 167 | 0.4% |
| yes (stale echo) | yes | 22,364 | 47.6% |
| yes (stale echo) | no | 0 | 0.0% |

**Verdict: `iter_decisions()`'s existing filter is correct and not a bug** —
it happens to discard stale echoes as a side effect of requiring
`len(action) == 1`, because stale echoes always pair to `[]`. The
"718,122 labeled rows" figure is a legitimate count of real decisions, not
half-corrupted by mispairing. The 0.4% residual (fresh decision, still
empty action) is small enough to be genuine declines or edge cases, not a
systemic problem. **I'd still recommend making this explicit rather than
implicit** — the current code relies on an emergent property of an
unrelated length check to do deduplication; a future encoder change that
loosens that check (e.g. to also accept 0-length `minCount==0` declines,
which you specifically want per Q10) would silently reintroduce stale
echoes into the training set unless the dedup is done explicitly first. I
did not fix this — it's a one-line addition to Phase 2's preprocessing, not
something to patch mid-discovery.

---

## Q1–Q30 — answered, with evidence, not guesses

Format: **answer** — evidence — what would change my mind. Only substantive
entries below; a few (11, 12, 15, 20, 22, 26) genuinely need a Phase-1
architecture decision first and are deferred to that conversation rather
than guessed at here.

**Q1 [BLOCKING] Winners/both/score-filtered?** No score signal exists for
train day (see 0.6). Cannot answer definitively without either a network
call (blocked, would ask first) or the team-name-overlap proxy. Recommend:
start with **plain BC on all rows** (not winner-filtered) as the v0
baseline, since majority-baseline evidence (0.5) shows real signal exists
even in the unconditional distribution, then layer in the team-name-overlap
skill proxy as a re-weighting pass once Phase 1 architecture is settled —
this is a data-availability constraint, not a modeling preference.

**Q2 [BLOCKING] Does the replay leak opponent info?** No — confirmed
empirically (0.1). `players[opp].hand` is always `None`; no `deck` key ever
exposed; encoder never reads opponent-private fields.

**Q3 Invalid/crash/timeout episodes present?** Negligible — ~0.02% TIMEOUT,
0% ERROR/INVALID (0.6). No filtering needed.

**Q4 Concentration by submitting agent?** 172 (train) / 136 (eval) distinct
teams; top-10 = 32.6% / 41.5% of decisions. Meaningful but not dominant
(0.6).

**Q5 Filter vs weight by outcome?** Deferred to Phase 1 — needs the Q1
answer settled first (can't weight by outcome quality we don't have for
train day). Leaning plain BC for v0 given data availability, would revisit
once the team-name proxy is built.

**Q6 How much of the row count is distinct?** Sampled (400 episodes, loose
state signature — turn/context/n_options/active+bench+opp_active
ids/hand+prize counts): **33.4% unique for turn ≤ 3, 48.0% unique for
turn > 3**. Real but not catastrophic duplication — roughly 2–3x, not 10x+.
Model-size implications: with ~718K labeled rows and this duplication rate,
effective distinct states are still in the ~250-350K range, which is
consistent with (not overturned by) a 2–15M param budget.

**Q7 [BLOCKING] Share of trivial decisions, does loss reflect it?**
13.6% of all decisions have exactly 1 option (0.5). Recommend dropping these
from the loss — confirmed cheap and mechanical.

**Q8 Shared model vs per-context heads?** MAIN alone is 53% of rows;
several near-deterministic contexts are <1% each. Row counts strongly
support per-context loss weighting at minimum; whether that extends to
separate heads is a Phase-1 architecture call I'll bring evidence to, not
decide here.

**Q9 maxCount>1 frequency?** 132,834 / 1,509,575 = **8.8%** of all decisions
(train day; eval nearly identical at 8.5%). Not 1% (ignorable) but not 15%+
either — a real minority that needs a real (if simple) treatment, matches
your own framing.

**Q10 minCount==0 frequency, decline rate?** 118,381 / 1,509,575 = 7.8% of
all decisions have `minCount==0`; of those, 21,292 (18.0%) are genuine
declines (`action == []`). **Important, and currently unhandled correctly
by the v1 pipeline**: `iter_decisions()` requires `len(action) == 1`, which
means every one of these 21,292 genuine declines is silently dropped from
training — the existing checkpoint has never seen a single labeled example
where the correct answer is "decline." This is the one item on your own
list where I'd flag the existing code as not meeting your stated bar
("Handle minCount == 0 ... as a real, learnable action") — it currently
does the opposite.

**Q13 Does the option head share the card embedding table?** Yes — verified
by reading `il_model.py`: `ref_embed = self.card_emb(opt_ref_card_id)`
reuses the same `nn.Embedding` as the state encoder. Already correct.

**Q16 OOV behaviour for unseen card ID?** **Undefined and crashes.**
Confirmed by direct test: `model.card_emb(torch.tensor([999999]))` raises
`IndexError: index out of range in self`, uncaught anywhere in
`agents/il_agent/agent_core.py`. This is a concrete gap against your own
"never crash" mandate — needs an explicit OOV row (e.g. clamp to a reserved
last vocab index) before this agent is trustworthy for submission.

**Q17 History in the input?** None currently (v1 encoder reads only
`current`, never `logs`). Cost of the untaken options is a Phase-1
discussion; flagging that `logs` is real and available (0.1) so "none" is a
choice, not a limitation of the data.

**Q18 [BLOCKING] Majority baseline, global + per-context?** 38.1% global,
full per-context table above (0.5) — measured on the complete train and
eval days, not sampled.

**Q19 Does row count justify model size?** ~718K labeled rows, ~33–48%
distinct (Q6) → roughly 250–350K effective distinct decision states. A
2–15M param budget (your own range) is not obviously overparameterized
against that, but I'd want the real encode-and-count from Phase 2.1 before
committing a specific number — this sampled estimate is directional, not
final.

**Q21 Same metagame both days?** Behaviorally, yes at the aggregate level —
per-context majority baselines are nearly identical train vs eval (0.5
table). I did not build a full archetype/deck-composition histogram (that
needs a card-name/archetype mapping I didn't have time to build in this
pass) — if you want that specific analysis before trusting the held-out
split, say so and I'll run it next.

**Q24 [BLOCKING] Matches needed for Rung 2 to mean anything?** Standard
two-proportion power calc, 5pp difference, α=0.05, 80% power, p≈0.5:
**≈1,570 matches per arm (≈3,100+ total)** for one pairwise comparison. At
the measured 3.98 matches/sec (rule_baseline self-play), that's **~13
minutes** for one comparison — affordable. Real matches against a
meaningfully different opponent may run slower/faster than self-play;
treat this as an order-of-magnitude estimate, not a promise.

**Q27 Inference latency, does batching help?** Measured on the *existing*
checkpoint, forced CPU, fp32, one real decision: **1.57 ms/decision**
(includes 0.05s one-time model load, excluded from the per-decision
figure). Cold-start import (`torch` + `transformers`): **1.92s**. Both are
trivially inside the 600s/2000s budgets — latency is not a concern at this
model size; I did not test the batched-vs-N-forward-passes question since
the current architecture already scores all options in one forward pass
(Pattern B, confirmed in `il_model.py`), so this is already answered by the
existing design, not open.

**Q28 [BLOCKING] Fallback / leaf-evaluator framing?** The existing
`agent_core_improved.py` and `_proto_agent.py` already call
`cg.api.search_begin/search_step` for internal lookahead — there is a live
search-based agent in this repo today, and per your own framing (and the
kiyotah RL+MCTS precedent in `notes/study-kiyotah-rl-mcts.md`) it is a
strong prior that search will keep beating a standalone BC policy here. The
current `PTCGImitationPolicy` already outputs a full softmax over legal
options (not just argmax) — `logits` are returned, not just the chosen
index — so it's already shaped correctly to serve as a leaf-evaluator/prior
input to search without any architecture change. This is worth designing
for explicitly in Phase 1 rather than only building a standalone-agent
benchmark path.

**Q29 Bundle size / checkpoint fraction?** Checkpoint: 4.7 MB. I did not
find a stated bundle size limit for this competition in the repo — flagging
that I don't have this number, not guessing it. The `transformers` package
itself is a much bigger cold-start/bundle cost than the checkpoint (adds
~1.4s to a 1.9s total import time, per the Q27 measurement) — worth
weighing against a hand-rolled encoder purely on bundle/cold-start grounds,
independent of the parameter-count question.

**Q30 DAgger/self-play in scope?** Not decided — explicitly a Phase-1/§F5
question you asked to defer; not answering it here.

*(Q11, 12, 14, 15, 20, 22, 23, 25, 26 need a Phase-1 architecture
conversation to answer meaningfully rather than a Phase-0 measurement — I
have relevant data for several of them (e.g. Q23 reproducibility, Q25
opponent-pool composition) I can pull quickly once we're there rather than
front-loading everything now.)*

---

## What I'd do differently from what's already built

Not asked, but relevant given how much exists already: I would **not**
throw away `il_dataset.py`/`il_model.py`/`scripts/train_il.py` — the core
shape (Pattern-B option scoring, shared card embedding, no leakage, no
gratuitous history) is sound and matches the reasoning your Phase 1 section
is set up to arrive at anyway. The fixes needed are bounded: (1) stop
dropping `minCount==0` declines (Q10), (2) add an OOV embedding row + wrap
inference in try/except (Q16), (3) extract a real `utils/device.py` (2.7),
(4) make the stale-echo dedup explicit rather than emergent, (5) decide and
document the Q1 label-filtering policy given the missing train-day scores.
None of these require a rewrite.

---

## Stopping here per your instructions

Phase 0's six gates and the BLOCKING open questions (Q1 partially — data
gap, not analysis gap; Q2, Q3, Q7, Q18, Q24, Q28) are answered above with
real measurements. I have not written any model code. Waiting for your
review before Phase 1.
