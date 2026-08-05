# Phase 2: What the search/battle API is FOR — measurements

Date: 2026-08-03. Machine: the training laptop (Apple Silicon, native ARM64
cabt). All numbers measured by `scripts/measure_search_api.py` (JSON:
`reports/search_api_measurements_{cpu,mps}.json`) and the corpus scan in this
session. Model in the search loop: the notebook's tiny MyModel (12.5M params,
embedding-dominated), NOT our IL model — IL-prior timings are Phase 3's job.

## a) search_step cost and the SEARCH_COUNT budget

Engine `search_step` (pure C engine, ctypes) is CHEAP:

| config | search_step mean | search_step p95 |
|---|---|---|
| forced CPU, torch 1 thread (evaluator rehearsal) | **0.09–0.14 ms** | 0.2–0.5 ms |
| laptop default (MPS model, 8 threads) | 0.34–0.39 ms | — |

Per-DECISION cost of the full MCTS agent (root `search_begin` + N sims,
each sim = 1 `search_step` + 1 tiny-model eval), forced CPU single-thread:

| SEARCH_COUNT | mean | p95 | max |
|---|---|---|---|
| 5 | 2.6 ms | 3.0 ms | 39.9 ms |
| 10 | 3.7 ms | 4.8 ms | 6.6 ms |
| 20 | 7.7 ms | 11.3 ms | 52.7 ms |
| 40 | 18.2 ms | 31.5 ms | 75.2 ms |

Linear fit: ~1.4 ms fixed + **~0.45 ms per simulation** (of which ~0.1 ms is
the engine; the rest is the tiny model's forward + feature building). On MPS
the same loop is ~15–20x SLOWER per decision (80 ms at N=10): per-move
single-sample inference is a CPU workload; MPS dispatch overhead dominates.

**Budget arithmetic for the ladder (~1 s/turn + overage bank):** with the
tiny model, even N=40 sims costs ~18 ms/decision — hundreds of sims fit.
The binding case is Phase 3's IL prior: at the known ~3 ms/forward CPU cost
of our 3.32M-param policy, each sim costs ~3.2 ms, so a decision at N sims
~= 3 + 3.2·N ms. A turn contains several decisions (MAIN action plus
sub-selects; ~5 is a fair planning number, own-turn decision counts vary
1–15+). Planning envelope:

| N (sims) | per decision | per ~5-decision turn |
|---|---|---|
| 10 | ~35 ms | ~0.18 s |
| 30 | ~100 ms | ~0.5 s |
| 60 | ~195 ms | ~1.0 s |

So **N≈30 is the safe Phase-3 operating point, N≈60 the ceiling** before
eating the overage bank — to be validated by a full forced-CPU game
rehearsal in Phase 3, not assumed. (Unverified until then: real decision
count per turn under the IL-prior agent, and `search_begin` cost for a
determinized 60-card state, which is bundled into the fixed term here.)

## b) Direct battle API vs kaggle_environments — self-play throughput

Same machine, same run, random-vs-random, serial single process:

| path | games/s | decisions (or env steps)/s |
|---|---|---|
| direct battle API (`battle_start/select/finish`) | **62–96** | **4,200–7,200 decisions/s** |
| `kaggle_environments` `env.run` | 4.2–7.5 | 318–606 steps/s |

(Ranges = forced-single-thread vs default-thread runs; the 606 steps/s
env.run figure reproduces the previously measured ~600/s serial.)

**LOUD FINDING: the direct battle API is ~10–15x faster serial than the
env.run path we currently use for ALL self-play.** Our measured Stage-3
economics (~600 steps/s serial, ~3.4K/s at 8 workers, 2026-08-03) are
env.run numbers; the direct API hits ~4–7K decisions/s in ONE process
before any multiprocessing. If Stage-3 PPO rollout collection moves to the
direct API (selfplay.py currently wraps env.run), rollout throughput
improves by roughly an order of magnitude at equal core count. Caveats:
(1) decisions != env steps exactly (env.run adds framework ticks and the
deck-submission step; per-game decision counts are comparable, ~68 vs ~75);
(2) with a real policy in the loop, inference becomes the bottleneck and
the gap narrows toward the inference floor; (3) the direct API bypasses
kaggle_environments' agent-error/timeout semantics — fine for rollouts,
NOT a substitute for pre-submission smoke tests through env.run.

## c) Determinization legality

**PASS — no opponent-private information reaches the agent path.** Verified
against `search_api.py::mcts_agent` / `cg/api.py::search_begin` line by line.
The determinization supplies:

- `your_deck` / `your_prize`: random samples from OUR OWN known 60-card
  list. Sizes read from `deckCount` / `len(prize)` — public counts.
- `opponent_deck/prize/hand`: FIXED placeholders (Snorlax 1072 / Basic
  Energy 1) replicated to the public counts `deckCount`, `len(prize)`,
  `handCount`. No opponent card identity is read from anywhere.
- `opponent_active`: placeholder only when the opponent's active is
  face-down (`active[0] == None` — itself public).

`search_begin` validates COUNT consistency only and requires the caller to
predict all opponent unknowns — the engine does not leak them (it errors if
you omit them). The opaque `search_begin_input` string is engine state we
pass through unread; the API contract (mandatory opponent predictions)
is behavior-consistent with it containing no opponent-private content, but
we have not decoded it — flagged as the one unverified corner.

## Extra: the 64-combination multi-select cap vs the real corpus

Scan of 1,500 Hub episodes (1,000 train / 500 eval days), 226,032 active
decisions:

- Decisions where C(n_options, maxCount) > 64 (enumeration TRUNCATED,
  chosen set possibly not representable): **842 = 0.37%**. Worst case
  C(30,27) = 4,060 (a 27-card DISCARD select). Typical exceeders are
  9–14-option selects with maxCount 3–11.
- Decisions with minCount < maxCount: **14.5%** — the notebook enumerates
  ONLY size-maxCount combinations, so legal smaller selections (including
  "stop early") are NEVER candidates. Our IL agent's DECLINE mechanism
  covers exactly this; the notebook scheme does not. This is a real
  behavioral gap of the sample scaffold, an order of magnitude more common
  than the raw 64-cap overflow.
- maxCount distribution: 94.6% of decisions are single-select (maxCount 1),
  where the cap is irrelevant.

Implication for Phase 3: keeping the notebook's enumeration is acceptable
for a first experiment (99.6% of decisions unaffected by the cap), but the
missing sub-maxCount selections mean the search scaffold can be forced into
over-picking; if IL-prior-MCTS underperforms plain il_agent specifically on
multi-select decisions, this is the first suspect.

## Phase-2 gate verdict

Search at inference is AFFORDABLE (a: even 60 IL-prior sims fit ~1 s/turn on
paper), the battle API is a big rollout win (b), and the determinization is
legal (c). Phase 3 proceeds.
